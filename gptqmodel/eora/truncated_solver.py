# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

# QuaSAR
# @article{quasar,
#   title={QuaSAR: Quantization Compensation via Stable Activation-Aware Rank Truncation},
#   journal={arXiv preprint arXiv:2608.14149},
#   year={2026}
# }
#
# Adapted for GPT-QModel: the paper's parameter-free truncated pseudoinverse is
# applied to the EoRA Gram matrix (eigen_scaling_diag_matrix) so rank-deficient
# activation statistics no longer destabilize the closed-form compensation solve.

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from ..utils.logger import setup_logger

log = setup_logger()

# Directions below this fraction of the leading eigenvalue are treated as
# numerically collapsed and dropped before inversion. Relative (not absolute) so
# the same default works across layers with very different activation scales.
DEFAULT_RANK_TOL = 1e-6


class TruncatedFactors(NamedTuple):
    """Whitening factors of a truncated eigendecomposition.

    ``forward`` maps activation space into the kept (well-conditioned)
    subspace; ``inverse`` maps back. Both drop collapsed directions, so
    ``inverse @ forward`` is a rank-``num_kept`` pseudoinverse rather than
    an unstable full inverse.
    """

    forward: Tensor
    inverse: Tensor
    eigenvalues: Tensor
    num_input: int

    @property
    def num_kept(self) -> int:
        """Number of non-collapsed directions retained by truncation."""

        return int(self.eigenvalues.numel())

    @property
    def num_dropped(self) -> int:
        """Number of collapsed directions removed before inversion."""

        return self.num_input - self.num_kept


def truncated_eigen_factors(
        gram_matrix: Tensor,
        rel_tol: float = DEFAULT_RANK_TOL,
) -> TruncatedFactors:
    """Eigendecompose ``gram_matrix`` keeping only well-conditioned directions.

    Collapsed eigenvalues (numerically zero or negative) are discarded before
    the reciprocal-sqrt whitening is formed, instead of being clamped up to the
    smallest surviving eigenvalue. Clamping amplifies noise directions by
    ``1 / sqrt(lambda_min)`` and can push the downstream SVD to non-finite
    values; truncation removes those directions entirely.
    """

    assert gram_matrix.ndim == 2, f"Gram matrix must be 2D, actual = `{gram_matrix.ndim}`"
    assert gram_matrix.shape[0] == gram_matrix.shape[1], (
        f"Gram matrix must be square, actual = `{gram_matrix.shape}`")

    n = gram_matrix.shape[0]
    scaled = gram_matrix.to(dtype=torch.float64)

    eigenvalues, eigenvectors = torch.linalg.eigh(scaled)

    # eigh is ascending, so the collapsed tail is dropped and `keep_idx` is
    # monotonically increasing for the index_copy_ placement below.
    scale_ref = float(eigenvalues.abs().max())
    if scale_ref == 0.0:
        raise ValueError(
            "EoRA truncated solver: activation Gram matrix is entirely zero; "
            "no direction carries signal to compensate against."
        )

    keep_idx = (eigenvalues > rel_tol * scale_ref * n).nonzero(as_tuple=True)[0]

    if keep_idx.numel() == 0:
        # Only reachable when every eigenvalue is non-positive noise.
        raise ValueError(
            "EoRA truncated solver: no positive eigenvalue survived rank truncation; "
            "please increase your calibration data set for EoRA."
        )

    kept = eigenvalues.index_select(0, keep_idx)
    kept_vectors = eigenvectors.index_select(1, keep_idx)

    sqrt_kept = torch.sqrt(kept)
    forward = kept_vectors * sqrt_kept.unsqueeze(0)          # Q diag(sqrt(L))
    inverse = kept_vectors / sqrt_kept.unsqueeze(0)          # Q diag(1/sqrt(L))

    return TruncatedFactors(
        forward=forward,
        inverse=inverse,
        eigenvalues=kept,
        num_input=n,
    )


def truncated_solve_lowrank(
        residual: Tensor,
        gram_matrix: Tensor,
        rank: int,
        rel_tol: float = DEFAULT_RANK_TOL,
        name: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Solve the EoRA eigenspace low-rank objective on the truncated subspace.

    Returns ``(A, B)`` with ``B @ A ~= residual`` restricted to the
    well-conditioned activation subspace, matching the contract of
    ``eora_compute_lora``: ``A`` is ``[rank, in_features]`` and ``B`` is
    ``[out_features, rank]``.
    """

    factors = truncated_eigen_factors(gram_matrix, rel_tol=rel_tol)

    if factors.num_dropped > 0 and name is not None:
        log.info.once(
            f"EoRA truncated solver: dropped {factors.num_dropped}/{factors.num_input} collapsed "
            f"activation directions in `{name}` before inversion."
        )

    whitened = torch.matmul(residual.to(dtype=torch.float64), factors.forward)
    u, s, v = torch.linalg.svd(whitened, full_matrices=False)

    lowrank_r = min(int(rank), s.numel())
    if lowrank_r < int(rank) and name is not None:
        log.info.once(
            f"EoRA truncated solver: requested rank {rank} exceeds the {s.numel()} usable "
            f"directions in `{name}`; capping to {lowrank_r}."
        )

    sqrt_sigma = torch.sqrt(torch.diag(s[:lowrank_r]))

    truc_u = u[:, :lowrank_r]
    # `inverse.T` is [num_kept, num_input], so it maps the kept-subspace rows
    # back to full activation space in one matmul, leaving collapsed columns
    # exactly zero rather than amplifying them through a tiny eigenvalue.
    truc_v = torch.matmul(v[:lowrank_r, :], factors.inverse.T)

    B = torch.matmul(truc_u, sqrt_sigma)
    A = torch.matmul(sqrt_sigma, truc_v)

    return A, B
