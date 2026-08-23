# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.eora.eora import eora_compute_lora
from gptqmodel.eora.truncated_solver import DEFAULT_RANK_TOL, truncated_eigen_factors


def _rank_deficient_gram(n: int, true_rank: int, samples: int = 512) -> torch.Tensor:
    """Builds an EoRA-style activation Gram matrix with a collapsed null space."""

    acts = torch.randn(samples, n)
    gram = acts.T @ acts / samples
    gram[true_rank:, :] = 0
    gram[:, true_rank:] = 0
    return gram


def test_truncated_factors_drop_collapsed_directions():
    gram = _rank_deficient_gram(n=8, true_rank=4)

    factors = truncated_eigen_factors(gram)

    assert factors.num_input == 8
    assert factors.num_kept == 4
    assert factors.num_dropped == 4
    assert factors.forward.shape == (8, 4)
    assert factors.inverse.shape == (8, 4)
    # The whitening must stay finite and bounded: the whole point is that no
    # direction is inverted through a collapsed eigenvalue.
    assert torch.isfinite(factors.inverse).all()
    assert float(factors.inverse.abs().max()) < 1e4

    # forward/inverse must be mutual pseudo-inverses on the kept subspace.
    pinv = factors.inverse.T @ factors.forward
    torch.testing.assert_close(pinv, torch.eye(4, dtype=pinv.dtype))


def test_eora_compute_lora_survives_rank_deficient_gram():
    """Exact-zero eigenvalues used to produce an all-NaN adapter.

    The pre-existing negative-eigenvalue clamp never fires here because the
    collapsed eigenvalues are exactly zero, so ``1/sqrt(0)`` propagated NaN
    straight into the returned A factor.
    """

    n, m, rank = 8, 3, 2
    gram = _rank_deficient_gram(n=n, true_rank=4)
    residual = (torch.randn(m, n, dtype=torch.float64) @ gram.double()).float()

    A, B = eora_compute_lora(
        w_wq_delta=residual,
        name="test.layer.0.mlp",
        eigen_scaling_diag_matrix=gram,
        rank=rank,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    assert A.shape == (rank, n)
    assert B.shape == (m, rank)
    assert A.dtype == B.dtype == torch.float16
    assert torch.isfinite(A.float()).all()
    assert torch.isfinite(B.float()).all()


def test_eora_compute_lora_recovers_signal_on_kept_subspace():
    """The adapter must still approximate the residual, not just stay finite."""

    n, m, rank = 8, 3, 2
    gram = _rank_deficient_gram(n=n, true_rank=4)
    residual = (torch.randn(m, n, dtype=torch.float64) @ gram.double()).float()

    A, B = eora_compute_lora(
        w_wq_delta=residual,
        name="test.layer.1.mlp",
        eigen_scaling_diag_matrix=gram,
        rank=rank,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    reconstructed = B.float() @ A.float()

    u, s, v = torch.linalg.svd(residual.double(), full_matrices=False)
    best = u[:, :rank] @ torch.diag(s[:rank]) @ v[:rank, :]

    # Truncation keeps the recoverable part; the residual lives in the kept
    # subspace by construction, so we should sit near the optimal rank-2 fit.
    # The slack covers float16 rounding of A/B plus the eigenspace reweighting.
    rel_err = float((reconstructed.double() - best).norm() / best.norm())
    assert rel_err < 0.1, f"truncated solve rel err {rel_err:.4f} vs optimal rank-{rank} fit"


def test_eora_compute_lora_full_rank_gram_matches_optimal_rank_fit():
    n, m, rank = 8, 3, 2
    acts = torch.randn(512, n)
    gram = (acts.T @ acts / 512).float()
    residual = torch.randn(m, n, dtype=torch.float32)

    A, B = eora_compute_lora(
        w_wq_delta=residual,
        name="test.layer.2.mlp",
        eigen_scaling_diag_matrix=gram,
        rank=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    reconstructed = B @ A
    u, s, v = torch.linalg.svd(residual.double(), full_matrices=False)
    best = u[:, :rank] @ torch.diag(s[:rank]) @ v[:rank, :]

    # A full-rank well-conditioned Gram is unchanged in spirit: the solve still
    # lands near the unconstrained best rank-`rank` approximation.
    rel_err = float((reconstructed.double() - best).norm() / best.norm())
    assert rel_err < 0.25, f"full-rank solve rel err {rel_err:.4f}"


def test_truncated_solver_rejects_degenerate_gram():
    with pytest.raises(ValueError):
        truncated_eigen_factors(torch.zeros(4, 4, dtype=torch.float32))

    # Slightly negative-definite noise must be truncated away, not clamped.
    gram = torch.eye(4, dtype=torch.float64) * -1e-12
    gram[0, 0] = 1.0
    factors = truncated_eigen_factors(gram, rel_tol=DEFAULT_RANK_TOL)
    assert factors.num_kept == 1
