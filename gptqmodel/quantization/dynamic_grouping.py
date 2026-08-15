# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

# Based on Binary Quantization For LLMs Through Dynamic Grouping
# @article{dynq,
#   title={Binary Quantization For LLMs Through Dynamic Grouping},
#   author={D. Marin, E. Azabou},
#   journal={arXiv preprint arXiv:2509.03054},
#   year={2025}
# }

import torch

from .gar import _supports_stable_argsort, extend_perm_with_tail


def _as_float(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return x.real.float()
    if not x.is_floating_point():
        return x.float()
    return x


def _quantization_error(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-column affine round-to-nearest reconstruction error of `W` at `bits`."""

    Wf = _as_float(W)
    scale = (Wf.max(dim=0).values - Wf.min(dim=0).values).clamp_min(1e-12) / max(2**bits - 1, 1)
    zero = torch.round(-Wf.min(dim=0).values / scale)
    maxq = 2**bits - 1
    q = torch.clamp(torch.round(Wf / scale) + zero, 0, maxq)
    err = Wf - scale * (q - zero)
    return err.pow(2).sum(dim=0)


def build_dynamic_group_perm(
    W: torch.Tensor,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    """DynaQ dynamic grouping: order columns by quantization similarity, then group.

    Columns are scored by their per-column RTN error at the target bitwidth, the
    matrix is reordered most-sensitive-first, and fixed contiguous groups are cut
    from the reordered columns. Columns with similar quantization behavior land
    in the same group, so the shared scale/zero each group must use is far less
    mismatched than in the default index-contiguous grouping — the grouping
    effect DynaQ shows is most useful at very low bitwidths. Any trailing
    columns outside full groups keep their identity order via the GAR tail
    extension.
    """

    if group_size <= 0:
        raise ValueError(f"dynamic grouping requires `group_size > 0`, got `{group_size}`.")

    err = _quantization_error(W, bits)
    if err.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=W.device)

    if _supports_stable_argsort():
        order = torch.argsort(err, descending=True, stable=True)
    else:
        order = torch.argsort(err, descending=True)
    order = order.to(device=W.device, dtype=torch.long)
    return extend_perm_with_tail(order, W.shape[1])
