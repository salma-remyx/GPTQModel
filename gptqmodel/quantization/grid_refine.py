# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fixed-grid discrete refinement of already-quantized integer assignments.

Takes the output of a PTQ pass (GPTQ, RTN, ...) and iteratively revisits the
discrete integer assignments on the *frozen* quantization grid — scale, zero
point, and bitwidth are never touched — so the serialized format is unchanged.

The objective is the same Hessian-weighted layer reconstruction error GPTQ
itself minimizes::

    err(W_hat) = 0.5 * tr((W - W_hat) H (W - W_hat)^T)

For a single weight whose dequantized value moves by ``delta``, the change in
that objective has the closed form::

    d_err = 0.5 * H_ii * delta^2 - delta * ((W - W_hat) H)_ji

Neighbor grid levels are scored with this expression in a vectorized pass over
the whole matrix — no backprop and no Hessian inverse. Only moves that strictly
decrease the objective are accepted, so the loss is monotone by construction
and the procedure terminates.

Adapted from "ReQuant: Fixed-Grid Discrete Refinement for Post-Training
Quantization" (arXiv:2608.07019). The paper's standalone sweep harness is
replaced by an in-process refinement against the Hessian GPTQModel already
accumulated for the layer.
"""

import math

import torch

from .config import GridRefineConfig


def _dense_group_params(param: torch.Tensor, columns: int, group_size: int) -> torch.Tensor:
    """Expand per-group ``param`` of shape (rows, groups) to (rows, columns)."""

    if param.dim() == 1:
        param = param.unsqueeze(0)
    dense = param.repeat_interleave(group_size, dim=1)[:, :columns]
    if dense.shape[1] < columns:
        # tail columns past the last full group reuse the final group's params
        tail = param[:, -1].unsqueeze(1).expand(-1, columns - dense.shape[1])
        dense = torch.cat([dense, tail], dim=1)
    return dense


def _hessian_loss(W: torch.Tensor, Q: torch.Tensor, H: torch.Tensor) -> float:
    E = W - Q
    return 0.5 * torch.einsum("ij,jk,ik->", E, H, E).item()


def refine(
    W: torch.Tensor,
    H: torch.Tensor,
    Q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
    group_size: int,
    cfg: GridRefineConfig,
    groupwise: bool = False,
):
    """Refine the integer assignments behind ``Q`` on the frozen grid.

    Args:
        W: (rows, columns) original float weights, in the same (possibly
            act-order permuted) column order the quantization loop used.
        H: (columns, columns) Hessian matching ``W``'s column order.
        Q: (rows, columns) dequantized weights produced by the PTQ pass; the
            refinement's starting point is recovered from these values.
        scale: per-group scales, (rows, groups) or (rows, 1).
        zero: per-group zero points, same layout as ``scale``.
        maxq: grid bound (``2**bits - 1``).
        group_size: columns per group; ``-1`` means one group per row.
        cfg: refinement hyperparameters (sweeps, neighbor radius).
        groupwise: grid is ``scale * clamp(round(w/scale), -maxq, maxq)``
            (symmetric integer range, implicit zero) instead of the affine
            ``scale * (clamp(round(w/scale) + zero, 0, maxq) - zero)`` form.

    Returns:
        ``(Q_refined, loss_before, loss_after)``.
    """
    rows, columns = W.shape
    if group_size == -1:
        group_size = columns

    scale_dense = _dense_group_params(scale, columns, group_size).to(W.dtype)
    if groupwise:
        zero_dense = torch.zeros_like(scale_dense)
    else:
        zero_dense = _dense_group_params(zero, columns, group_size).to(W.dtype)

    # Recover the initializer's integer assignments from the dequantized
    # weights; rounding only repairs float noise, it does not re-quantize.
    if groupwise:
        q = torch.clamp(torch.round(Q / scale_dense), -maxq, maxq)
    else:
        q = torch.clamp(torch.round(Q / scale_dense + zero_dense), 0, maxq)

    def _dequant(q_int: torch.Tensor) -> torch.Tensor:
        if groupwise:
            return scale_dense * q_int
        return scale_dense * (q_int - zero_dense)

    Q_cur = _dequant(q)
    loss_cur = _hessian_loss(W, Q_cur, H)
    loss_before = loss_cur

    diag = torch.diagonal(H)
    # Columns with a zero Hessian diagonal carry no reconstruction signal.
    active = diag > 0
    if not bool(active.any()) or cfg.sweeps <= 0 or cfg.radius <= 0:
        return Q_cur, loss_before, loss_cur

    radius = min(int(cfg.radius), maxq)
    lo = -maxq if groupwise else 0
    levels = torch.arange(lo, maxq + 1, device=W.device, dtype=W.dtype).view(1, 1, -1)
    n_levels = levels.shape[2]

    # One move is applied per iteration; `sweeps` bounds the iteration count.
    # Each accepted move's closed-form delta is exact, so the objective
    # decreases strictly and the procedure stops once no candidate improves it.
    for _ in range(int(cfg.sweeps)):
        d = (W - Q_cur) @ H
        dist = levels - q.unsqueeze(-1).to(W.dtype)  # levels away from current
        step = scale_dense.unsqueeze(-1) * dist
        delta = 0.5 * diag.view(1, -1, 1) * step.square() - step * d.unsqueeze(-1)
        # Only levels within `radius` of the current assignment, and only
        # columns carrying Hessian signal, are candidates.
        delta = delta.masked_fill((dist.abs() > radius) | (dist == 0) | ~active.view(1, -1, 1), torch.inf)

        flat = int(torch.argmin(delta))
        best = float(delta.flatten()[flat].item())
        if not math.isfinite(best) or best >= 0:
            break

        j = flat // (columns * n_levels)
        i = (flat // n_levels) % columns
        q[j, i] = levels.flatten()[flat % n_levels].to(q.dtype)
        Q_cur = _dequant(q)

    return Q_cur, loss_before, _hessian_loss(W, Q_cur, H)


__all__ = ["refine"]
