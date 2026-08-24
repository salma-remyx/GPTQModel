# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Class-aware rate allocation for softmax-head quantization.

Adapted from *SoftWater: Class-Aware Rate Allocation for Softmax Quantization*
(arXiv:2608.12026). The paper poses lm_head quantization as a rate-distortion
problem under the KL divergence between the original and quantized output
distributions. A second-order expansion of that KL shows the distortion of a
weight perturbation is weighted jointly by

    J_k(E) ~= G_k * ||E_k||_Sigma^2 ,   Sigma = 2*E_xx^T

where ``G_k`` is the class-side curvature of class ``k`` and ``Sigma`` is the
feature-side (input) covariance. The ``Kn x Kn`` joint metric is separable:
one input-side statistic plus a per-class rescaling.

That separable geometry has a direct consequence for a uniform-grid quantizer.
Because the per-row step ``s_k = range_k / maxq`` is set by the row's own
dynamic range, the quantization error of row ``k`` is proportional to
``s_k * ||E_k||_Sigma`` -- so rows with large ``G_k`` (frequent classes, high
softmax curvature) dominate the output KL and deserve a finer relative grid,
while rare classes tolerate a coarser one. Token frequencies are strongly
Zipfian, so the spread of ``G_k`` is large and the allocation matters.

This module spends a *rate budget* on that spread: waterfilling over per-class
scale multipliers that hits a target mean, then a ``Quantizer`` subclass that
applies the per-class grid.

Substitutions relative to the paper (target-native auxiliaries):

- The lattice / successive-interference-cancellation encoder and its
  ``Kn x Kn`` Cholesky are replaced by GPTQ's existing per-column error
  feedback, which already performs the input-side whitening. Only the
  class-side rescaling is added here.
- The learned rate allocator is replaced by closed-form waterfilling on the
  same class-side statistic, which is the paper's waterfilling core.
"""

from typing import Dict, Optional

import torch

from ..utils.logger import setup_logger
from .config import BaseQuantizeConfig
from .quantizer import Quantizer

log = setup_logger()


def class_side_statistic(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return the per-class curvature ``G_k`` from one batch of teacher logits.

    For a softmax over logits ``z`` the second-order sensitivity of
    ``KL(p || p + delta)`` to a perturbation of row ``k`` is, to leading order,
    proportional to ``p_k * (1 - p_k)`` scaled by the feature covariance. We
    accumulate ``sum_t p_kt * (1 - p_kt)`` over calibration tokens: frequent
    classes accumulate more mass, and classes that are never competitive
    contribute almost nothing. Both effects are the ones the paper's curvature
    term captures, and both come from a single forward pass.
    """

    z = logits.detach().to(dtype=torch.float32)
    if z.dim() > 2:
        z = z.reshape(-1, z.shape[-1])
    if temperature <= 0:
        raise ValueError(f"head_rate: `temperature` must be > 0, got `{temperature}`.")

    probs = torch.softmax(z / temperature, dim=-1)
    # p * (1 - p) is the softmax second derivative w.r.t. its own logit.
    curvature = probs * (1.0 - probs)
    return curvature.sum(dim=0)


def waterfill(
    weights: torch.Tensor,
    budget: float,
    floor: float = 1e-6,
) -> torch.Tensor:
    """Allocate a fixed rate budget across classes by waterfilling.

    Given non-negative class weights ``w_k`` (the class-side statistic) and a
    target average multiplier ``budget`` (mean of ``m_k``), return multipliers
    ``m >= floor`` with ``mean(m) == budget`` that are increasing in ``w``.

    A multiplier is a share of the row's dynamic range (see
    :class:`ClassAwareQuantizer`), so a larger ``m`` is more rate spent on that
    class. Levelling the marginal distortion across classes is the KKT
    condition of the paper's rate allocation; here it is solved by bisection
    on a water level ``L`` with ``m_k = w_k / L``, which is monotone in ``L``
    and therefore converges to the budget.
    """

    if budget <= 0:
        raise ValueError(f"head_rate: `budget` must be > 0, got `{budget}`.")
    w = weights.detach().to(dtype=torch.float32).clamp_min(0.0)
    if w.numel() == 0:
        return w.clone()

    # Dead classes get the floor rate; the rest share the budget.
    active = w > 0
    m = torch.full_like(w, float(floor))
    if not bool(active.any()):
        # No curvature signal at all: fall back to a flat allocation.
        return torch.full_like(w, float(budget))

    w_active = w[active]
    n_active = w_active.numel()
    n_total = w.numel()
    # Reserve the floor mass spent on dead classes so the mean stays on budget.
    active_budget = budget * n_total - floor * (n_total - n_active)
    if active_budget <= floor * n_active:
        return torch.full_like(w, float(budget))

    # Bisect on the water level: m_k = w_k / L for classes above it, and the
    # floor for classes below it. Raising L lowers every m_k, so mean(m) is
    # monotone decreasing in L and bisection converges.
    low, high = 0.0, float(w_active.max()) / floor

    def _mean_at(level: float) -> float:
        m_k = (w_active / max(level, 1e-12)).clamp(min=floor)
        dead_mass = floor * (n_total - n_active)
        return float((m_k.sum() + dead_mass) / n_total)

    if _mean_at(high) > budget:
        # Even at the deepest water level the budget is not reachable; every
        # active class sits at the floor. Spend the surplus uniformly instead.
        m[active] = active_budget / n_active
        return m

    for _ in range(64):
        mid = 0.5 * (low + high)
        if _mean_at(mid) > budget:
            low = mid
        else:
            high = mid

    m[active] = (w_active / max(0.5 * (low + high), 1e-12)).clamp(min=floor)
    return m


class ClassAwareQuantizer(Quantizer):
    """``Quantizer`` whose per-row dynamic range follows the class rate.

    For a fixed integer bitwidth, a row's step size is ``range_k / maxq``, so
    the only way to make one row's grid finer without clipping is to shrink
    the range it covers. This quantizer therefore allocates *range*: it
    pre-clips each row's weights to ``m_k`` times the row's own max magnitude,
    then lets the parent's search run on the clipped row.

    Rows the allocator flagged sensitive (``m_k > 1``) keep more of their
    dynamic range; rows it flagged cheap are clipped harder, which shrinks
    their step and their covered range together. Because the parent's search
    still emits exactly one scale and zero per row, the packed checkpoint
    layout is unchanged -- only the error distribution across classes moves.
    """

    def __init__(self, qcfg: BaseQuantizeConfig, shape=1, name: str = None):
        super().__init__(qcfg=qcfg, shape=shape, name=name)
        self.class_multipliers: Optional[torch.Tensor] = None

    def set_class_multipliers(self, multipliers: torch.Tensor) -> None:
        """Attach per-row range multipliers; one entry per output row."""

        self.class_multipliers = multipliers.detach().to(dtype=torch.float32)

    def _range_limits(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-row magnitude limits implied by the class multipliers."""

        if self.class_multipliers is None:
            return None

        rows = weight.shape[0]
        if self.class_multipliers.numel() != rows:
            log.warn(
                "head_rate: class multipliers (%s) do not match output rows (%s); skipping.",
                self.class_multipliers.numel(),
                rows,
            )
            return None

        peak = weight.detach().to(dtype=torch.float32).abs().amax(dim=1)
        # m <= 1 clips the row harder; m > 1 keeps the row's full range
        # (the row's own peak is already the no-clip limit).
        return peak * self.class_multipliers.to(device=peak.device).clamp(max=1.0)

    def find_params(self, x, weight=False):
        limits = self._range_limits(x) if weight else None
        if limits is not None:
            x = x.detach().to(dtype=torch.float32).clamp(
                min=-limits.unsqueeze(1), max=limits.unsqueeze(1)
            ).to(x.dtype)
        super().find_params(x, weight=weight)


def floor_multiplier() -> float:
    """Smallest admissible class multiplier (a row can never be fully erased)."""

    return 1e-3


def build_class_aware_quantizer(
    quantizer: Quantizer,
    class_statistic: torch.Tensor,
    output_rows: int,
    budget: float,
) -> Optional[ClassAwareQuantizer]:
    """Replace ``quantizer`` with a class-aware one, or return None.

    Returns ``None`` (caller keeps its original quantizer) when the statistic
    does not line up with the module's output rows, so a mis-sized statistic
    degrades to plain GPTQ rather than corrupting the run.
    """

    statistic = class_statistic.detach().to(dtype=torch.float32).flatten()
    if statistic.numel() != output_rows or output_rows == 0:
        return None
    if not bool(torch.isfinite(statistic).all()):
        return None

    multipliers = waterfill(statistic, budget=budget)
    if not bool(torch.isfinite(multipliers).all()):
        return None

    replacement = ClassAwareQuantizer(qcfg=quantizer.qcfg, name=quantizer.name)
    replacement.perchannel = getattr(quantizer, "perchannel", True)
    replacement.grid = getattr(quantizer, "grid", 100)
    replacement.maxshrink = getattr(quantizer, "maxshrink", 0.8)
    replacement.maxq = quantizer.maxq.detach().clone()
    replacement.set_class_multipliers(multipliers)
    return replacement


def head_rate_summary(
    class_statistic: torch.Tensor,
    multipliers: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Return spread statistics describing one class-aware allocation."""

    stat = class_statistic.detach().to(dtype=torch.float32).flatten()
    total = float(stat.sum())
    summary: Dict[str, float] = {"classes": float(stat.numel()), "statistic_total": total}
    if stat.numel() == 0 or total <= 0.0:
        return summary

    probs = stat / total
    sorted_probs, _ = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=0)
    top1 = int((cum < 0.5).sum().item()) + 1
    summary["mass_top50pct_classes"] = float(top1)
    summary["share_top1pct_classes"] = float(
        sorted_probs[: max(1, stat.numel() // 100)].sum().item()
    )
    if multipliers is not None:
        m = multipliers.detach().to(dtype=torch.float32).flatten()
        if m.numel() == stat.numel():
            summary["multiplier_min"] = float(m.min())
            summary["multiplier_max"] = float(m.max())
            summary["multiplier_mean"] = float(m.mean())
    return summary
