# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import HeadRateConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.head_rate import (
    ClassAwareQuantizer,
    class_side_statistic,
    head_rate_summary,
    waterfill,
)
from gptqmodel.quantization.quantizer import Quantizer


def _head_rate_qcfg(**rate_kwargs) -> QuantizeConfig:
    """Build a config with lm_head rate allocation enabled."""

    return QuantizeConfig(
        bits=3,
        group_size=-1,
        sym=True,
        act_group_aware=False,
        lm_head=True,
        lm_head_rate=rate_kwargs or None,
    )


def _zipf_statistic(classes: int) -> torch.Tensor:
    """Zipfian class-side statistic, as calibration on natural text produces."""

    ranks = torch.arange(classes, dtype=torch.float32)
    return (1.0 / (ranks + 1) ** 1.4) * (0.7 + 0.6 * torch.rand(classes))


def test_waterfill_hits_budget_and_ranks_classes():
    statistic = _zipf_statistic(64)

    multipliers = waterfill(statistic, budget=0.75)

    assert multipliers.numel() == 64
    assert multipliers.mean().item() == pytest.approx(0.75, abs=1e-3)
    assert (multipliers >= 0).all()
    # sensitive (frequent) classes get more rate than rare ones
    order = torch.argsort(statistic, descending=True)
    assert multipliers[order[0]] > multipliers[order[-1]]
    assert multipliers[order[0]].item() == multipliers.max().item()


def test_waterfill_budget_one_keeps_mean_on_budget():
    multipliers = waterfill(_zipf_statistic(32), budget=1.0)

    assert multipliers.mean().item() == pytest.approx(1.0, abs=1e-3)


def test_class_side_statistic_tracks_token_frequency():
    # class 0 is the argmax on every token, so it carries most of the softmax
    # mass and therefore most of the curvature.
    logits = torch.full((16, 8), -5.0)
    logits[:, 0] = 5.0

    statistic = class_side_statistic(logits)

    assert statistic.shape == (8,)
    assert statistic[0] > 0.0
    # the winner holds several times the curvature of any loser class
    assert statistic[0] > 5.0 * statistic[1:].max()


def test_gptq_accumulates_class_statistic_from_calibration_outputs():
    layer = nn.Linear(32, 16, bias=False, dtype=torch.float32)
    gptq = GPTQ(layer, qcfg=_head_rate_qcfg(budget=0.75, temperature=1.0))
    gptq.quantizer.configure(perchannel=True)

    activations = torch.randn(2, 8, 32)
    logits = torch.log_softmax(torch.randn(2, 8, 16) * 3.0, dim=-1)
    gptq.add_batch(activations, logits)

    assert gptq._class_statistic is not None
    assert gptq._class_statistic.shape == (16,)

    gptq.add_batch(activations, logits)
    # the second batch folds into the same accumulator
    assert gptq._class_statistic.shape == (16,)


def test_gptq_without_lm_head_rate_skips_statistic():
    layer = nn.Linear(32, 16, bias=False, dtype=torch.float32)
    gptq = GPTQ(layer, qcfg=QuantizeConfig(bits=3, group_size=-1, act_group_aware=False))

    gptq.add_batch(torch.randn(2, 8, 32), torch.randn(2, 8, 16))

    assert gptq._class_statistic is None


def test_apply_class_rate_allocation_swaps_quantizer():
    layer = nn.Linear(32, 16, bias=False, dtype=torch.float32)
    gptq = GPTQ(layer, qcfg=_head_rate_qcfg(budget=0.75))
    gptq.quantizer.configure(perchannel=True)
    gptq._class_statistic = _zipf_statistic(16)

    gptq.apply_class_rate_allocation()

    assert isinstance(gptq.quantizer, ClassAwareQuantizer)
    assert gptq.quantizer.class_multipliers.numel() == 16
    assert gptq.quantizer.class_multipliers.mean().item() == pytest.approx(0.75, abs=1e-3)


def test_apply_class_rate_allocation_ignores_mismatched_statistic():
    layer = nn.Linear(32, 16, bias=False, dtype=torch.float32)
    gptq = GPTQ(layer, qcfg=_head_rate_qcfg(budget=0.75))
    gptq.quantizer.configure(perchannel=True)
    gptq._class_statistic = _zipf_statistic(99)  # wrong class count

    gptq.apply_class_rate_allocation()

    # falls back to the base quantizer rather than corrupting the run
    assert type(gptq.quantizer) is Quantizer


def test_class_aware_grid_shifts_error_toward_insensitive_classes():
    """The allocation is only useful if it actually moves the error mass.

    A multiplier above 1 keeps a row's full dynamic range (it is already at
    the no-clip limit), so sensitive rows are left untouched while the rate
    they free up is taken from insensitive rows, whose grids get coarser.
    """

    torch.manual_seed(0)
    layer = nn.Linear(64, 32, bias=False, dtype=torch.float32)
    statistic = _zipf_statistic(32)
    order = torch.argsort(statistic, descending=True)
    sensitive = order[:4]
    insensitive = order[-4:]

    def _row_error(quantizer) -> torch.Tensor:
        weight = layer.weight.data.clone().float()
        quantizer.find_params(weight, weight=True)
        return (quantizer.quantize(weight) - weight).norm(dim=1)

    base = Quantizer(_head_rate_qcfg())
    base.configure(perchannel=True)
    base_error = _row_error(base)

    aware = ClassAwareQuantizer(_head_rate_qcfg())
    aware.configure(perchannel=True)
    aware.set_class_multipliers(waterfill(statistic, budget=0.75))
    aware_error = _row_error(aware)

    # sensitive rows keep the base grid: the allocator never spends range on a
    # row that already has all of it.
    torch.testing.assert_close(aware_error[sensitive], base_error[sensitive])
    # insensitive rows absorb the reduction, so their error grows.
    assert aware_error[insensitive].mean() > base_error[insensitive].mean()
    assert aware_error[insensitive].mean() > 2.0 * base_error[insensitive].mean()


def test_head_rate_summary_reports_zipf_concentration():
    statistic = _zipf_statistic(100)
    multipliers = waterfill(statistic, budget=0.75)

    summary = head_rate_summary(statistic, multipliers)

    assert summary["classes"] == 100.0
    # a Zipf(1.4) distribution puts half its mass in a small minority of classes
    assert summary["mass_top50pct_classes"] < 25.0
    assert summary["multiplier_max"] > summary["multiplier_min"]


def test_lm_head_rate_config_requires_lm_head_flag(caplog):
    config = QuantizeConfig(bits=3, group_size=-1, act_group_aware=False, lm_head_rate={"budget": 0.75})

    assert config.lm_head_rate is None


def test_lm_head_rate_config_rejects_invalid_budget():
    with pytest.raises(ValueError, match="budget"):
        HeadRateConfig(budget=0.0)

    with pytest.raises(ValueError, match="temperature"):
        HeadRateConfig(budget=1.0, temperature=0.0)


def test_lm_head_rate_serializes_into_meta():
    config = _head_rate_qcfg(budget=0.8, temperature=1.5)

    payload = config.to_dict()

    assert payload["meta"]["lm_head_rate"] == {"budget": 0.8, "temperature": 1.5}
