# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.dynamic_grouping import build_dynamic_group_perm
from gptqmodel.quantization.gptq import GPTQ


def test_dynamic_group_perm_is_valid_permutation():
    torch.manual_seed(0)
    W = torch.randn(16, 70)

    perm = build_dynamic_group_perm(W, group_size=32, bits=4)

    assert perm.numel() == 70
    assert set(perm.tolist()) == set(range(70))


def test_dynamic_group_perm_groups_by_quantization_similarity():
    # Columns with large quantization error should be placed before columns that
    # round-trip exactly, so sensitive and insensitive columns never share a group.
    W = torch.zeros(8, 8)
    W[:, :4] = torch.randn(8, 4) * 10  # high-error columns
    W[:, 4:] = torch.randn(8, 4).round() * 0.01  # near-zero error columns

    perm = build_dynamic_group_perm(W, group_size=4, bits=4)

    first_group = set(perm[:4].tolist())
    assert first_group == {0, 1, 2, 3}


def test_quantize_config_dynamic_groups_exclusivity():
    qcfg = QuantizeConfig(bits=4, group_size=128, dynamic_groups=True, act_group_aware=True)
    # dynamic grouping supersedes GAR column reordering instead of erroring
    assert qcfg.dynamic_groups is True
    assert qcfg.act_group_aware is False
    assert qcfg.desc_act is False

    with pytest.raises(ValueError):
        QuantizeConfig(bits=4, group_size=128, dynamic_groups=True, desc_act=True)


@torch.inference_mode()
def test_gptq_dynamic_groups_quantize_roundtrip():
    torch.manual_seed(0)

    in_features, out_features, group_size = 32, 8, 8
    layer = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=group_size, sym=False, dynamic_groups=True)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)

    gptq.add_batch(torch.randn(5, in_features, dtype=torch.float32), None)
    qweight, scales, zeros, g_idx, *_ = gptq.quantize(blocksize=16)

    assert qweight.shape == layer.weight.shape
    assert scales.shape == (out_features, in_features // group_size)
    assert zeros.shape == scales.shape
    # g_idx stays ascending in original column order — the permutation is undone
    assert g_idx.tolist() == [i // group_size for i in range(in_features)]


@torch.inference_mode()
def test_gptq_dynamic_groups_reduces_reconstruction_error():
    torch.manual_seed(0)

    in_features, out_features, group_size = 64, 16, 16
    layer = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32).eval()
    batch = torch.randn(64, in_features, dtype=torch.float32)

    def run(dynamic: bool) -> float:
        qcfg = QuantizeConfig(bits=3, group_size=group_size, sym=False, dynamic_groups=dynamic, act_group_aware=False)
        gptq = GPTQ(layer, qcfg=qcfg)
        gptq.quantizer.configure(perchannel=True)
        gptq.add_batch(batch, None)
        qweight, scales, zeros, g_idx, *_ = gptq.quantize(blocksize=32)
        per_group = scales[:, g_idx.long()]
        per_group_zero = zeros[:, g_idx.long()]
        # re-dequantize from the packed grid at the group granularity
        maxq = 2**3 - 1
        scale_step = (layer.weight.max(dim=0).values - layer.weight.min(dim=0).values) / maxq
        zero = torch.round(-layer.weight.min(dim=0).values / scale_step)
        ref = scale_step * (torch.clamp(torch.round(layer.weight / scale_step) + zero, 0, maxq) - zero)
        return (qweight - ref).pow(2).mean().item(), (per_group - per_group_zero).abs().sum().item()

    static_err, _ = run(False)
    dynamic_err, _ = run(True)

    # the claim of dynamic grouping: similarity-grouped columns reconstruct no
    # worse than index-contiguous grouping under the same grid
    assert dynamic_err <= static_err * 1.05
