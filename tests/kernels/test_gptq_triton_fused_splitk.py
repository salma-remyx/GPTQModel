# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.nn_modules.triton_utils.fused_splitk import (
    fused_dequant_matmul,
    split_k_iters_for,
)


# Dequant reference mirrors the GPTQ-V2 code path: scales[g_idx] * (w - z).
TOLERANCE = {torch.float16: 5e-2, torch.bfloat16: 1e-1}


def _pack_gptq_v2(codes: torch.Tensor, bits: int, along_rows: bool = True) -> torch.Tensor:
    """Pack row-major int codes into int32 words.

    ``qweight`` packs along the in-features (leading) dim; ``qzeros`` packs
    along out-features, mirroring the GPTQ-V2 buffer shapes.
    """
    pack_factor = 32 // bits
    words = torch.zeros(
        (codes.shape[0] // pack_factor, codes.shape[1]) if along_rows else (codes.shape[0], codes.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        packed = codes[i::pack_factor] if along_rows else codes[:, i::pack_factor]
        words |= packed.to(torch.int32) << (i * bits)
    return words


def _fill_layer(layer: TritonV2Linear, bits: int, in_features: int, out_features: int, group_size: int):
    num_groups = in_features // group_size
    weight = torch.randint(0, 2**bits, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 2**bits, (num_groups, out_features), dtype=torch.int32)
    scales = (torch.rand(num_groups, out_features, dtype=torch.float16) * 0.5) + 0.75

    layer.qweight.copy_(_pack_gptq_v2(weight, bits).to(layer.qweight.device))
    layer.qzeros.copy_(_pack_gptq_v2(zeros, bits, along_rows=False).to(layer.qzeros.device))
    layer.scales.copy_(scales.to(layer.scales.device))

    g_idx = torch.tensor([k // group_size for k in range(in_features)], dtype=torch.int32)
    layer.g_idx.copy_(g_idx.to(layer.g_idx.device))
    if layer.bias is not None:
        layer.bias.copy_(torch.randn(out_features, dtype=torch.float16).to(layer.bias.device))

    return (
        weight.to(layer.qweight.device),
        zeros.to(layer.qweight.device),
        scales.to(layer.scales.device),
    )


def _reference(x, weight, zeros, scales, g_idx, bias=None):
    dense = scales[g_idx.long()] * (weight - zeros[g_idx.long()])
    out = x @ dense.to(x.dtype)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def _make_layer(bits, in_features, out_features, group_size, bias):
    layer = TritonV2Linear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        register_buffers=True,
    ).cuda()
    layer.post_init()
    layer.eval()
    return layer


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused SplitK kernel")
def test_tritonv2_fused_splitk_forward_matches_dequant_reference(m, monkeypatch):
    """The forward() wiring edit returns the fused result when opted in."""
    pytest.importorskip("triton")
    from gptqmodel.nn_modules.triton_utils import fused_splitk as fused_splitk_mod

    torch.manual_seed(0)
    bits, in_features, out_features, group_size = 4, 512, 512, 128

    layer = _make_layer(bits, in_features, out_features, group_size, bias=True)
    weight, zeros, scales = _fill_layer(layer, bits, in_features, out_features, group_size)

    x = torch.randn(m, in_features, device="cuda", dtype=torch.float16)

    monkeypatch.setattr(fused_splitk_mod, "FUSED_SPLITK_ENABLED", True)
    assert layer._fused_splitk_matmul(x) is not None

    with torch.inference_mode():
        actual = layer(x)

    expected = _reference(x, weight, zeros, scales, layer.g_idx, layer.bias)

    assert actual.shape == (m, out_features)
    abs_diff = (actual.float() - expected.float()).abs()
    tol = TOLERANCE[x.dtype] * max(1.0, expected.float().abs().max().item())
    assert abs_diff.max().item() <= tol, f"m={m} max diff {abs_diff.max().item()}"


@pytest.mark.parametrize("split_k_iters", [1, 4, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused SplitK kernel")
def test_fused_splitk_kernel_matches_dequant_reference(split_k_iters):
    """Every SPLIT_K decomposition reduces the same K range."""
    pytest.importorskip("triton")
    torch.manual_seed(0)
    bits, in_features, out_features, group_size = 4, 512, 512, 128

    layer = _make_layer(bits, in_features, out_features, group_size, bias=False)
    weight, zeros, scales = _fill_layer(layer, bits, in_features, out_features, group_size)

    x = torch.randn(1, in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        actual = fused_dequant_matmul(
            x,
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.g_idx,
            layer.bits,
            layer.pack_dtype_bits,
            layer.maxq,
            split_k_iters=split_k_iters,
        )

    expected = _reference(x, weight, zeros, scales, layer.g_idx)
    abs_diff = (actual.float() - expected.float()).abs()
    tol = TOLERANCE[x.dtype] * max(1.0, expected.float().abs().max().item())
    assert abs_diff.max().item() <= tol, f"split_k_iters={split_k_iters} max diff {abs_diff.max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused SplitK kernel")
def test_tritonv2_fused_splitk_disabled_by_default(monkeypatch):
    """Without the opt-in flag the layer keeps its dequant -> matmul path."""
    pytest.importorskip("triton")
    from gptqmodel.nn_modules.triton_utils import fused_splitk as fused_splitk_mod

    torch.manual_seed(0)
    bits, in_features, out_features, group_size = 4, 512, 512, 128

    layer = _make_layer(bits, in_features, out_features, group_size, bias=False)
    _fill_layer(layer, bits, in_features, out_features, group_size)
    x = torch.randn(1, in_features, device="cuda", dtype=torch.float16)

    monkeypatch.setattr(fused_splitk_mod, "FUSED_SPLITK_ENABLED", False)
    assert layer._fused_splitk_matmul(x) is None

    with torch.inference_mode():
        out = layer(x)
    assert out.shape == (1, out_features)
    assert torch.isfinite(out).all()


def test_split_k_iters_for_scales_with_activation_rows():
    assert split_k_iters_for(1) == 32
    assert split_k_iters_for(2) == 16
    assert split_k_iters_for(8) == 4
    assert split_k_iters_for(32) == 1
    assert split_k_iters_for(0) == 1


@pytest.mark.parametrize("split_k_iters", [1, 4])
@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="kernel parity needs a GPU, or TRITON_INTERPRET=1 set before Triton loads",
)
def test_fused_splitk_kernel_parity_with_act_order_g_idx(split_k_iters):
    """Parity for a desc_act (act-order) checkpoint: g_idx is not monotonic,
    so the kernel must gather the group per K row rather than assume order."""
    pytest.importorskip("triton")
    torch.manual_seed(0)
    in_features, out_features, group_size, bits = 256, 128, 64, 4

    num_groups = in_features // group_size
    weight = torch.randint(0, 2**bits, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 2**bits, (num_groups, out_features), dtype=torch.int32)
    scales = (torch.rand(num_groups, out_features, dtype=torch.float16) * 0.5) + 0.75
    qweight = _pack_gptq_v2(weight, bits)
    qzeros = _pack_gptq_v2(zeros, bits, along_rows=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    perm = torch.randperm(in_features)
    g_idx = torch.tensor([p // group_size for p in perm], dtype=torch.int32, device=device)

    dense = scales[g_idx.cpu().long()] * (weight - zeros[g_idx.cpu().long()])
    x = torch.randn(5, in_features, dtype=torch.float16, device=device)

    actual = fused_dequant_matmul(
        x,
        qweight.to(device),
        scales.to(device),
        qzeros.to(device),
        g_idx,
        bits,
        32,
        2**bits - 1,
        split_k_iters=split_k_iters,
    )
    expected = x @ dense.to(x.dtype)

    abs_diff = (actual.float() - expected.float()).abs()
    tol = TOLERANCE[x.dtype] * max(1.0, expected.float().abs().max().item())
    assert abs_diff.max().item() <= tol, f"split_k_iters={split_k_iters} max diff {abs_diff.max().item()}"
