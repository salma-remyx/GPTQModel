# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from gptqmodel.quantization.config import GridRefineConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def _make_module(hidden_dim: int) -> nn.Linear:
    return nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float32).eval()


def _affine_grid(W: torch.Tensor, maxq: int, group_size: int):
    """RTN-style affine grid params for `W`, the shape Quantizer.find_params emits."""

    columns = W.shape[1]
    groups = (columns + group_size - 1) // group_size if group_size != -1 else 1
    span = group_size if group_size != -1 else columns
    scale = torch.empty(W.shape[0], groups)
    zero = torch.empty(W.shape[0], groups)
    for g in range(groups):
        block = W[:, g * span:(g + 1) * span]
        xmin = block.min(dim=1).values.clamp(max=0)
        xmax = block.max(dim=1).values.clamp(min=0)
        s = ((xmax - xmin) / maxq).clamp(min=1e-8)
        scale[:, g] = s
        zero[:, g] = torch.round(-xmin / s).clamp(0, maxq)
    return scale, zero


def _quantize(W: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int, group_size: int):
    columns = W.shape[1]
    span = group_size if group_size != -1 else columns
    scale_dense = scale.repeat_interleave(span, dim=1)[:, :columns]
    zero_dense = zero.repeat_interleave(span, dim=1)[:, :columns]
    q = torch.clamp(torch.round(W / scale_dense) + zero_dense, 0, maxq)
    return q, scale_dense * (q - zero_dense), scale_dense, zero_dense


def _hessian_loss(W: torch.Tensor, Q: torch.Tensor, H: torch.Tensor) -> float:
    E = W - Q
    return 0.5 * torch.einsum("ij,jk,ik->", E, H, E).item()


def test_grid_refine_lowers_hessian_loss_and_stays_on_grid():
    torch.manual_seed(0)
    from gptqmodel.quantization.grid_refine import refine

    W = torch.randn(8, 16)
    H = (torch.randn(128, 16).T @ torch.randn(128, 16)) / 128

    maxq = 2**2 - 1  # 2-bit: where greedy initializers leave the most on the table
    scale, zero = _affine_grid(W, maxq, group_size=8)
    _, Q0, scale_dense, zero_dense = _quantize(W, scale, zero, maxq, 8)

    Q1, loss_before, loss_after = refine(
        W, H, Q0, scale, zero, maxq, 8, GridRefineConfig(sweeps=64)
    )

    assert loss_after < loss_before
    # every refined value must sit on the original (scale, zero) grid
    q1 = torch.clamp(torch.round(Q1 / scale_dense) + zero_dense, 0, maxq)
    assert torch.allclose(Q1, scale_dense * (q1 - zero_dense), atol=1e-5)


def test_grid_refine_disabled_by_default_and_enabled_via_config():
    torch.manual_seed(0)
    inp = torch.randn(1, 4, 8)
    W = _make_module(8).weight.data.float()
    X = inp.reshape(-1, 8).float()
    H = 2.0 * (X.T @ X) / X.shape[0]  # the Hessian GPTQ.add_batch accumulates

    def _run(grid_refine):
        layer = _make_module(8)
        with torch.no_grad():
            layer.weight.copy_(W)
        qcfg = QuantizeConfig(bits=2, group_size=4, grid_refine=grid_refine)
        gptq = GPTQ(layer, qcfg=qcfg)
        gptq.quantizer.configure(perchannel=True)
        gptq.add_batch(inp, None)
        return gptq.quantize(blocksize=4)[0]

    Q_off = _run(None)
    Q_on = _run(True)

    assert QuantizeConfig(bits=2, group_size=4).grid_refine is None  # opt-in by default
    assert isinstance(QuantizeConfig(bits=2, group_size=4, grid_refine=True).grid_refine, GridRefineConfig)
    # the refined run must lower the Hessian-weighted reconstruction error
    assert _hessian_loss(W, Q_on.float(), H) < _hessian_loss(W, Q_off.float(), H)


def test_grid_refine_config_normalizes_dict_and_bool():
    from gptqmodel.quantization.config import GPTQConfig

    assert GPTQConfig(grid_refine=None).grid_refine is None
    assert GPTQConfig(grid_refine=False).grid_refine is None
    assert isinstance(GPTQConfig(grid_refine=True).grid_refine, GridRefineConfig)
    tuned = GPTQConfig(grid_refine={"sweeps": 9, "radius": 2}).grid_refine
    assert isinstance(tuned, GridRefineConfig) and tuned.sweeps == 9 and tuned.radius == 2

    import pytest

    with pytest.raises(ValueError, match="radius"):
        GPTQConfig(grid_refine={"radius": 0})
