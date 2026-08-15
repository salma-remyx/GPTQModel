# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.adapter.loftq_init import loftq_init
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.quantizer import Quantizer


def _make_quantizer(bits: int = 4, sym: bool = True) -> Quantizer:
    qcfg = QuantizeConfig(bits=bits, sym=sym, group_size=-1)
    quantizer = Quantizer(qcfg=qcfg, name="test")
    quantizer.configure(perchannel=True)
    return quantizer


def test_loftq_init_shapes_and_device():
    torch.manual_seed(0)
    W = torch.randn(64, 96, dtype=torch.float32)
    quantizer = _make_quantizer()

    A, B, Q = loftq_init(W=W, rank=8, quantizer=quantizer, num_iters=4, dtype=torch.float16)

    assert A.shape == (8, 96)
    assert B.shape == (64, 8)
    assert Q.shape == W.shape
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    assert A.device == W.device and B.device == W.device


def test_loftq_init_reduces_reconstruction_error():
    torch.manual_seed(0)
    W = torch.randn(64, 96, dtype=torch.float32)
    quantizer = _make_quantizer()

    plain_q = quantizer.quantize(W.clone()).to(dtype=torch.float32)
    baseline = torch.norm(W - plain_q).item()

    A, B, Q = loftq_init(W=W, rank=16, quantizer=quantizer, num_iters=4, dtype=torch.float32)
    compensated = torch.norm(W - Q - B @ A).item()

    assert compensated < baseline


def test_loftq_init_rejects_invalid_args():
    W = torch.randn(8, 8, dtype=torch.float32)
    quantizer = _make_quantizer()

    with pytest.raises(ValueError):
        loftq_init(W=W, rank=0, quantizer=quantizer)
    with pytest.raises(ValueError):
        loftq_init(W=W, rank=4, quantizer=quantizer, num_iters=0)


def test_lora_adapter_carries_loftq_mode():
    lora = Lora(rank=8, init="loftq", num_iters=4)
    assert lora.init == "loftq"
    assert lora.to_dict()["rank"] == 8

    plain = Lora(rank=8)
    assert plain.init is None
