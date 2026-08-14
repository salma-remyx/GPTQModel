# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import torch

from gptqmodel.adapter.adapter import ADAPTER_MAPPING, Lora, normalize_adapter
from gptqmodel.adapter.language_lora import LanguageAwareLora, save_language_adapters
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


RANK = 2
IN_FEATURES = 8
OUT_FEATURES = 4


def _make_language_lora() -> LanguageAwareLora:
    torch.manual_seed(0)
    language_loras = {
        "en": (torch.randn(IN_FEATURES, RANK, dtype=torch.float16), torch.randn(RANK, OUT_FEATURES, dtype=torch.float16)),
        "ko": (torch.randn(IN_FEATURES, RANK, dtype=torch.float16), torch.randn(RANK, OUT_FEATURES, dtype=torch.float16)),
    }
    default_A, default_B = language_loras["en"]
    return LanguageAwareLora(
        rank=RANK,
        lora_A=default_A,
        lora_B=default_B,
        language_loras=language_loras,
        default_language="en",
    )


def teardown_function():
    # class-level routing must not leak across tests
    LanguageAwareLora.set_language(None)


def test_normalize_adapter_builds_language_lora():
    adapter = normalize_adapter({"name": "language_lora", "rank": RANK})

    assert isinstance(adapter, LanguageAwareLora)
    assert ADAPTER_MAPPING["language_lora"] is LanguageAwareLora
    assert adapter.rank == RANK


def test_apply_dispatches_to_active_language():
    adapter = _make_language_lora()
    x = torch.randn(3, IN_FEATURES, dtype=torch.float16)

    LanguageAwareLora.set_language("ko")
    out = torch.zeros(3, OUT_FEATURES, dtype=torch.float16)
    result = adapter.apply(x=x, out=out.clone())

    ko_A, ko_B = adapter.language_loras["ko"]
    expected = (x @ ko_A) @ ko_B
    assert torch.allclose(result, expected)

    LanguageAwareLora.set_language("en")
    result = adapter.apply(x=x, out=torch.zeros(3, OUT_FEATURES, dtype=torch.float16))

    en_A, en_B = adapter.language_loras["en"]
    expected = (x @ en_A) @ en_B
    assert torch.allclose(result, expected)


def test_apply_falls_back_to_default_language():
    adapter = _make_language_lora()
    x = torch.randn(3, IN_FEATURES, dtype=torch.float16)

    # no language routed: default correction applies, matching plain Lora math
    result = adapter.apply(x=x, out=torch.zeros(3, OUT_FEATURES, dtype=torch.float16))
    en_A, en_B = adapter.language_loras["en"]
    assert torch.allclose(result, (x @ en_A) @ en_B)

    # unknown language: warn + default fallback, no crash
    LanguageAwareLora.set_language("ja")
    result = adapter.apply(x=x, out=torch.zeros(3, OUT_FEATURES, dtype=torch.float16))
    assert torch.allclose(result, (x @ en_A) @ en_B)


def test_default_language_matches_plain_lora_apply():
    adapter = _make_language_lora()
    en_A, en_B = adapter.language_loras["en"]
    plain = Lora(rank=RANK, lora_A=en_A.clone(), lora_B=en_B.clone())

    x = torch.randn(3, IN_FEATURES, dtype=torch.float16)
    out = torch.randn(3, OUT_FEATURES, dtype=torch.float16)

    language_result = adapter.apply(x=x, out=out.clone())
    plain_result = plain.apply(x=x, out=out.clone())

    assert torch.allclose(language_result, plain_result)


def test_batched_output_reshape_preserved():
    adapter = _make_language_lora()
    x = torch.randn(2, 3, IN_FEATURES, dtype=torch.float16)

    LanguageAwareLora.set_language("ko")
    out = torch.zeros(2, 3, OUT_FEATURES, dtype=torch.float16)
    result = adapter.apply(x=x.view(-1, IN_FEATURES), out=out)

    assert result.shape == (2, 3, OUT_FEATURES)
    ko_A, ko_B = adapter.language_loras["ko"]
    expected = ((x.view(-1, IN_FEATURES) @ ko_A) @ ko_B).view(2, 3, OUT_FEATURES)
    assert torch.allclose(result, expected)


def test_torch_linear_accepts_lora_subclass():
    ok, err = TorchLinear._validate_shared(
        pack_dtype=torch.int32,
        dtype=torch.float16,
        adapter=_make_language_lora(),
    )

    assert ok, f"TorchLinear rejected LanguageAwareLora: {err}"


def test_save_language_adapters_roundtrip():
    adapter = _make_language_lora()
    model = SimpleNamespace(
        quantize_config=SimpleNamespace(adapter=Lora(rank=RANK), dynamic=None),
        lora_results={"model.layers.0.mlp.gate_proj": adapter},
    )

    with tempfile.TemporaryDirectory() as save_dir:
        save_language_adapters(model=model, save_dir=save_dir, model_save_dir="base-model")

        for language in ("en", "ko"):
            language_dir = os.path.join(save_dir, language)
            assert os.path.isfile(os.path.join(language_dir, "adapter_model.safetensors"))
            assert os.path.isfile(os.path.join(language_dir, "adapter_config.json"))

        # each saved language dir loads through the existing plain Lora path
        loaded = Lora(rank=RANK, path=os.path.join(save_dir, "ko"))
        loaded.post_init(weight_key="model.layers.0.mlp.gate_proj", device=torch.device("cpu"))

        ko_A, ko_B = adapter.language_loras["ko"]
        assert torch.allclose(loaded.lora_A, ko_A)
        assert torch.allclose(loaded.lora_B, ko_B)
