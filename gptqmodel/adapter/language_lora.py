# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Language-conditional LoRA corrections for quantized models.
# Adapted from "Language-Conditional Dequantization: Recovering What Quantization
# Steals from Non-English Languages" (arXiv:2608.11786): aggressive low-bit
# quantization disproportionately damages non-English languages, and small
# per-language low-rank corrections recover most of that gap where a single
# language-agnostic correction (plain EoRA) cannot.

import os
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import save_file

from ..utils.logger import setup_logger
from .adapter import HF_ADAPTER_CONFIG_FILE_NAME, HF_ADAPTER_FILE_NAME, HF_ADAPTER_WEIGHT_KEY_PREFIX, Lora
from .peft import LoraConfig

log = setup_logger()

DEFAULT_LANGUAGE = "default"


class LanguageAwareLora(Lora):
    """LoRA adapter that holds one A/B correction per language and routes at runtime.

    Each per-module instance is a deepcopy of the shared adapter, so the active
    language is tracked as a class attribute: one ``set_language()`` call
    switches every module in the model. When no language is set (or the requested
    language is missing) the default language's correction is applied, matching
    plain ``Lora`` behavior.
    """

    # process-wide routing shared by all deepcopied per-module instances
    _active_language: Optional[str] = None

    def __init__(
        self,
        rank: int,
        path: str = None,
        lora_A: torch.Tensor = None,
        lora_B: torch.Tensor = None,
        language_loras: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None,
        default_language: str = None,
    ):
        """Initializes the adapter with a default correction plus per-language corrections."""

        super().__init__(rank=rank, path=path, lora_A=lora_A, lora_B=lora_B)

        self.language_loras: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = dict(language_loras) if language_loras else {}
        if lora_A is not None and lora_B is not None:
            self.language_loras.setdefault(default_language or DEFAULT_LANGUAGE, (lora_A, lora_B))

        if default_language is not None:
            self.default_language = default_language
        elif self.language_loras:
            self.default_language = sorted(self.language_loras)[0]
        else:
            self.default_language = DEFAULT_LANGUAGE

    @classmethod
    def name(cls) -> str:
        """Returns the serialized adapter type name."""

        return "language_lora"

    @classmethod
    def set_language(cls, language: Optional[str]):
        """Routes all subsequent forward passes to the given language's correction."""

        cls._active_language = language

    @classmethod
    def active_language(cls) -> Optional[str]:
        """Returns the currently routed language, if one was set."""

        return cls._active_language

    def available_languages(self):
        """Lists the languages this adapter carries corrections for."""

        return sorted(self.language_loras)

    def _active_matrices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects the A/B pair for the routed language, converting dtype/device once."""

        language = type(self)._active_language or self.default_language
        pair = self.language_loras.get(language)
        if pair is None:
            log.warn.once(
                f"Adapter: language `{language}` has no correction, falling back to `{self.default_language}`."
            )
            language = self.default_language
            pair = self.language_loras.get(language)
        if pair is None:
            pair = (self.lora_A, self.lora_B)

        lora_A, lora_B = pair
        if lora_A is None or lora_B is None:
            raise ValueError(f"Adapter: no LoRA weights available for language `{language}`.")

        if x.dtype != lora_A.dtype or x.device != lora_A.device:
            lora_A = lora_A.to(device=x.device, dtype=x.dtype)
            lora_B = lora_B.to(device=x.device, dtype=x.dtype)
            self.language_loras[language] = (lora_A, lora_B)
            if language == self.default_language:
                self.lora_A, self.lora_B = lora_A, lora_B

        return lora_A, lora_B

    def apply(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Adds the routed language's LoRA update to the kernel output."""

        lora_A, lora_B = self._active_matrices(x)

        # same batched-output reshape handling as Lora.apply
        if out.dim() > x.dim() and out.shape[0] > 1:
            out_orgi_shape = out.shape
            out = out.view(-1, out.shape[-1])
            out.add_((x @ lora_A) @ lora_B)
            return out.view(out_orgi_shape)

        return out.add_((x @ lora_A) @ lora_B)

    def post_init(self, weight_key: str, device: torch.device, lora_A: torch.Tensor = None, lora_B: torch.Tensor = None):
        """Loads per-language corrections from sibling adapter directories under `path`.

        Expected save layout (produced by `save_language_adapters`): one standard
        HF adapter directory per language, ``{path}/{language}/adapter_model.safetensors``.
        A `path` without language sub-directories degrades to plain `Lora` loading.
        """

        if lora_A is not None and lora_B is not None:
            super().post_init(weight_key=weight_key, device=device, lora_A=lora_A, lora_B=lora_B)
            self.language_loras[self.default_language] = (self.lora_A, self.lora_B)
            return

        language_paths = discover_language_adapter_paths(self.path)
        if not language_paths:
            super().post_init(weight_key=weight_key, device=device)
            self.language_loras[self.default_language] = (self.lora_A, self.lora_B)
            return

        for language, language_path in language_paths.items():
            lora = Lora(rank=self.rank, path=language_path)
            lora.post_init(weight_key=weight_key, device=device)
            self.language_loras[language] = (lora.lora_A, lora.lora_B)

        if self.default_language not in self.language_loras:
            self.default_language = sorted(self.language_loras)[0]
        self.lora_A, self.lora_B = self.language_loras[self.default_language]

    def to_dict(self):
        """Serializes the minimal adapter descriptor used by GPT-QModel."""

        data = super().to_dict()
        data["default_language"] = self.default_language
        return data


def discover_language_adapter_paths(path: str) -> Dict[str, str]:
    """Maps language tag -> adapter directory for sub-directories of `path` holding an adapter config."""

    if not isinstance(path, str) or not os.path.isdir(path):
        return {}

    found = {}
    for entry in sorted(os.listdir(path)):
        sub_dir = os.path.join(path, entry)
        if os.path.isdir(sub_dir) and os.path.isfile(os.path.join(sub_dir, HF_ADAPTER_CONFIG_FILE_NAME)):
            found[entry] = sub_dir
    return found


def save_language_adapters(model, save_dir: str, model_save_dir: str = None):
    """Writes one standard HF LoRA adapter directory per language.

    Consumes ``model.lora_results`` (full module name -> `LanguageAwareLora`) and
    emits ``{save_dir}/{language}/adapter_model.safetensors`` plus an
    ``adapter_config.json`` per language, so each language stays loadable through
    the existing plain `Lora` path as well.

    `language_loras` pairs use the runtime layout (A: in x rank, B: rank x out),
    matching `Lora.lora_A/lora_B` after `post_init`; they are transposed into the
    HF adapter layout (A: rank x in, B: out x rank) on write.
    """

    qcfg = model.quantize_config

    rank_pattern = {}
    if qcfg.dynamic:
        rank_pattern = qcfg.extract_adapter_rank_patterns()

    languages = set()
    for adapter in model.lora_results.values():
        languages.update(adapter.language_loras)

    for language in sorted(languages):
        weights = {}
        target_modules = set()
        for key, adapter in model.lora_results.items():
            pair = adapter.language_loras.get(language)
            if pair is None:
                log.warn(f"Adapter: module `{key}` has no `{language}` correction, skipping.")
                continue
            key = key.lower()
            target_modules.add(key.split(".")[-1])
            weight_key = f"{HF_ADAPTER_WEIGHT_KEY_PREFIX}{key}"
            weights[f"{weight_key}.lora_A.weight"] = pair[0].T.contiguous()
            weights[f"{weight_key}.lora_B.weight"] = pair[1].T.contiguous()

        if not weights:
            log.warn(f"Adapter: no corrections found for language `{language}`, skipping save.")
            continue

        language_dir = f"{save_dir.removesuffix('/')}/{language}"
        os.makedirs(language_dir, exist_ok=True)

        lora_cfg = LoraConfig(base_model_name_or_path=model_save_dir,
                              r=qcfg.adapter.rank,
                              lora_alpha=qcfg.adapter.rank,
                              target_modules=list(target_modules),
                              rank_pattern=rank_pattern)
        lora_cfg.save_pretrained(save_dir=language_dir)

        weight_file_path = f"{language_dir}/{HF_ADAPTER_FILE_NAME}"
        log.info(f"Adapter: Saving `{language}` correction weights to -> `{language_dir}`")
        save_file(tensors=weights, filename=weight_file_path, metadata={"format": "pt"})
