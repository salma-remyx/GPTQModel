# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Per-language EoRA corrections (Language-Conditional Dequantization style).
# Adapted from "Language-Conditional Dequantization: Recovering What Quantization
# Steals from Non-English Languages" (arXiv:2608.11786): instead of one
# language-agnostic LoRA correction (EoRA), activation covariances are
# accumulated per language and each language gets its own low-rank correction.
# The paper's gradient-trained rank-2 LoRA is substituted with GPTQModel's
# native training-free eigenspace SVD correction (EoRA math), driven by
# language-partitioned calibration data.

import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Module

from ..adapter.adapter import Lora
from ..adapter.language_lora import LanguageAwareLora
from ..eora.eora import merge_eora_segments
from ..looper.eora_processor import EoraProcessor
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN
from ..looper.named_module import NamedModule
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                             PROCESS_LOG_NAME, PROCESS_LOG_TIME, PROCESS_USED_MEMORY)
from ..quantization.config import QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.model import move_to
from ..utils.torch import CPU, DEVICE_0, DEVICE_1

log = setup_logger()


class LanguageEoraProcessor(EoraProcessor):
    """Builds one LoRA-style correction per language from dequantization residuals.

    `calibration` must be a ``Dict[str, dataset]`` mapping a language tag to that
    language's calibration examples. Each language's examples are prepared
    separately and concatenated in sorted language order, so every batch index
    maps back to exactly one language. The forward hook routes each batch's
    covariance contribution into the per-language accumulator, and `process()`
    computes one eigenspace low-rank correction per language. Corrections are
    kept language-conditional at runtime (via `LanguageAwareLora`); the
    quantized weight itself is left untouched.
    """

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration: Dict[str, List],
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        require_fwd: bool = True,
        calibration_concat_separator: Optional[str] = None,
    ):
        """Prepares each language's calibration separately to preserve batch alignment."""

        if not isinstance(calibration, Dict) or not calibration:
            raise ValueError(
                "LanguageEoraProcessor requires `calibration` as a Dict[str, dataset] "
                "mapping language tag -> calibration examples."
            )

        self.languages = sorted(calibration.keys())
        self.default_language = self.languages[0]

        # prepare per-language so prepared-batch -> language alignment stays exact
        prepared: List = []
        self._batch_languages: List[str] = []
        for language in self.languages:
            language_batches = prepare_dataset_func(
                calibration_dataset=calibration[language],
                calibration_dataset_concat_size=calibration_concat_size,
                calibration_dataset_sort=calibration_sort,
                batch_size=batch_size,
                calibration_concat_separator=calibration_concat_separator,
            )
            prepared.extend(language_batches)
            self._batch_languages.extend([language] * len(language_batches))

        self._language_batch_counts: Dict[str, int] = {
            language: self._batch_languages.count(language) for language in self.languages
        }

        log.info(f"LanguageEoRA: languages = `{self.languages}`, batches per language = `{self._language_batch_counts}`")

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=prepare_dataset_func,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            batch_size=batch_size,
            require_fwd=require_fwd,
        )

        # super().__init__ ran with calibration=None; install the concatenated set now
        self.set_calibration_dataset(prepared)

    def _language_for_batch(self, batch_index: Optional[int]) -> str:
        """Maps a calibration batch index back to its language tag."""

        if batch_index is None or not 0 <= int(batch_index) < len(self._batch_languages):
            return self.default_language
        return self._batch_languages[int(batch_index)]

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that routes activation statistics per language."""

        def tmp(module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            """Processes one batch of inputs into the language's contribution segment."""

            batch_index = self.current_batch_index()
            language = self._language_for_batch(batch_index)

            batch, contribution, scale = self.eora_process_input(
                input=input,
                name=name,
                sample_size=self._language_batch_counts[language],
                device=module.weight.data.device,
            )

            self._accumulate_eora_contribution(
                name=name,
                language=language,
                batch_index=batch_index,
                batch=batch,
                contribution=contribution,
                scale=scale,
            )
        return tmp

    def _accumulate_eora_contribution(
        self,
        *,
        name: str,
        language: str,
        batch_index: Optional[int],
        batch: int,
        contribution: torch.Tensor,
        scale: float,
    ) -> None:
        """Merges one contribution segment into the per-language, per-device accumulator."""

        if batch <= 0:
            return

        contribution = contribution.detach()
        device = torch.device(contribution.device)
        scale_value = float(scale)

        with self.lock:
            language_accumulators = self._segment_accumulators.setdefault(name, {})
            accumulators = language_accumulators.setdefault(language, {})
            record = accumulators.get(device)

            index_value = int(batch_index) if batch_index is not None else 0

            if record is None:
                record = {
                    "total": contribution,
                    "scale_product": scale_value,
                    "start_index": index_value,
                    "end_index": index_value,
                    "count": 1,
                }
                accumulators[device] = record
                return

            total = record["total"]
            if total.device != contribution.device:
                total = total.to(device=contribution.device)

            total.mul_(scale_value)
            total.add_(contribution)

            record["total"] = total
            record["scale_product"] *= scale_value
            record["count"] += 1

            if batch_index is not None:
                batch_value = int(batch_index)
                if record["start_index"] is None or batch_value < record["start_index"]:
                    record["start_index"] = batch_value
                if record["end_index"] is None or batch_value > record["end_index"]:
                    record["end_index"] = batch_value
            else:
                if record.get("start_index") is None:
                    record["start_index"] = record["count"] - 1
                record["end_index"] = record["count"] - 1

            del contribution

    def _finalize_language_scaling_matrices(self, name: str) -> Dict[str, torch.Tensor]:
        """Merges accumulated segments into one eigen scaling matrix per language."""

        with self.lock:
            language_segments = self._segment_accumulators.pop(name, {})
            target_device = self._module_target_devices.pop(name, None)

        if not language_segments:
            raise RuntimeError(
                f"LanguageEoRA statistics for module '{name}' were not collected before processing."
            )

        matrices: Dict[str, torch.Tensor] = {}
        for language, segments in language_segments.items():
            ordered_segments = sorted(
                segments.values(),
                key=lambda record: record.get("start_index", 0),
            )

            language_device = target_device
            if language_device is None:
                language_device = torch.device(ordered_segments[0]["total"].device)

            segment_pairs = []
            for record in ordered_segments:
                total = record["total"]
                if total.device != language_device:
                    total = total.to(device=language_device, dtype=torch.float32)
                segment_pairs.append((total, float(record["scale_product"])))

            matrices[language] = merge_eora_segments(segment_pairs)

        return matrices

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Computes and installs the per-language LoRA corrections for one module."""

        assert isinstance(module.adapter_cfg, Lora)

        self.draw_progress(f"LanguageEoRA: Processing {module.name} ({module.module_dtype}) in layer")

        start = time.time()

        scaling_matrices = self._finalize_language_scaling_matrices(module.name)

        tp_info = module.state.get("tp_pad_info")
        pad_cols = 0
        original_cols = module.weight.data.shape[1]
        if isinstance(tp_info, dict):
            pad_cols = int(tp_info.get("pad_cols", 0) or 0)
            original_cols = int(tp_info.get("original_columns", original_cols))

        target_device = module.weight.data.device

        w_wq_delta: torch.Tensor = module.state.pop("w_wq_diff").to(
            dtype=torch.float32,
            device=target_device,
        )
        if pad_cols:
            valid_cols = original_cols + pad_cols
            w_wq_delta = w_wq_delta[:, :valid_cols]

        wq: torch.Tensor = module.state["wq"]
        if pad_cols:
            wq = wq[:, :valid_cols]

        assert w_wq_delta.dtype == torch.float32, f"w_wq_delta dtype: {w_wq_delta.dtype}"

        language_loras: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for language, eigen_scaling_diag_matrix in scaling_matrices.items():
            A, B = self.eora_compute_lora(
                w_wq_delta=w_wq_delta,
                name=f"{module.name} [{language}]",
                eigen_scaling_diag_matrix=eigen_scaling_diag_matrix,
                rank=module.adapter_cfg.rank,
                dtype=module.module_dtype,
                device=module.weight.data.device,
            )
            # eora_compute_lora returns the HF layout (A: rank x in, B: out x rank);
            # transpose into the runtime layout used by Lora.apply (A: in x rank, B: rank x out)
            language_loras[language] = (
                move_to(A.T.contiguous().to(dtype=module.module_dtype), device=CPU),
                move_to(B.T.contiguous().to(dtype=module.module_dtype), device=CPU),
            )

        del scaling_matrices, w_wq_delta

        if pad_cols:
            wq_trim = wq[:, :original_cols]
        else:
            wq_trim = wq

        module.state.update({
            "wq": move_to(wq_trim, device=CPU),
        })

        # unlike EoRA, no correction is baked into the weight: corrections stay
        # language-conditional and are applied at runtime by LanguageAwareLora
        module.weight.data = wq_trim.to(dtype=module.weight.data.dtype, device=target_device)

        del wq

        duration = time.time() - start
        with self.lock:
            self.durations.append(duration)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stats_0 = torch.cuda.memory_stats(DEVICE_0)
        active_0 = stats_0.get("active_bytes.all.current", 0) / 1024 ** 2
        peak_active_0 = stats_0.get("active_bytes.all.peak", 0) / 1024 ** 2

        if torch.cuda.device_count() > 1:
            stats_1 = torch.cuda.memory_stats(DEVICE_1)
            active_1 = stats_1.get("active_bytes.all.current", 0) / 1024 ** 2
            peak_active_1 = stats_1.get("active_bytes.all.peak", 0) / 1024 ** 2

            max_memory = f"{peak_active_0:.2f}MB, {peak_active_1:.2f}MB"
        else:
            max_memory = f"{peak_active_0:.2f}MB"

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: max_memory,
            "languages": f"{len(language_loras)}",
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.log.append(stat)

        self.log_new_row(stat)

        default_A, default_B = language_loras[self.default_language]
        adapter = LanguageAwareLora(
            rank=module.adapter_cfg.rank,
            lora_A=default_A,
            lora_B=default_B,
            language_loras=language_loras,
            default_language=self.default_language,
        )

        module.state.update({
            "adapter": adapter
        })

        module.state.pop("tp_pad_info", None)

    def name(self) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "language_eora"
