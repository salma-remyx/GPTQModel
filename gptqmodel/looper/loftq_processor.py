# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# LoftQ arXiv https://arxiv.org/abs/2310.08659

import copy
import time
from typing import Dict, Optional

import torch

from ..adapter.adapter import Lora
from ..adapter.loftq_init import DEFAULT_LOFTQ_ITERS, loftq_init
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                             PROCESS_LOG_NAME, PROCESS_LOG_TIME)
from ..quantization.config import QuantizeConfig
from ..quantization.quantizer import Quantizer
from ..utils.logger import setup_logger
from ..utils.torch import CPU

log = setup_logger()


class LoftqProcessor(LoopProcessor):
    """Builds LoRA-Fine-Tuning-Aware (LoftQ) adapters from dequantization residuals."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        calibration_concat_separator: Optional[str] = None,
        num_iters: int = DEFAULT_LOFTQ_ITERS,
    ):
        """Initializes LoftQ processing; residuals come from DequantizeProcessor."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            execution_config=ExecutionConfig(require_fwd=True),
        )

        self.num_iters = num_iters

    def preprocess(self, module: NamedModule, **kwargs):
        """Clones adapter config, applies rank overrides, and gates the module."""

        if self.qcfg.dynamic_get(layer_name=module.full_name) == False:  # noqa: E712
            module.adapter_cfg = None  # hack
            return

        adapter_cfg = copy.deepcopy(self.qcfg.adapter)

        adapter_cfg.rank = self.qcfg.dynamic_get(
            module.full_name,
            key="adapter",
            sub_key="rank",
            default=adapter_cfg.rank,
        )

        module.adapter_cfg = adapter_cfg

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether LoftQ was disabled for this module by dynamic config."""

        return module.adapter_cfg in [None, {}]

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Computes and installs the LoftQ LoRA correction for one quantized module."""

        assert isinstance(module.adapter_cfg, Lora)

        self.draw_progress(f"LoftQ: Processing {module.name} ({module.module_dtype}) in layer")

        start = time.time()

        w_wq_delta: torch.Tensor = module.state.pop("w_wq_diff")
        wq: torch.Tensor = module.state["wq"]

        target_device = w_wq_delta.device

        # recover the pre-quantization weight from the stored residual
        W = w_wq_delta + wq.to(dtype=torch.float32)
        del w_wq_delta

        # LoftQ re-quantizes the compensated backbone on the same grid the
        # checkpoint already uses; RTN grid search on W needs no calibration.
        quantizer = Quantizer(qcfg=self.qcfg, name=module.name)
        quantizer.configure(perchannel=True)
        A, B, _ = loftq_init(
            W=W,
            rank=module.adapter_cfg.rank,
            quantizer=quantizer,
            num_iters=self.num_iters,
            dtype=module.module_dtype,
        )

        del W

        # wq with A/B applied
        computed_wq = (wq.to(device=target_device) + (B @ A)).to(dtype=wq.dtype)

        module.state.update({
            "wq": computed_wq.to(device=CPU),
        })

        # override module weight with computed weight with B@A delta
        module.weight.data = computed_wq.to(dtype=module.weight.data.dtype, device=target_device)

        duration = time.time() - start
        with self.lock:
            self.durations.append(duration)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.log.append(stat)

        self.log_new_row(stat)

        loftq = Lora(
            rank=module.adapter_cfg.rank,
            lora_A=A.to(device=CPU),
            lora_B=B.to(device=CPU),
        )

        module.state.update({
            "adapter": loftq
        })

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Stores the finalized adapter object in the processor result map."""

        self.result_save(module.full_name, module.state.pop("adapter"))

    def finalize(self, model: BaseQModel, **kwargs):
        """Attaches the collected adapters to the model until `save()` is called."""

        # hack: store loras into model until `save()` is called
        model.lora_results = self.results()

        super().finalize(model=model, **kwargs)

    def name(self) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "loftq"
