# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# LoftQ arXiv https://arxiv.org/abs/2310.08659
# LoftQ Official Repo: https://github.com/yxli2123/LoftQ (MIT)
#
# This file adapts the alternating SVD initialization from LoftQ for the
# GPT-QModel adapter path. GPT-QModel quantizes with GPTQ (not RTN) and runs
# the low-rank init as a post-quant correction in the same slot EoRA occupies,
# so the paper's RTN inner step is replaced by dequantize(quantize(W)) using
# this repo's own Quantizer, and its PEFT-side initialization is replaced by
# saving HF-compatible lora_A/lora_B tensors.

from typing import Tuple

import torch
from torch import Tensor

from ..quantization.quantizer import Quantizer
from ..utils.logger import setup_logger

log = setup_logger()

# Paper default; PEFT's LoftQConfig exposes the same knob.
DEFAULT_LOFTQ_ITERS = 4


def loftq_init(
    W: Tensor,
    rank: int,
    quantizer: Quantizer,
    num_iters: int = DEFAULT_LOFTQ_ITERS,
    dtype: torch.dtype = torch.float16,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Approximate W with Q + B@A via alternating SVD on the residual.

    Starting from Q = dequantize(quantize(W)), each iteration takes the SVD of
    the residual W - Q and folds its leading singular directions into the
    low-rank factors, then re-quantizes the compensated backbone. Mirrors the
    reference implementation (loftq_quantizer / replace_peft_model_int8) with
    RTN substituted by this repo's Quantizer.

    Returns (A, B, Q); A/B are cast to `dtype`, the caller owns device placement.
    """

    if rank <= 0:
        raise ValueError(f"Adapter: LoftQ `rank` must be > 0: actual = `{rank}`.")
    if num_iters <= 0:
        raise ValueError(f"Adapter: LoftQ `num_iters` must be > 0: actual = `{num_iters}`.")

    W = W.to(dtype=torch.float32)
    r = min(rank, min(W.shape))
    if r < rank:
        log.warn(f"Adapter: LoftQ `rank` clamped to `{r}` for weight shape `{tuple(W.shape)}`.")

    # R = W - Q; the low-rank factors accumulate -R so that Q + B@A tracks W.
    Q = quantizer.quantize(W.clone()).to(dtype=torch.float32)
    R = W - Q

    U, S, V = torch.linalg.svd(R, full_matrices=False)
    A = torch.zeros((r, W.shape[1]), dtype=torch.float32, device=W.device)
    B = U[:, :r] @ torch.sqrt(torch.diag(S[:r]))

    for _ in range(num_iters - 1):
        R = W - Q - B @ A
        U, S, V = torch.linalg.svd(R, full_matrices=False)
        A = torch.sqrt(torch.diag(S[:r])) @ V[:r, :]
        B = U[:, :r] @ torch.sqrt(torch.diag(S[:r]))

        Q = quantizer.quantize((W - B @ A).clone()).to(dtype=torch.float32)

    del R, U, S, V

    return A.to(dtype=dtype), B.to(dtype=dtype), Q
