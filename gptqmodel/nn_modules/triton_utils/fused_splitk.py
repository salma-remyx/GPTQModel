# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Fused dequant + matmul for the W4A16 (and W2/W8) GPTQ-V2 layout with SplitK
# work decomposition. Adapted from "Accelerating a Triton Fused Kernel for W4A16
# Quantized Inference with SplitK work decomposition" (arXiv:2402.00025v2):
# the paper's contribution is the SplitK grid for skinny x @ W shapes, where
# per-tile reuse of the packed weight is too small to saturate the device and
# the K reduction must be split across programs to recover occupancy.

import contextlib

import torch
import triton
import triton.language as tl

from ...utils.env import env_flag


# SplitK raises occupancy on skinny matmuls (decode-shaped m << n == k) but
# costs an atomic reduction per output tile. Keep it opt-in so the default
# inference path is unchanged until the speedup is validated per-device.
FUSED_SPLITK_ENABLED = env_flag("GPTQMODEL_TRITON_FUSED_SPLITK", default=False)

# Paper: split_k_iters is a power of two and <= 32.
SPLIT_K_MAX = 32

# Tile shapes for the skinny (m << n == k) shapes this kernel targets. SPLIT_K
# is passed by the caller, not autotuned, so the grid's second axis and the
# kernel's K stride always agree.
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64

SUPPORTED_BITS = (2, 4, 8)


@triton.jit
def fused_dequant_matmul_splitk_kernel(
    a_ptr,
    qweight_ptr,
    c_ptr,
    scales_ptr,
    qzeros_ptr,
    g_idx_ptr,
    M,
    N,
    K,
    bits: tl.constexpr,
    maxq: tl.constexpr,
    pack_bits: tl.constexpr,
    stride_am,
    stride_ak,
    stride_qwk,
    stride_qwn,
    stride_cm,
    stride_cn,
    stride_scales_g,
    stride_scales_n,
    stride_qzeros_g,
    stride_qzeros_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """C = A @ dequant(qweight, qzeros, scales), GPTQ-V2 column-major layout.

    A is (M, K) fp16/bf16, qweight is (ceil(K*bits/pack_bits), N) int, scales
    is (G, N), qzeros is (G, ceil(N*bits/pack_bits)) int, g_idx is (K,) int32.
    """
    pack_scale: tl.constexpr = pack_bits // bits

    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # qweight words are laid out along K: word k // pack_scale holds rows
    # k % pack_scale .. of the packed column block.
    qw_ptrs = qweight_ptr + (offs_k[:, None] // pack_scale) * stride_qwk + offs_n[None, :] * stride_qwn

    g_ptrs = g_idx_ptr + offs_k
    # One word of qzeros covers `pack_scale` adjacent output columns.
    qz_ptrs = qzeros_ptr + (offs_n[None, :] // pack_scale) * stride_qzeros_n
    sc_ptrs = scales_ptr + offs_n[None, :] * stride_scales_n

    w_shift = (offs_k % pack_scale) * bits
    z_shift = (offs_n % pack_scale) * bits

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        mask_k = offs_k < K
        mask_a = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_w = mask_k[:, None] & mask_n[None, :]
        qw = tl.load(qw_ptrs, mask=mask_w, other=0)

        g_idx = tl.load(g_ptrs, mask=mask_k, other=0)

        zeros_mask = (g_idx[:, None] >= 0) & mask_n[None, :]
        qz = tl.load(qz_ptrs + g_idx[:, None] * stride_qzeros_g, mask=zeros_mask, other=0)
        qz = (qz >> z_shift[None, :]) & maxq

        sc = tl.load(sc_ptrs + g_idx[:, None] * stride_scales_g, mask=zeros_mask, other=0.0)

        w = (qw >> w_shift[:, None]) & maxq
        w = (w.to(sc.dtype) - qz.to(sc.dtype)) * sc

        accumulator = tl.dot(a, w, accumulator, out_dtype=tl.float32)

        offs_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        qw_ptrs += (BLOCK_SIZE_K // pack_scale) * SPLIT_K * stride_qwk
        g_ptrs += BLOCK_SIZE_K * SPLIT_K

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def _next_pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _device_ctx(device: torch.device):
    if device.type == "xpu":
        return torch.xpu.device(device.index)
    if device.type == "cuda":
        return torch.cuda.device(device.index)
    # TRITON_INTERPRET=1 runs the kernel on the host; no device switch needed.
    return contextlib.nullcontext()


def split_k_iters_for(m: int) -> int:
    """Paper: parallelism along K for a skinny activation matrix.

    A tile row of BLOCK_SIZE_M covers 1..m rows, so the (m, n, k) shape offers
    only ceil(m/BLOCK_SIZE_M) * ceil(n/BLOCK_SIZE_N) tiles. SplitK recovers the
    idle SMs by partitioning K, at the cost of one atomic per output element.
    """
    if m <= 0:
        return 1
    iters = 32 // _next_pow2(m)
    return max(1, min(SPLIT_K_MAX, iters))


def fused_dequant_matmul(
    input_tensor: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    bits: int,
    pack_bits: int,
    maxq: int,
    split_k_iters: int | None = None,
) -> torch.Tensor:
    """Fused dequant + GEMM in one kernel, with optional SplitK decomposition.

    Shapes follow the GPTQ-V2 buffers held by ``TritonV2Linear``/``TorchLinear``:
    qweight (ceil(K*bits/pack_bits), N), qzeros (G, ceil(N*bits/pack_bits)),
    scales (G, N), g_idx (K,). Falls back to the caller when the layout or bit
    width is not one this kernel decodes.
    """
    m, k = input_tensor.shape
    out_features = qweight.shape[1]

    if bits not in SUPPORTED_BITS:
        raise ValueError(f"fused SplitK matmul supports bits {SUPPORTED_BITS}, got {bits}")
    if k % (pack_bits // bits):
        raise ValueError(f"in_features {k} must be divisible by {pack_bits // bits}")
    if out_features % (pack_bits // bits):
        raise ValueError(f"out_features {out_features} must be divisible by {pack_bits // bits}")
    if g_idx.device != qweight.device:
        raise ValueError("g_idx must live on the same device as qweight")

    if split_k_iters is None:
        split_k_iters = split_k_iters_for(m)
    if split_k_iters not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"split_k_iters must be a power of two <= {SPLIT_K_MAX}, got {split_k_iters}")

    # Accumulate in fp32 and reduce the partials through atomics, so the split
    # reduction does not add rounding noise on top of the quantization error.
    output = torch.zeros((m, out_features), device=input_tensor.device, dtype=torch.float32)

    grid = (
        triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(out_features, BLOCK_SIZE_N),
        split_k_iters,
    )

    with _device_ctx(qweight.device):
        fused_dequant_matmul_splitk_kernel[grid](
            input_tensor,
            qweight,
            output,
            scales,
            qzeros,
            g_idx,
            m,
            out_features,
            k,
            bits=bits,
            maxq=maxq,
            pack_bits=pack_bits,
            stride_am=input_tensor.stride(0),
            stride_ak=input_tensor.stride(1),
            stride_qwk=qweight.stride(0),
            stride_qwn=qweight.stride(1),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            stride_scales_g=scales.stride(0),
            stride_scales_n=scales.stride(1),
            stride_qzeros_g=qzeros.stride(0),
            stride_qzeros_n=qzeros.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            SPLIT_K=split_k_iters,
        )

    return output.to(input_tensor.dtype)


__all__ = [
    "BLOCK_SIZE_K",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "FUSED_SPLITK_ENABLED",
    "SPLIT_K_MAX",
    "SUPPORTED_BITS",
    "fused_dequant_matmul",
    "fused_dequant_matmul_splitk_kernel",
    "split_k_iters_for",
]
