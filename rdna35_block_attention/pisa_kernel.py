from __future__ import annotations

import importlib
import math
import os
import pathlib
import sys
import sysconfig

import torch


_BLOCK_STATS_KERNEL = None


def _prepend_env_path(name: str, path: pathlib.Path) -> None:
    if not path.exists():
        return
    value = str(path)
    current = os.environ.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if any(os.path.normcase(part) == os.path.normcase(value) for part in parts):
        return
    os.environ[name] = value if not current else value + os.pathsep + current


def _prepare_windows_rocm_include_path() -> None:
    if os.name != "nt":
        return
    candidates = [
        pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "_rocm_sdk_core" / "include",
        pathlib.Path(sysconfig.get_paths().get("purelib", "")) / "_rocm_sdk_core" / "include",
        pathlib.Path(sysconfig.get_paths().get("platlib", "")) / "_rocm_sdk_core" / "include",
    ]
    for include_dir in candidates:
        _prepend_env_path("INCLUDE", include_dir)


def _get_block_stats_kernel():
    global _BLOCK_STATS_KERNEL
    if _BLOCK_STATS_KERNEL is not None:
        return globals()["triton"], _BLOCK_STATS_KERNEL

    _prepare_windows_rocm_include_path()
    globals()["triton"] = importlib.import_module("triton")
    globals()["tl"] = importlib.import_module("triton.language")
    triton = globals()["triton"]

    @triton.jit
    def _pisa_block_stats_kernel(
        k_ptr,
        v_ptr,
        k_mean_ptr,
        v_sum_ptr,
        h_sum_ptr,
        tokens: tl.constexpr,
        head_dim: tl.constexpr,
        padded_head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        block = tl.program_id(0)
        bh = tl.program_id(1)
        rows = tl.arange(0, block_size)
        dims = tl.arange(0, padded_head_dim)
        token_offsets = block * block_size + rows
        offsets = bh * tokens * head_dim + token_offsets[:, None] * head_dim + dims[None, :]
        valid = (token_offsets[:, None] < tokens) & (dims[None, :] < head_dim)
        k = tl.load(k_ptr + offsets, mask=valid, other=0.0)
        v = tl.load(v_ptr + offsets, mask=valid, other=0.0)
        length = tl.minimum(block_size, tokens - block * block_size).to(tl.float32)
        k_mean = tl.sum(k, axis=0) / length
        v_sum = tl.sum(v, axis=0)

        total_blocks = (tokens + block_size - 1) // block_size
        stat_offsets = bh * total_blocks * head_dim + block * head_dim + dims
        tl.store(k_mean_ptr + stat_offsets, k_mean, mask=dims < head_dim)
        tl.store(v_sum_ptr + stat_offsets, v_sum, mask=dims < head_dim)

        # Triton dot operands must match; the matrix product still accumulates in fp32.
        centered = (k - k_mean[None, :]).to(v.dtype)
        h_block = tl.dot(tl.trans(centered), v, input_precision="ieee")
        h_offsets = bh * head_dim * head_dim + dims[:, None] * head_dim + dims[None, :]
        h_valid = (dims[:, None] < head_dim) & (dims[None, :] < head_dim)
        tl.atomic_add(h_sum_ptr + h_offsets, h_block, mask=h_valid)

    _BLOCK_STATS_KERNEL = _pisa_block_stats_kernel
    return triton, _BLOCK_STATS_KERNEL


def pisa_prepare_triton(k: torch.Tensor, v: torch.Tensor, *, block_size: int = 64):
    """gfx1151 wave64-oriented PISA preparation stage.

    This fuses K centroids, V sums, and global centered H accumulation. Selection
    and mixed online-softmax remain staged PyTorch operations in the prototype.
    """
    if k.ndim != 3 or k.shape != v.shape:
        raise ValueError("PISA Triton preparation expects matching [BH,T,D] k/v.")
    if block_size != 64:
        raise ValueError("PISA Triton preparation requires block_size=64.")
    if k.dtype not in (torch.float16, torch.bfloat16) or v.dtype != k.dtype:
        raise ValueError("PISA Triton preparation requires matching fp16/bf16 k/v.")
    if not k.is_cuda or not v.is_cuda:
        raise ValueError("PISA Triton preparation requires CUDA/HIP tensors.")
    if not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("PISA Triton preparation requires contiguous k/v.")

    triton, kernel = _get_block_stats_kernel()
    batch_heads, tokens, head_dim = k.shape
    padded_head_dim = triton.next_power_of_2(head_dim)
    if padded_head_dim > 256:
        raise ValueError(f"PISA Triton preparation requires D<=256, got D={head_dim}.")
    total_blocks = math.ceil(tokens / block_size)
    k_means = torch.empty((batch_heads, total_blocks, head_dim), device=k.device, dtype=torch.float32)
    v_sums = torch.empty_like(k_means)
    h_sum = torch.zeros((batch_heads, head_dim, head_dim), device=k.device, dtype=torch.float32)
    kernel[(total_blocks, batch_heads)](
        k,
        v,
        k_means,
        v_sums,
        h_sum,
        tokens,
        head_dim,
        padded_head_dim,
        block_size,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    lengths = [min(block_size, tokens - block * block_size) for block in range(total_blocks)]
    return k_means, v_sums, h_sum, lengths
