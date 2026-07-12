from __future__ import annotations

import importlib
import math
import os
import pathlib
import sys
import sysconfig

import torch

_KERNEL = None
_FULL_ATTENTION_KERNEL = None


def _prepend_env_path(name: str, path: pathlib.Path) -> None:
    if not path.exists():
        return
    value = str(path)
    current = os.environ.get(name, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if any(os.path.normcase(p) == os.path.normcase(value) for p in parts):
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


def _load_triton_modules():
    _prepare_windows_rocm_include_path()
    globals()["triton"] = importlib.import_module("triton")
    globals()["tl"] = importlib.import_module("triton.language")
    return globals()["triton"], globals()["tl"]


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return globals()["triton"], _KERNEL

    triton, _ = _load_triton_modules()

    @triton.jit
    def _fixed_block_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        tokens: tl.constexpr,
        scale: tl.constexpr,
        causal: tl.constexpr,
        block_d: tl.constexpr,
        block_m: tl.constexpr,
    ):
        pid_block = tl.program_id(0)
        pid_bh = tl.program_id(1)
        block_start = pid_block * block_m

        offs_m = tl.arange(0, block_m)
        offs_n = tl.arange(0, block_m)
        offs_d = tl.arange(0, block_d)

        base = pid_bh * tokens * block_d
        q_offsets = base + (block_start + offs_m[:, None]) * block_d + offs_d[None, :]
        k_offsets = base + (block_start + offs_n[:, None]) * block_d + offs_d[None, :]
        v_offsets = base + (block_start + offs_n[:, None]) * block_d + offs_d[None, :]

        q_mask = (block_start + offs_m[:, None] < tokens)
        kv_mask = (block_start + offs_n[:, None] < tokens)

        q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        k = tl.load(k_ptr + k_offsets, mask=kv_mask, other=0.0)
        v = tl.load(v_ptr + v_offsets, mask=kv_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        valid = (block_start + offs_m[:, None] < tokens) & (block_start + offs_n[None, :] < tokens)
        if causal:
            valid = valid & (offs_n[None, :] <= offs_m[:, None])
        scores = tl.where(valid, scores, -float("inf"))

        row_max = tl.max(scores, 1)
        shifted = scores - row_max[:, None]
        numer = tl.exp(shifted)
        denom = tl.sum(numer, 1)
        denom = tl.where(denom == 0.0, 1.0, denom)
        probs = numer / denom[:, None]
        out = tl.dot(probs.to(v.dtype), v, input_precision="ieee")

        o_offsets = base + (block_start + offs_m[:, None]) * block_d + offs_d[None, :]
        tl.store(o_ptr + o_offsets, out, mask=q_mask)

    _KERNEL = _fixed_block_attention_kernel
    return triton, _KERNEL


def _get_full_attention_kernel():
    global _FULL_ATTENTION_KERNEL
    if _FULL_ATTENTION_KERNEL is not None:
        return globals()["triton"], _FULL_ATTENTION_KERNEL

    triton, _ = _load_triton_modules()

    @triton.jit
    def _full_attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        q_tokens,
        kv_tokens,
        scale,
        block_d: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        q_base = pid_bh * q_tokens * block_d
        kv_base = pid_bh * kv_tokens * block_d

        q_offsets = q_base + offs_m[:, None] * block_d + offs_d[None, :]
        q = tl.load(q_ptr + q_offsets, mask=offs_m[:, None] < q_tokens, other=0.0)

        row_max = tl.full((block_m,), -float("inf"), tl.float32)
        row_sum = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, block_d), tl.float32)

        for start_n in range(0, kv_tokens, block_n):
            current_n = start_n + offs_n
            kv_offsets = kv_base + current_n[:, None] * block_d + offs_d[None, :]
            kv_mask = current_n[:, None] < kv_tokens
            k = tl.load(k_ptr + kv_offsets, mask=kv_mask, other=0.0)
            v = tl.load(v_ptr + kv_offsets, mask=kv_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            scores = tl.where(current_n[None, :] < kv_tokens, scores, -float("inf"))
            tile_max = tl.max(scores, 1)
            next_max = tl.maximum(row_max, tile_max)
            correction = tl.exp(row_max - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v, input_precision="ieee")
            row_sum = row_sum * correction + tl.sum(probabilities, 1)
            row_max = next_max

        out = acc / row_sum[:, None]
        tl.store(o_ptr + q_offsets, out, mask=offs_m[:, None] < q_tokens)

    _FULL_ATTENTION_KERNEL = _full_attention_forward_kernel
    return triton, _FULL_ATTENTION_KERNEL


def full_attention_triton_bh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    block_m: int,
    block_n: int,
    num_warps: int,
    waves_per_eu: int,
) -> torch.Tensor:
    batch_heads, q_tokens, head_dim = q.shape
    kv_tokens = k.shape[1]
    out = torch.empty_like(q)
    triton, kernel = _get_full_attention_kernel()
    grid = (triton.cdiv(q_tokens, block_m), batch_heads)
    kernel[grid](
        q,
        k,
        v,
        out,
        q_tokens,
        kv_tokens,
        scale,
        head_dim,
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=waves_per_eu,
    )
    return out


def fixed_block_attention_triton_bh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int = 64,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"Triton path expects [BH,T,D], got rank {q.ndim}.")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Triton path requires identical q/k/v shapes.")
    if block_size != 64:
        raise ValueError("Triton path only supports block_size=64.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Triton path supports fp16/bf16 only, got {q.dtype}.")
    if not q.is_cuda:
        raise ValueError("Triton path requires a CUDA/HIP device tensor.")

    batch_heads, tokens, head_dim = q.shape
    if head_dim not in (32, 64, 128):
        raise ValueError(f"Triton path supports head_dim 32/64/128 only, got {head_dim}.")

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    out = torch.empty_like(q_c)
    triton, kernel = _get_kernel()
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    grid = (triton.cdiv(tokens, block_size), batch_heads)
    kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        tokens,
        scale_value,
        bool(causal),
        head_dim,
        block_size,
        num_warps=4 if head_dim <= 64 else 8,
        num_stages=1,
    )
    return out
