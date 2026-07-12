from __future__ import annotations

import math
import sys
from typing import Any

import torch

from . import diagnostics


SUPPORTED_HEAD_DIMS = {64, 128}
SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def _kernel_config(head_dim: int, gfx_target: str | None) -> dict[str, int]:
    if head_dim == 128:
        config = {"block_m": 32, "block_n": 32, "num_warps": 4}
    else:
        config = {"block_m": 64, "block_n": 32, "num_warps": 4}
    config["waves_per_eu"] = 1 if gfx_target == "gfx1151" else 2
    return config


def validate_full_attention_triton_bh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> dict[str, Any]:
    reason = None
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        reason = "expected_rank_3_bh_q_or_k_d"
    elif q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        reason = "batch_heads_mismatch"
    elif q.shape[2] != k.shape[2] or k.shape[2] != v.shape[2]:
        reason = "head_dim_mismatch"
    elif k.shape[1] != v.shape[1]:
        reason = "key_value_length_mismatch"
    elif q.shape[1] == 0 or k.shape[1] == 0:
        reason = "empty_sequence"
    elif q.dtype != k.dtype or k.dtype != v.dtype:
        reason = "qkv_dtype_mismatch"
    elif q.dtype not in SUPPORTED_DTYPES:
        reason = f"unsupported_dtype_{q.dtype}"
    elif q.shape[2] not in SUPPORTED_HEAD_DIMS:
        reason = f"unsupported_head_dim_{q.shape[2]}"
    elif q.device != k.device or k.device != v.device:
        reason = "qkv_device_mismatch"
    elif q.device.type != "cuda":
        reason = f"not_cuda_or_hip_device_{q.device.type}"
    elif not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        reason = "qkv_must_be_contiguous"
    elif any(t.requires_grad for t in (q, k, v)):
        reason = "forward_only_requires_grad_not_supported"
    elif not diagnostics.is_rocm_pytorch():
        reason = "torch_version_hip_not_detected"
    elif "triton" not in sys.modules:
        triton_ok, triton_info = diagnostics.has_triton(import_module=True)
        if not triton_ok:
            reason = f"triton_unavailable_{triton_info}"

    gfx_target = diagnostics.best_effort_gfx_target() if q.device.type == "cuda" else None
    head_dim = q.shape[2] if q.ndim == 3 else None
    config = _kernel_config(head_dim, gfx_target) if head_dim in SUPPORTED_HEAD_DIMS else None
    return {
        "supported": reason is None,
        "reason": reason,
        "backend": "triton" if reason is None else None,
        "exact": True,
        "causal": False,
        "mask": None,
        "layout": "bhqd",
        "batch_heads": q.shape[0] if q.ndim == 3 else None,
        "q_tokens": q.shape[1] if q.ndim == 3 else None,
        "kv_tokens": k.shape[1] if k.ndim == 3 else None,
        "head_dim": head_dim,
        "dtype": str(q.dtype),
        "device": str(q.device),
        "gfx_target": gfx_target,
        "gfx1151_optimized": gfx_target == "gfx1151",
        "config": config,
    }


def full_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    info = validate_full_attention_triton_bh(q, k, v)
    if not info["supported"]:
        raise ValueError(f"Full attention Triton path is unavailable: {info['reason']}")

    from .triton_kernel import full_attention_triton_bh

    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[2])
    out = full_attention_triton_bh(q, k, v, scale=scale_value, **info["config"])
    if return_diagnostics:
        return out, info
    return out
