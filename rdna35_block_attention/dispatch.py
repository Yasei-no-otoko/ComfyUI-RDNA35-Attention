from __future__ import annotations

from typing import Any

import torch

from . import diagnostics
from .reference import fixed_block_attention_ref, normalize_qkv_to_bh_t_d


SUPPORTED_HEAD_DIMS = {32, 64, 128}
SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def _fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    scale: float | None,
    causal: bool,
    layout: str,
    heads: int | None,
    mask: torch.Tensor | None,
    reason: str,
    return_diagnostics: bool,
) -> Any:
    out = fixed_block_attention_ref(
        q,
        k,
        v,
        block_size=block_size,
        scale=scale,
        causal=causal,
        layout=layout,
        heads=heads,
        mask=mask,
    )
    info = {
        "backend": "reference",
        "fallback_reason": reason,
        "optimized": False,
    }
    if return_diagnostics:
        return out, info
    return out


def _triton_reject_reason(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    *,
    block_size: int,
    mask: torch.Tensor | None,
) -> str | None:
    if mask is not None:
        return "arbitrary_mask_uses_reference"
    if any(t.requires_grad for t in (q_flat, k_flat, v_flat)):
        return "requires_grad_uses_reference"
    if block_size != 64:
        return f"unsupported_block_size_{block_size}"
    if q_flat.shape != k_flat.shape or q_flat.shape != v_flat.shape:
        return "qkv_shape_mismatch"
    if q_flat.dtype not in SUPPORTED_DTYPES:
        return f"unsupported_dtype_{q_flat.dtype}"
    if q_flat.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return f"unsupported_head_dim_{q_flat.shape[-1]}"
    if q_flat.device != k_flat.device or q_flat.device != v_flat.device:
        return "qkv_device_mismatch"
    if q_flat.device.type != "cuda":
        return f"not_cuda_or_hip_device_{q_flat.device.type}"
    if not diagnostics.is_rocm_pytorch():
        return "torch_version_hip_not_detected"
    triton_ok, triton_info = diagnostics.has_triton(import_module=True)
    if not triton_ok:
        return f"triton_unavailable_{triton_info}"
    return None


def fixed_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int = 64,
    scale: float | None = None,
    causal: bool = False,
    layout: str = "auto",
    heads: int | None = None,
    mode: str = "auto",
    allow_approximate: bool = False,
    return_diagnostics: bool = False,
    mask: torch.Tensor | None = None,
) -> Any:
    del allow_approximate  # The function is exact for fixed block-diagonal semantics.

    if mode not in {"auto", "reference", "triton"}:
        raise ValueError("mode must be one of: auto, reference, triton.")

    if mode == "reference":
        return _fallback(
            q,
            k,
            v,
            block_size=block_size,
            scale=scale,
            causal=causal,
            layout=layout,
            heads=heads,
            mask=mask,
            reason="mode_reference",
            return_diagnostics=return_diagnostics,
        )

    try:
        q_flat, k_flat, v_flat, layout_info = normalize_qkv_to_bh_t_d(q, k, v, layout=layout, heads=heads)
    except Exception as exc:
        return _fallback(
            q,
            k,
            v,
            block_size=block_size,
            scale=scale,
            causal=causal,
            layout=layout,
            heads=heads,
            mask=mask,
            reason=f"layout_normalization_failed_{type(exc).__name__}: {exc}",
            return_diagnostics=return_diagnostics,
        )

    reject_reason = _triton_reject_reason(q_flat, k_flat, v_flat, block_size=block_size, mask=mask)
    if reject_reason is not None:
        return _fallback(
            q,
            k,
            v,
            block_size=block_size,
            scale=scale,
            causal=causal,
            layout=layout,
            heads=heads,
            mask=mask,
            reason=reject_reason if mode == "auto" else f"requested_triton_but_{reject_reason}",
            return_diagnostics=return_diagnostics,
        )

    try:
        from .triton_kernel import fixed_block_attention_triton_bh

        out_flat = fixed_block_attention_triton_bh(
            q_flat,
            k_flat,
            v_flat,
            block_size=block_size,
            scale=scale,
            causal=causal,
        )
        out = layout_info.restore(out_flat)
        info = {
            "backend": "triton",
            "fallback_reason": None,
            "optimized": True,
            "layout": layout_info.layout,
            "tokens": layout_info.tokens,
            "head_dim": layout_info.head_dim,
        }
        if return_diagnostics:
            return out, info
        return out
    except Exception as exc:
        return _fallback(
            q,
            k,
            v,
            block_size=block_size,
            scale=scale,
            causal=causal,
            layout=layout,
            heads=heads,
            mask=mask,
            reason=f"triton_runtime_error_{type(exc).__name__}: {exc}",
            return_diagnostics=return_diagnostics,
        )
