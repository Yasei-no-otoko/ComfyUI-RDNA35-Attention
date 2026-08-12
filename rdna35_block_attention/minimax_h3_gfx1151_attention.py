from __future__ import annotations

import logging
from typing import Any, Callable

import torch


LOG_PREFIX = "RDNA35 MiniMax-H3 gfx1151 Attention"
MINIMAX_H3_HEADS = 56
MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_MIN_PACK_TOKENS = 4608
MINIMAX_H3_PACK_ALL_TOKENS = 12000
MINIMAX_H3_MAX_PACK_TOKENS = 16500


def _fallback_call(previous_override: Callable | None, original_func: Callable, *args, **kwargs):
    if previous_override is not None:
        return previous_override(original_func, *args, **kwargs)
    return original_func(*args, **kwargs)


def _gfx1151_device_index() -> tuple[int | None, str | None]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return None, "PyTorch ROCm device is unavailable"
    try:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
    except (RuntimeError, AssertionError) as exc:
        return None, f"could not query the current ROCm device ({type(exc).__name__}: {exc})"
    target = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if target != "gfx1151":
        return None, f"gfx1151 is required, got {target or 'unknown'}"
    return device_index, None


def _locate_minimax_h3(model: Any) -> Any | None:
    if hasattr(model, "get_model_object"):
        try:
            diffusion_model = model.get_model_object("diffusion_model")
        except (AttributeError, KeyError):
            diffusion_model = None
        if diffusion_model is not None:
            return diffusion_model
    outer = getattr(model, "model", None)
    return getattr(outer, "diffusion_model", None)


def _is_native_minimax_h3(model: Any) -> bool:
    diffusion_model = _locate_minimax_h3(model)
    return (
        diffusion_model is not None
        and type(diffusion_model).__module__ == "comfy.ldm.minimax.model"
        and type(diffusion_model).__name__ == "MiniMaxH3Model"
        and hasattr(diffusion_model, "blocks")
        and hasattr(diffusion_model, "hidden_size")
    )


def minimax_h3_pack_profile(tokens: int) -> str:
    return "qkv" if tokens >= MINIMAX_H3_PACK_ALL_TOKENS else "kv"


def pack_minimax_h3_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    profile = minimax_h3_pack_profile(q.shape[2])
    if profile == "qkv":
        packed = torch.stack((q, k, v))
        return packed[0], packed[1], packed[2], profile
    packed = torch.stack((k, v))
    return q, packed[0], packed[1], profile


def _reject_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None,
    skip_reshape: bool,
    skip_output_reshape: bool,
    device_index: int,
    kwargs: dict[str, Any],
) -> str | None:
    if mask is not None:
        return "attention_mask_is_not_supported"
    if not skip_reshape or skip_output_reshape:
        return "minimax_h3_merged_output_contract_is_required"
    if kwargs.get("enable_gqa", False):
        return "gqa_is_not_supported"
    if kwargs.get("is_causal", False) or kwargs.get("causal", False):
        return "causal_attention_is_not_supported"
    if kwargs.get("attn_bias") is not None:
        return "attention_bias_is_not_supported"
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        return "qkv_are_not_tensors"
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        return "matching_bhtd_qkv_are_required"
    if q.shape[0] != 1 or q.shape[1] != MINIMAX_H3_HEADS or heads != MINIMAX_H3_HEADS or q.shape[3] != MINIMAX_H3_HEAD_DIM:
        return "minimax_h3_b1_h56_d128_is_required"
    if q.shape[2] < MINIMAX_H3_MIN_PACK_TOKENS:
        return f"tokens_below_{MINIMAX_H3_MIN_PACK_TOKENS}"
    if q.shape[2] > MINIMAX_H3_MAX_PACK_TOKENS:
        return f"tokens_above_{MINIMAX_H3_MAX_PACK_TOKENS}"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return "matching_bfloat16_qkv_are_required"
    if q.device != k.device or q.device != v.device or q.device.type != "cuda":
        return "matching_rocm_qkv_device_is_required"
    tensor_device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    if tensor_device_index != device_index:
        return "qkv_are_not_on_the_validated_gfx1151_device"
    if any(tensor.requires_grad for tensor in (q, k, v)):
        return "forward_only_requires_grad_not_supported"
    if any(tensor.stride(3) != 1 for tensor in (q, k, v)):
        return "contiguous_head_dimension_is_required"
    if q.is_contiguous() and k.is_contiguous() and v.is_contiguous():
        return "qkv_are_already_packed"
    return None


def make_minimax_h3_gfx1151_attention_override(
    *,
    device_index: int,
    verbose_fallbacks: bool,
    previous_override: Callable | None,
) -> Callable:
    def attention_override(original_func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        reason = _reject_reason(
            q,
            k,
            v,
            heads,
            mask,
            skip_reshape,
            skip_output_reshape,
            device_index,
            kwargs,
        )
        if reason is not None:
            if verbose_fallbacks:
                logging.info("%s: existing backend used: %s", LOG_PREFIX, reason)
            return _fallback_call(
                previous_override,
                original_func,
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        packed_q, packed_k, packed_v, profile = pack_minimax_h3_qkv(q, k, v)
        if verbose_fallbacks:
            logging.info("%s: packed %s at T=%d", LOG_PREFIX, profile.upper(), q.shape[2])
        return _fallback_call(
            previous_override,
            original_func,
            packed_q,
            packed_k,
            packed_v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    return attention_override


def patch_minimax_h3_gfx1151_attention(
    model,
    *,
    enabled: bool,
    verbose_fallbacks: bool,
):
    if not enabled:
        return model, "disabled; model returned unchanged"
    if not hasattr(model, "clone") or not hasattr(model, "model_options"):
        return model, "MODEL does not expose ComfyUI ModelPatcher clone/model_options; model returned unchanged"
    if not _is_native_minimax_h3(model):
        return model, "native ComfyUI MiniMaxH3Model was not found; model returned unchanged"

    device_index, device_error = _gfx1151_device_index()
    if device_index is None:
        return model, f"{device_error}; model returned unchanged"

    previous_override = model.model_options.get("transformer_options", {}).get("optimized_attention_override")
    if previous_override is not None and getattr(previous_override, "container_function", None) is not None:
        return model, "container-aware optimized_attention_override is already installed; model returned unchanged"

    model_clone = model.clone()
    transformer_options = model_clone.model_options.setdefault("transformer_options", {})
    transformer_options["optimized_attention_override"] = make_minimax_h3_gfx1151_attention_override(
        device_index=device_index,
        verbose_fallbacks=verbose_fallbacks,
        previous_override=previous_override,
    )
    info = (
        "model-local MiniMax-H3 gfx1151 QKV packing override installed for BF16 B=1,H=56,D=128 "
        f"with {MINIMAX_H3_MIN_PACK_TOKENS}<=T<={MINIMAX_H3_MAX_PACK_TOKENS}; "
        f"KV is packed below T={MINIMAX_H3_PACK_ALL_TOKENS}, "
        "QKV is packed at longer sequences, and the existing attention backend remains responsible for exact attention"
    )
    if previous_override is not None:
        info += "; existing optimized_attention_override is chained after packing and for fallback"
    return model_clone, info
