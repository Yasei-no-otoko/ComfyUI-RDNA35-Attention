from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from .dispatch import fixed_block_attention


LOG_PREFIX = "RDNA35 FixedBlockAttention"


def _fallback_call(previous_override: Callable | None, original_func: Callable, *args, **kwargs):
    if previous_override is not None:
        return previous_override(original_func, *args, **kwargs)
    return original_func(*args, **kwargs)


def _has_fixed_block_marker(kwargs: dict[str, Any]) -> bool:
    transformer_options = kwargs.get("transformer_options") or {}
    return (
        kwargs.get("rdna35_attention_semantics") == "fixed_block_diagonal"
        or transformer_options.get("rdna35_attention_semantics") == "fixed_block_diagonal"
    )


def _declares_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kwargs: dict[str, Any]) -> bool:
    transformer_options = kwargs.get("transformer_options") or {}
    if kwargs.get("rdna35_is_self_attention") is True:
        return True
    if transformer_options.get("rdna35_is_self_attention") is True:
        return True
    try:
        return q.data_ptr() == k.data_ptr() == v.data_ptr()
    except Exception:
        return False


def _format_call_reason(info: dict[str, Any]) -> str:
    reason = info.get("fallback_reason")
    backend = info.get("backend")
    if reason:
        return f"{backend}: {reason}"
    return str(backend)


def make_attention_override(
    *,
    mode: str,
    semantic_mode: str,
    block_size: int,
    causal: bool,
    verbose_fallbacks: bool,
    previous_override: Callable | None,
) -> Callable:
    def attention_override(original_func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        marked_fixed = _has_fixed_block_marker(kwargs)
        can_apply = marked_fixed

        if semantic_mode == "experimental_force_block_local" and not can_apply:
            can_apply = _declares_self_attention(q, k, v, kwargs)

        if not can_apply:
            if verbose_fallbacks:
                logging.info("%s: original attention used; call is not proven fixed block-diagonal self-attention.", LOG_PREFIX)
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

        if semantic_mode == "experimental_force_block_local" and not marked_fixed and not _declares_self_attention(q, k, v, kwargs):
            if verbose_fallbacks:
                logging.info("%s: original attention used; experimental mode could not prove self-attention.", LOG_PREFIX)
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

        try:
            if skip_reshape:
                out, info = fixed_block_attention(
                    q,
                    k,
                    v,
                    block_size=block_size,
                    causal=causal,
                    layout="bhtd",
                    mode=mode,
                    return_diagnostics=True,
                    mask=mask,
                )
                if not skip_output_reshape:
                    batch, n_heads, tokens, head_dim = out.shape
                    out = out.transpose(1, 2).contiguous().reshape(batch, tokens, n_heads * head_dim)
            else:
                out, info = fixed_block_attention(
                    q,
                    k,
                    v,
                    block_size=block_size,
                    causal=causal,
                    layout="bthd",
                    heads=heads,
                    mode=mode,
                    return_diagnostics=True,
                    mask=mask,
                )
                if skip_output_reshape:
                    batch, tokens, channels = out.shape
                    head_dim = channels // heads
                    out = out.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3).contiguous()

            if verbose_fallbacks and info.get("fallback_reason"):
                logging.info("%s: %s", LOG_PREFIX, _format_call_reason(info))
            return out
        except Exception as exc:
            logging.warning("%s: fixed block attention failed; original attention used: %s: %s", LOG_PREFIX, type(exc).__name__, exc)
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

    return attention_override


def patch_model_attention(
    model,
    *,
    enabled: bool,
    mode: str,
    semantic_mode: str,
    block_size: int,
    causal: bool,
    verbose_fallbacks: bool,
):
    if not enabled:
        return model, "disabled; model returned unchanged"

    if block_size != 64:
        return model, f"unsupported block_size={block_size}; only 64 is implemented, model returned unchanged"

    if semantic_mode not in {"exact_only", "experimental_force_block_local"}:
        return model, f"unsupported semantic_mode={semantic_mode}; model returned unchanged"

    if not hasattr(model, "clone") or not hasattr(model, "model_options"):
        return model, "MODEL does not expose ComfyUI ModelPatcher clone/model_options; model returned unchanged"

    model_clone = model.clone()
    transformer_options = model_clone.model_options.setdefault("transformer_options", {})
    previous_override = transformer_options.get("optimized_attention_override")
    transformer_options["optimized_attention_override"] = make_attention_override(
        mode=mode,
        semantic_mode=semantic_mode,
        block_size=block_size,
        causal=causal,
        verbose_fallbacks=verbose_fallbacks,
        previous_override=previous_override,
    )

    if semantic_mode == "exact_only":
        info = (
            "model-local optimized_attention_override installed in exact_only mode; "
            "normal ComfyUI attention calls are left unchanged unless they explicitly declare "
            "rdna35_attention_semantics='fixed_block_diagonal'"
        )
    else:
        info = (
            "model-local optimized_attention_override installed in experimental_force_block_local mode; "
            "only calls explicitly marked as fixed block-diagonal or self-attention are eligible; "
            "unproven calls fall back to the original attention backend"
        )

    if previous_override is not None:
        info += "; existing optimized_attention_override is chained for fallback"
    return model_clone, info
