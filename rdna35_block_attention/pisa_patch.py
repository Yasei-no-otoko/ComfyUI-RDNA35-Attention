from __future__ import annotations

import logging
import math
from typing import Any, Callable

import torch

from .anima_pisa_integration import (
    ANIMA_PISA_FIRST_LAYER,
    ANIMA_PISA_LAST_LAYER,
    install_anima_pisa_attention,
    validate_anima_pisa_model,
)
from .generic_pisa import make_generic_pisa_override
from .pisa_runtime import PISA_RUNTIME_ATTACHMENT, PISARuntimeState


LOG_PREFIX = "RDNA35 PISA"
TOKEN_POLICIES = {
    "anima_1536_spatial": frozenset((9216,)),
    "auto_9216": frozenset((9216,)),
}
SPATIAL_BLOCK_EDGE = 8
ANIMA_TOKEN_SHAPE = (1, 96, 96)


def _run_pisa_attention(
    native_forward: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_budget: float,
    scale: float | None,
):
    total_blocks = math.ceil(q.shape[-2] / 64)
    exact_blocks = min(total_blocks, math.ceil(exact_budget * total_blocks))
    return native_forward(q, k, v, exact_blocks, scale=scale)


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


def _pisa_reject_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None,
    attn_precision: torch.dtype | None,
    skip_reshape: bool,
    skip_output_reshape: bool,
    allowed_tokens: frozenset[int],
    expected_token_shape: tuple[int, ...],
    device_index: int,
    kwargs: dict[str, Any],
) -> str | None:
    if kwargs.get("is_self_attention") is not True:
        return "call_is_not_explicitly_self_attention"
    if mask is not None:
        return "attention_mask_is_not_supported"
    if not skip_reshape:
        return "pre_reshaped_qkv_is_required"
    if skip_output_reshape:
        return "merged_attention_output_is_required"
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        return "qkv_are_not_tensors"
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        return "matching_bhtd_qkv_are_required"
    if q.shape[0] <= 0 or q.shape[1] != heads or q.shape[-1] != 128:
        return "unsupported_batch_heads_or_head_dim"
    if q.shape[-2] not in allowed_tokens:
        return f"tokens_{q.shape[-2]}_are_not_enabled"
    token_shape = kwargs.get("attention_token_shape")
    if not isinstance(token_shape, (tuple, list)) or tuple(token_shape) != expected_token_shape:
        return f"token_shape_{token_shape}_is_not_{expected_token_shape}"
    side = math.isqrt(q.shape[-2])
    if side * side != q.shape[-2] or side % SPATIAL_BLOCK_EDGE:
        return "square_spatial_tokens_divisible_by_8_are_required"
    if kwargs.get("is_initial_transformer_block") is not False:
        return "initial_or_unmarked_transformer_block"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return "matching_bfloat16_qkv_are_required"
    if q.device != k.device or q.device != v.device:
        return "qkv_must_share_one_device"
    if q.device.type != "cuda":
        return "rocm_cuda_device_is_required"
    if q.device.index != device_index:
        return f"rocm_device_{q.device.index}_is_not_validated_device_{device_index}"
    expected_stride = (q.shape[1] * q.shape[2] * q.shape[3], q.shape[3], q.shape[1] * q.shape[3], 1)
    if any(tensor.stride() != expected_stride for tensor in (q, k, v)):
        return "bhtd_views_over_contiguous_bthd_storage_are_required"
    if any(tensor.requires_grad for tensor in (q, k, v)):
        return "forward_only_qkv_are_required"
    if attn_precision == torch.float32:
        return "float32_attention_precision_is_not_supported"
    if kwargs.get("enable_gqa", False):
        return "gqa_is_not_supported"
    return None


def make_pisa_attention_override(
    *,
    exact_budget: float,
    allowed_tokens: frozenset[int],
    expected_token_shape: tuple[int, ...],
    device_index: int,
    native_forward: Callable,
    verbose_fallbacks: bool,
    previous_override: Callable | None,
) -> Callable:
    def attention_override(original_func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        reject_reason = _pisa_reject_reason(
            q,
            k,
            v,
            heads,
            mask,
            attn_precision,
            skip_reshape,
            skip_output_reshape,
            allowed_tokens,
            expected_token_shape,
            device_index,
            kwargs,
        )
        if reject_reason is not None:
            if verbose_fallbacks:
                logging.info("%s: original attention used: %s", LOG_PREFIX, reject_reason)
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

        return _run_pisa_attention(
            native_forward,
            q,
            k,
            v,
            exact_budget,
            kwargs.get("scale"),
        )

    return attention_override


def patch_model_pisa_attention(
    model,
    *,
    enabled: bool,
    exact_budget: float,
    token_policy: str,
    verbose_fallbacks: bool,
    start_layer: int = ANIMA_PISA_FIRST_LAYER,
    end_layer: int = ANIMA_PISA_LAST_LAYER,
):
    if not enabled:
        return model, "disabled; model returned unchanged"
    if not 0.0 <= exact_budget <= 1.0:
        return model, f"invalid exact_budget={exact_budget}; model returned unchanged"
    if token_policy not in TOKEN_POLICIES:
        return model, f"unsupported token_policy={token_policy}; model returned unchanged"
    if not 0 <= start_layer <= end_layer < 28:
        return model, f"invalid Anima layer range {start_layer}:{end_layer}; model returned unchanged"
    if not hasattr(model, "clone") or not hasattr(model, "model_options"):
        return model, "MODEL does not expose ComfyUI ModelPatcher clone/model_options; model returned unchanged"

    device_index, device_error = _gfx1151_device_index()
    if device_index is None:
        return model, f"{device_error}; model returned unchanged"

    is_anima = False
    if hasattr(model, "get_model_object") and hasattr(model, "add_object_patch"):
        try:
            validate_anima_pisa_model(model)
            is_anima = True
        except (AttributeError, TypeError, ValueError):
            pass

    if not is_anima:
        model_clone = model.clone()
        runtime_state = PISARuntimeState(armed=True)
        if hasattr(model_clone, "set_attachments"):
            model_clone.set_attachments(PISA_RUNTIME_ATTACHMENT, runtime_state)
        transformer_options = model_clone.model_options.setdefault("transformer_options", {})
        previous_override = transformer_options.get("optimized_attention_override")
        transformer_options["optimized_attention_override"] = make_generic_pisa_override(
            exact_budget=exact_budget,
            device_index=device_index,
            previous_override=previous_override,
            runtime_state=runtime_state,
            validate_output=verbose_fallbacks,
        )
        info = "model-local generic gfx1151 PISA installed for explicit self-attention with T>=8192 and arbitrary head dimension"
        if previous_override is not None:
            info += "; existing optimized_attention_override is chained for fallback"
        return model_clone, info

    try:
        import rdna35_pisa_ck

        build_info = rdna35_pisa_ck.build_info()
        capabilities = rdna35_pisa_ck.capabilities()
        rdna35_pisa_ck.prepare()
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        return model, f"rdna35-pisa-ck is unavailable ({type(exc).__name__}: {exc}); model returned unchanged"
    if build_info.get("api") != 6:
        return model, f"rdna35-pisa-ck API 6 is required, got {build_info.get('api')}; model returned unchanged"
    exact_blocks = math.ceil(exact_budget * 144)
    validated_exact_blocks = capabilities.get("spatial_sparse_exact_blocks")
    if not isinstance(validated_exact_blocks, (tuple, list)) or exact_blocks not in validated_exact_blocks:
        return model, (
            f"exact_budget={exact_budget:.6g} selects {exact_blocks} blocks, but this wheel validates sparse blocks "
            f"{validated_exact_blocks}; model returned unchanged"
        )

    model_clone = model.clone()
    if hasattr(model_clone, "get_model_object") and hasattr(model_clone, "add_object_patch"):
        runtime_state = PISARuntimeState(armed=True, expected_layers=tuple(range(start_layer, end_layer + 1)))
        model_clone.set_attachments(PISA_RUNTIME_ATTACHMENT, runtime_state)
        try:
            patched_blocks = install_anima_pisa_attention(
                model_clone,
                native_forward=rdna35_pisa_ck.forward_spatial_bhtd,
                exact_blocks=exact_blocks,
                device_index=device_index,
                first_layer=start_layer,
                last_layer=end_layer,
                runtime_state=runtime_state if verbose_fallbacks else None,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            return model, f"validated Anima direct PISA integration is unavailable ({exc}); model returned unchanged"
        return model_clone, (
            f"model-local gfx1151 direct PISA installed on Anima self-attention blocks {start_layer}:{end_layer} "
            f"at T=9216; exact_blocks={exact_blocks}; patched_blocks={patched_blocks}; "
            f"runtime_accounting={'enabled' if verbose_fallbacks else 'disabled'}"
        )

    transformer_options = model_clone.model_options.setdefault("transformer_options", {})
    previous_override = transformer_options.get("optimized_attention_override")
    transformer_options["optimized_attention_override"] = make_pisa_attention_override(
        exact_budget=exact_budget,
        allowed_tokens=TOKEN_POLICIES[token_policy],
        expected_token_shape=ANIMA_TOKEN_SHAPE,
        device_index=device_index,
        native_forward=rdna35_pisa_ck.forward_spatial_bhtd,
        verbose_fallbacks=verbose_fallbacks,
        previous_override=previous_override,
    )

    tokens = ",".join(str(value) for value in sorted(TOKEN_POLICIES[token_policy]))
    info = (
        f"model-local gfx1151 PISA override installed for Anima spatial self-attention at T={tokens}; "
        f"exact_budget={exact_budget:.6g}; layers={start_layer}:{end_layer}"
    )
    if previous_override is not None:
        info += "; existing optimized_attention_override is chained for fallback"
    return model_clone, info
