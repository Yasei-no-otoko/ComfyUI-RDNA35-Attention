from __future__ import annotations

from typing import Any, Callable

import torch


ANIMA_BLOCK_COUNT = 28
ANIMA_PISA_START_LAYER = 4
ANIMA_PISA_TOKENS = 9216
ANIMA_PISA_HEADS = 16
ANIMA_PISA_HEAD_DIM = 128


def _record(runtime_state: Any | None, **kwargs) -> None:
    if runtime_state is not None:
        runtime_state.record(**kwargs)


def validate_anima_pisa_model(model_patcher) -> None:
    diffusion_model = model_patcher.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) != ANIMA_BLOCK_COUNT:
        raise ValueError(f"validated Anima model requires {ANIMA_BLOCK_COUNT} blocks")

    for index, block in enumerate(blocks):
        self_attn = getattr(block, "self_attn", None)
        if self_attn is None:
            raise ValueError(f"Anima block {index} has no self_attn")
        if getattr(self_attn, "n_heads", None) != ANIMA_PISA_HEADS or getattr(self_attn, "head_dim", None) != ANIMA_PISA_HEAD_DIM:
            raise ValueError(f"Anima block {index} is not the validated H=16 D=128 attention profile")


def _runtime_shape(q: torch.Tensor) -> tuple[int, int, int, int]:
    return q.shape[0], q.shape[2], q.shape[1], q.shape[3]


def _eligible_reason(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, device_index: int) -> str | None:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        return "qkv_are_not_tensors"
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        return "matching_bshd_qkv_are_required"
    if q.shape[1:] != (ANIMA_PISA_TOKENS, ANIMA_PISA_HEADS, ANIMA_PISA_HEAD_DIM):
        return f"shape_{tuple(q.shape[1:])}_is_not_t9216_h16_d128"
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return "matching_bfloat16_qkv_are_required"
    if q.device != k.device or q.device != v.device:
        return "qkv_must_share_one_device"
    if q.device.type != "cuda" or q.device.index != device_index:
        return "validated_gfx1151_device_is_required"
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        return "contiguous_bshd_qkv_are_required"
    if any(tensor.requires_grad for tensor in (q, k, v)):
        return "forward_only_qkv_are_required"
    return None


def make_anima_pisa_attn_op(
    original_op: Callable,
    *,
    native_forward: Callable,
    exact_blocks: int,
    device_index: int,
    layer_index: int,
    runtime_state: Any | None = None,
) -> Callable:
    def anima_pisa_attn_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        transformer_options: dict | None = None,
        is_self_attention: bool = False,
        is_initial_transformer_block: bool | None = None,
    ) -> torch.Tensor:
        reason = _eligible_reason(q, k, v, device_index)
        if reason is not None:
            _record(runtime_state, is_self_attention=True, shape=_runtime_shape(q), fallback_reason=reason)
            return original_op(
                q,
                k,
                v,
                transformer_options=transformer_options,
            )

        q_bhtd, k_bhtd, v_bhtd = (tensor.permute(0, 2, 1, 3) for tensor in (q, k, v))
        try:
            result = native_forward(q_bhtd, k_bhtd, v_bhtd, exact_blocks=exact_blocks)
        except Exception as exc:
            _record(runtime_state, is_self_attention=True, shape=_runtime_shape(q), error=exc)
            raise RuntimeError("RDNA35 PISA failed for the eligible Anima T=9216 BF16 profile") from exc
        _record(runtime_state, layer=layer_index, is_self_attention=True, shape=_runtime_shape(q))
        return result

    return anima_pisa_attn_op


def install_anima_pisa_attention(
    model_patcher,
    *,
    native_forward: Callable,
    exact_blocks: int,
    device_index: int,
    runtime_state: Any | None = None,
) -> int:
    validate_anima_pisa_model(model_patcher)
    diffusion_model = model_patcher.get_model_object("diffusion_model")
    blocks = diffusion_model.blocks

    for index in range(ANIMA_PISA_START_LAYER, ANIMA_BLOCK_COUNT):
        self_attn = blocks[index].self_attn
        model_patcher.add_object_patch(
            f"diffusion_model.blocks.{index}.self_attn.attn_op",
            make_anima_pisa_attn_op(
                self_attn.attn_op,
                native_forward=native_forward,
                exact_blocks=exact_blocks,
                device_index=device_index,
                layer_index=index,
                runtime_state=runtime_state,
            ),
        )
    return ANIMA_BLOCK_COUNT - ANIMA_PISA_START_LAYER
