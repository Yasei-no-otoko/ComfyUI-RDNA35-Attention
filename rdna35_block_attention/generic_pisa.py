from __future__ import annotations

import functools
import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention


BLOCK_SIZE = 64
MIN_TOKENS = 8192


@functools.lru_cache(maxsize=8)
def _full_block_mask(device: torch.device) -> BlockMask:
    return BlockMask.from_kv_blocks(
        torch.ones((1, 1, 1), device=device, dtype=torch.int32),
        torch.zeros((1, 1, 1, 1), device=device, dtype=torch.int32),
        BLOCK_SIZE=1 << 30,
        seq_lengths=(1, 1),
    )


def _flex_kernels(scale: float):
    options = {
        "fwd_BLOCK_M": 64,
        "fwd_BLOCK_N": 32,
        "fwd_num_stages": 1,
        "fwd_num_warps": 4,
        "ROWS_GUARANTEED_SAFE": True,
    }

    def exact_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: BlockMask):
        output, auxiliary = flex_attention(
            q[:, None],
            k[:, None],
            v[:, None],
            block_mask=block_mask,
            scale=scale,
            return_aux=AuxRequest(lse=True),
            kernel_options=options,
        )
        return output[:, 0], auxiliary.lse[:, 0]

    def approximate_attention(
        q: torch.Tensor,
        k_centroids: torch.Tensor,
        v_means: torch.Tensor,
        selected: torch.Tensor,
        log_lengths: torch.Tensor,
        block_mask: BlockMask,
    ):
        def score_mod(score, batch, head, query_index, key_index):
            return torch.where(
                selected[batch, query_index // BLOCK_SIZE, key_index],
                -torch.inf,
                score + log_lengths[key_index],
            )

        output, auxiliary = flex_attention(
            q[:, None],
            k_centroids[:, None],
            v_means[:, None],
            block_mask=block_mask,
            score_mod=score_mod,
            scale=scale,
            return_aux=AuxRequest(lse=True),
            kernel_options=options,
        )
        return output[:, 0], auxiliary.lse[:, 0]

    return torch.compile(exact_attention, fullgraph=True, dynamic=False), torch.compile(approximate_attention, fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def _compiled_flex_kernels(tokens: int, head_dim: int, dtype: torch.dtype, device_index: int, scale: float):
    del tokens, head_dim, dtype, device_index
    return _flex_kernels(scale)


def _block_stats(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, return_backend: bool = False):
    batch_heads, tokens, head_dim = q.shape
    blocks = math.ceil(tokens / BLOCK_SIZE)
    padded_tokens = blocks * BLOCK_SIZE
    pad = padded_tokens - tokens
    if pad:
        q_padded = F.pad(q, (0, 0, 0, pad))
        k_padded = F.pad(k, (0, 0, 0, pad))
        v_padded = F.pad(v, (0, 0, 0, pad))
    else:
        q_padded, k_padded, v_padded = q, k, v

    lengths = torch.full((blocks,), BLOCK_SIZE, device=q.device, dtype=torch.float32)
    lengths[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    try:
        import rdna35_pisa_ck

        capabilities = rdna35_pisa_ck.capabilities()
        if q.shape[-1] in capabilities.get("hyd_stats_head_dims", ()) and q.dtype in capabilities.get("hyd_stats_dtypes", ()):
            result = (*rdna35_pisa_ck.block_stats_hyd(q, k, v), lengths)
            return (*result, "ck_hyd") if return_backend else result
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
        pass

    q_blocks = q_padded.unflatten(1, (blocks, BLOCK_SIZE)).float()
    denominator = lengths[None, :, None]
    q_centroids = q_blocks.sum(dim=2) / denominator
    from .pisa_kernel import pisa_prepare_triton

    k_centroids, v_sums, h_sum, _ = pisa_prepare_triton(k, v, block_size=BLOCK_SIZE)
    v_means = v_sums / denominator
    result = (q_centroids, k_centroids, v_means, h_sum, lengths)
    return (*result, "triton_hyd") if return_backend else result


def generic_pisa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    exact_budget: float,
    scale: float | None = None,
    return_backend: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, str]:
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("generic PISA requires matching [BH,T,D] q/k/v")
    if q.shape[1] < MIN_TOKENS:
        raise ValueError(f"generic PISA requires at least {MIN_TOKENS} tokens")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("generic PISA requires matching fp16/bf16 q/k/v")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("generic PISA requires q/k/v on one CUDA/HIP device")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("generic PISA requires contiguous q/k/v")

    batch_heads, tokens, head_dim = q.shape
    blocks = math.ceil(tokens / BLOCK_SIZE)
    exact_blocks = min(blocks, max(1, math.ceil(exact_budget * blocks)))
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    if exact_blocks == blocks:
        output = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], scale=scale_value).squeeze(1)
        return (output, "dense_sdpa") if return_backend else output

    q_centroids, k_centroids_float, v_means, h_sum, lengths, backend = _block_stats(q, k, v, return_backend=True)
    route_scores = torch.bmm(q_centroids.float(), k_centroids_float.float().transpose(1, 2))
    route_scores.mul_(scale_value)
    indices = torch.topk(route_scores, exact_blocks, dim=-1).indices.sort(dim=-1).values
    selected = torch.zeros((batch_heads, blocks, blocks), device=q.device, dtype=torch.bool)
    selected.scatter_(2, indices, True)

    padded_indices = F.pad(indices[:, None].to(torch.int32), (0, blocks - exact_blocks))
    exact_counts = torch.full((batch_heads, 1, blocks), exact_blocks, device=q.device, dtype=torch.int32)
    empty_counts = torch.zeros_like(exact_counts)
    exact_mask = BlockMask.from_kv_blocks(
        empty_counts,
        padded_indices,
        exact_counts,
        padded_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        seq_lengths=(tokens, tokens),
        compute_q_blocks=False,
    )
    exact_attention, approximate_attention = _compiled_flex_kernels(tokens, head_dim, q.dtype, q.get_device(), scale_value)
    exact_output, exact_lse = exact_attention(q, k, v, exact_mask)
    approximate_output, approximate_lse = approximate_attention(
        q,
        k_centroids_float.to(q.dtype),
        v_means,
        selected,
        lengths.log(),
        _full_block_mask(q.device),
    )

    correction = torch.bmm(q, h_sum.to(q.dtype), out_dtype=torch.float32)
    correction.mul_(scale_value / tokens)
    total_lse = torch.logaddexp(exact_lse, approximate_lse)
    exact_weight = torch.exp(exact_lse - total_lse)
    approximate_weight = torch.exp(approximate_lse - total_lse)
    output = exact_output.float() * exact_weight[..., None]
    output.add_(approximate_output.float() * approximate_weight[..., None])
    output.add_(correction * approximate_weight[..., None])
    output = output.to(q.dtype)
    return (output, backend) if return_backend else output


def make_generic_pisa_override(
    *,
    exact_budget: float,
    device_index: int,
    previous_override: Callable | None,
    runtime_state,
) -> Callable:
    def fallback(original_func, *args, **kwargs):
        if previous_override is not None:
            return previous_override(original_func, *args, **kwargs)
        return original_func(*args, **kwargs)

    def attention_override(original_func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        reason = None
        if kwargs.get("is_self_attention") is not True:
            reason = "not_explicit_self_attention"
        elif mask is not None:
            reason = "attention_mask_is_not_supported"
        elif not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
            reason = "qkv_are_not_tensors"
        elif q.device.type != "cuda" or q.device.index != device_index:
            reason = "validated_gfx1151_device_is_required"
        elif q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
            reason = "matching_fp16_or_bf16_qkv_are_required"

        if reason is None:
            if skip_reshape:
                if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
                    reason = "matching_bhtd_qkv_are_required"
                else:
                    batch, actual_heads, tokens, head_dim = q.shape
                    if actual_heads != heads:
                        reason = "head_count_mismatch"
                    else:
                        q_bhtd, k_bhtd, v_bhtd = q, k, v
            else:
                if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] % heads:
                    reason = "matching_btc_qkv_divisible_by_heads_are_required"
                else:
                    batch, tokens, channels = q.shape
                    head_dim = channels // heads
                    q_bhtd, k_bhtd, v_bhtd = (
                        tensor.unflatten(2, (heads, head_dim)).transpose(1, 2) for tensor in (q, k, v)
                    )
            if reason is None and tokens < MIN_TOKENS:
                reason = f"tokens_{tokens}_below_{MIN_TOKENS}"

        shape = None
        if isinstance(q, torch.Tensor) and q.ndim >= 2:
            shape = tuple(q.shape)
        if reason is not None:
            runtime_state.record(is_self_attention=kwargs.get("is_self_attention"), shape=shape, fallback_reason=reason)
            return fallback(original_func, q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

        q_flat, k_flat, v_flat = (tensor.contiguous().flatten(0, 1) for tensor in (q_bhtd, k_bhtd, v_bhtd))
        budget = exact_budget if head_dim == 128 and q.dtype == torch.bfloat16 else max(0.25, exact_budget)
        try:
            output, backend = generic_pisa_attention(
                q_flat,
                k_flat,
                v_flat,
                exact_budget=budget,
                scale=kwargs.get("scale"),
                return_backend=True,
            )
        except Exception as exc:
            runtime_state.record(is_self_attention=True, shape=(batch, heads, tokens, head_dim), error=exc)
            raise RuntimeError(f"RDNA35 generic PISA failed for B={batch} H={heads} T={tokens} D={head_dim}") from exc
        runtime_state.record(layer=-1, is_self_attention=True, shape=(batch, heads, tokens, head_dim), backend=backend)
        output = output.unflatten(0, (batch, heads))
        if skip_output_reshape:
            return output
        return output.transpose(1, 2).contiguous().flatten(2)

    return attention_override
