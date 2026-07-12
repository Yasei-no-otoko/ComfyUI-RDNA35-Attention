from __future__ import annotations

import functools
import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention


BLOCK_SIZE = 64
MIN_TOKENS = 8192
SPATIAL_BLOCK_EDGE = 8


def _pack_spatial_blocks(tensor: torch.Tensor, side: int) -> torch.Tensor:
    grid = side // SPATIAL_BLOCK_EDGE
    return tensor.reshape(tensor.shape[0], tensor.shape[1], grid, SPATIAL_BLOCK_EDGE, grid, SPATIAL_BLOCK_EDGE, tensor.shape[-1]).permute(0, 1, 2, 4, 3, 5, 6).reshape_as(tensor).contiguous()


def _unpack_spatial_blocks(tensor: torch.Tensor, side: int) -> torch.Tensor:
    grid = side // SPATIAL_BLOCK_EDGE
    return tensor.reshape(tensor.shape[0], tensor.shape[1], grid, grid, SPATIAL_BLOCK_EDGE, SPATIAL_BLOCK_EDGE, tensor.shape[-1]).permute(0, 1, 2, 4, 3, 5, 6).reshape_as(tensor).contiguous()


def _pack_video_blocks(tensor: torch.Tensor, grid_sizes: tuple[int, int, int]) -> torch.Tensor:
    frames, height, width = grid_sizes
    batch, heads, tokens, head_dim = tensor.shape
    if height != width or tokens != frames * height * width:
        raise ValueError("video token shape does not match a square spatial grid")
    per_frame = tensor.unflatten(2, (frames, height * width)).transpose(1, 2).flatten(0, 1)
    return _pack_spatial_blocks(per_frame, height).unflatten(0, (batch, frames)).transpose(1, 2).flatten(2, 3)


def _unpack_video_blocks(tensor: torch.Tensor, grid_sizes: tuple[int, int, int]) -> torch.Tensor:
    frames, height, width = grid_sizes
    batch, heads, tokens, head_dim = tensor.shape
    if height != width or tokens != frames * height * width:
        raise ValueError("video token shape does not match a square spatial grid")
    per_frame = tensor.unflatten(2, (frames, height * width)).transpose(1, 2).flatten(0, 1)
    return _unpack_spatial_blocks(per_frame, height).unflatten(0, (batch, frames)).transpose(1, 2).flatten(2, 3)


def _flex_kernels(scale: float):
    exact_options = {
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
            kernel_options=exact_options,
        )
        return output[:, 0], auxiliary.lse[:, 0]

    return torch.compile(exact_attention, fullgraph=True, dynamic=False)


def _approximate_kernel(scale: float):
    def approximate_attention(q: torch.Tensor, k_centroids: torch.Tensor, v_means: torch.Tensor, selected: torch.Tensor, log_lengths: torch.Tensor):
        scores = torch.bmm(q.float(), k_centroids.float().transpose(1, 2))
        scores.mul_(scale)
        selected_tokens = selected.repeat_interleave(BLOCK_SIZE, dim=1)[:, : q.shape[1]]
        scores.masked_fill_(selected_tokens, -torch.inf)
        scores.add_(log_lengths[None, None])
        lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.bmm(probabilities, v_means.float()).to(q.dtype)
        return output, lse

    return torch.compile(approximate_attention, fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def _compiled_flex_kernels(tokens: int, head_dim: int, dtype: torch.dtype, device_index: int, scale: float):
    del tokens, head_dim, dtype, device_index
    return _flex_kernels(scale)


@functools.lru_cache(maxsize=32)
def _compiled_approximate_kernel(tokens: int, head_dim: int, dtype: torch.dtype, device_index: int, scale: float):
    del tokens, head_dim, dtype, device_index
    return _approximate_kernel(scale)


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
    use_first_order: bool = True,
    debug_finite: bool = False,
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
    if debug_finite:
        finite_stats = {
            "q_centroids": torch.isfinite(q_centroids).all().item(),
            "k_centroids": torch.isfinite(k_centroids_float).all().item(),
            "v_means": torch.isfinite(v_means).all().item(),
            "route_scores": torch.isfinite(route_scores).all().item(),
        }
        failed_stats = [stage for stage, finite in finite_stats.items() if not finite]
        if failed_stats:
            raise ValueError(f"generic PISA non-finite stages: {','.join(failed_stats)}")
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
    exact_attention = _compiled_flex_kernels(tokens, head_dim, q.dtype, q.get_device(), scale_value)
    approximate_attention = _compiled_approximate_kernel(tokens, head_dim, q.dtype, q.get_device(), scale_value)
    exact_output, exact_lse = exact_attention(q, k, v, exact_mask)
    approximate_output, approximate_lse = approximate_attention(
        q,
        k_centroids_float.to(q.dtype),
        v_means,
        selected,
        lengths.log(),
    )
    if debug_finite:
        finite_stages = {
            "exact_output": torch.isfinite(exact_output).all().item(),
            "exact_lse": torch.isfinite(exact_lse).all().item(),
            "approximate_output": torch.isfinite(approximate_output).all().item(),
            "approximate_lse": torch.isfinite(approximate_lse).all().item(),
        }
        failed_stages = [stage for stage, finite in finite_stages.items() if not finite]
        if failed_stages:
            raise ValueError(f"generic PISA non-finite stages: {','.join(failed_stages)}")

    total_lse = torch.logaddexp(exact_lse, approximate_lse)
    exact_weight = torch.exp(exact_lse - total_lse)
    approximate_weight = torch.exp(approximate_lse - total_lse)
    output = exact_output.float() * exact_weight[..., None]
    output.add_(approximate_output.float() * approximate_weight[..., None])
    if use_first_order:
        correction = torch.bmm(q, h_sum.to(q.dtype), out_dtype=torch.float32)
        correction.mul_(scale_value / tokens)
        output.add_(correction * approximate_weight[..., None])
    output = output.to(q.dtype)
    reported_backend = backend if use_first_order else f"{backend.removesuffix('_hyd')}_0th"
    return (output, reported_backend) if return_backend else output


def make_generic_pisa_override(
    *,
    exact_budget: float,
    device_index: int,
    previous_override: Callable | None,
    runtime_state,
    validate_output: bool = False,
) -> Callable:
    def fallback(original_func, *args, **kwargs):
        if previous_override is not None:
            return previous_override(original_func, *args, **kwargs)
        return original_func(*args, **kwargs)

    def attention_override(original_func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        transformer_options = kwargs.get("transformer_options") or {}
        block = transformer_options.get("block")
        block_index = transformer_options.get("block_index", 0)
        context = f"{block[0]}:{block[1]}:{block_index}" if isinstance(block, (tuple, list)) and len(block) == 2 else None
        reason = None
        if kwargs.get("is_self_attention") is not True:
            reason = "not_explicit_self_attention"
        elif kwargs.get("is_kv_cached_attention", False):
            reason = "kv_cached_attention_is_not_supported"
        elif mask is not None:
            reason = "attention_mask_is_not_supported"
        elif kwargs.get("enable_gqa", False):
            reason = "gqa_is_not_supported"
        elif not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
            reason = "qkv_are_not_tensors"
        elif q.device.type != "cuda" or q.device.index != device_index:
            reason = "validated_gfx1151_device_is_required"
        elif k.device != q.device or v.device != q.device:
            reason = "qkv_must_share_one_device"
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
            runtime_state.record(is_self_attention=kwargs.get("is_self_attention"), shape=shape, fallback_reason=reason, context=context)
            return fallback(original_func, q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

        grid_sizes = transformer_options.get("grid_sizes")
        video_layout = (
            isinstance(grid_sizes, (tuple, list))
            and len(grid_sizes) == 3
            and all(isinstance(value, int) and value > 0 for value in grid_sizes)
            and math.prod(grid_sizes) == tokens
            and grid_sizes[1] == grid_sizes[2]
            and grid_sizes[1] % SPATIAL_BLOCK_EDGE == 0
        )
        video_grid = tuple(grid_sizes) if video_layout else None
        side = math.isqrt(tokens)
        spatial_layout = side * side == tokens and side % SPATIAL_BLOCK_EDGE == 0
        if spatial_layout and isinstance(block, (tuple, list)) and block[0] == "input":
            reason = "spatial_input_stage_uses_dense_attention"
            runtime_state.record(is_self_attention=True, shape=(batch, heads, tokens, head_dim), fallback_reason=reason, context=context)
            return fallback(
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
        if video_grid is not None:
            q_bhtd, k_bhtd, v_bhtd = (_pack_video_blocks(tensor, video_grid) for tensor in (q_bhtd, k_bhtd, v_bhtd))
        elif spatial_layout:
            q_bhtd, k_bhtd, v_bhtd = (_pack_spatial_blocks(tensor, side) for tensor in (q_bhtd, k_bhtd, v_bhtd))
        q_flat, k_flat, v_flat = (tensor.contiguous().flatten(0, 1) for tensor in (q_bhtd, k_bhtd, v_bhtd))
        use_first_order = head_dim == 128 and q.dtype == torch.bfloat16
        budget = exact_budget if use_first_order else max(0.25, exact_budget)
        try:
            output, backend = generic_pisa_attention(
                q_flat,
                k_flat,
                v_flat,
                exact_budget=budget,
                scale=kwargs.get("scale"),
                return_backend=True,
                use_first_order=use_first_order,
                debug_finite=validate_output,
            )
            profile = (backend, batch, heads, tokens, head_dim, str(q.dtype))
            if not runtime_state.is_profile_validated(profile):
                if not torch.isfinite(output).all().item():
                    raise ValueError("generic PISA produced non-finite output")
                runtime_state.mark_profile_validated(profile)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            reason = f"pisa_backend_error_{type(exc).__name__}"
            runtime_state.record(
                is_self_attention=True,
                shape=(batch, heads, tokens, head_dim),
                fallback_reason=reason,
                context=context,
                error=exc if validate_output else None,
            )
            return fallback(
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
        quality = None
        if validate_output and runtime_state.quality_sample is None:
            reference = F.scaled_dot_product_attention(q_flat[:, None], k_flat[:, None], v_flat[:, None], scale=kwargs.get("scale")).squeeze(1)
            output_float = output.float()
            reference_float = reference.float()
            cosine = F.cosine_similarity(output_float.flatten(), reference_float.flatten(), dim=0).item()
            mae = (output_float - reference_float).abs().mean().item()
            quality = f"cos={cosine:.6f},mae={mae:.6g},pisa_max={output_float.abs().max().item():.6g},sdpa_max={reference_float.abs().max().item():.6g}"
        runtime_state.record(layer=-1, is_self_attention=True, shape=(batch, heads, tokens, head_dim), backend=backend, context=context, quality=quality)
        output = output.unflatten(0, (batch, heads))
        if video_grid is not None:
            output = _unpack_video_blocks(output, video_grid)
        elif spatial_layout:
            output = _unpack_spatial_blocks(output, side)
        if skip_output_reshape:
            return output
        return output.transpose(1, 2).contiguous().flatten(2)

    return attention_override
