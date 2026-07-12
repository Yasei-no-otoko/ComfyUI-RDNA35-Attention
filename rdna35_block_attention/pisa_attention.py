from __future__ import annotations

import math
import os
from typing import Any

import torch


BLOCK_SIZE = 64
HEAD_DIM = 128
SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int) -> None:
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v)):
        raise TypeError("q, k, and v must be torch tensors.")
    if q.ndim != 3:
        raise ValueError(f"PISA expects contiguous [BH,T,D] tensors, got rank {q.ndim}.")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"PISA self-attention requires identical q/k/v shapes, got {q.shape}, {k.shape}, {v.shape}.")
    if q.shape[-1] != HEAD_DIM:
        raise ValueError(f"PISA currently requires D={HEAD_DIM}, got D={q.shape[-1]}.")
    if q.shape[1] == 0:
        raise ValueError("PISA requires at least one token.")
    if q.dtype not in SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"PISA requires matching fp16/bf16 q/k/v, got {q.dtype}, {k.dtype}, {v.dtype}.")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"PISA requires q/k/v on one device, got {q.device}, {k.device}, {v.device}.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("PISA expects contiguous q/k/v tensors.")
    if any(x.requires_grad for x in (q, k, v)):
        raise ValueError("PISA is a training-free forward prototype and does not accept tensors requiring gradients.")
    if block_size != BLOCK_SIZE:
        raise ValueError(f"PISA currently supports block_size={BLOCK_SIZE} only, got {block_size}.")


def _resolve_exact_blocks(
    total_blocks: int,
    *,
    exact_budget: float | None,
    sparsity: float | None,
    exact_blocks: int | None,
    sink_block: int | None,
) -> tuple[int, float]:
    controls = sum(x is not None for x in (exact_budget, sparsity, exact_blocks))
    if controls > 1:
        raise ValueError("Set only one of exact_budget, sparsity, or exact_blocks.")

    if exact_blocks is not None:
        if isinstance(exact_blocks, bool) or not isinstance(exact_blocks, int):
            raise TypeError("exact_blocks must be an integer.")
        if not 0 <= exact_blocks <= total_blocks:
            raise ValueError(f"exact_blocks must be in [0,{total_blocks}], got {exact_blocks}.")
        count = exact_blocks
    else:
        budget = 0.25 if exact_budget is None and sparsity is None else exact_budget
        if sparsity is not None:
            if not 0.0 <= float(sparsity) <= 1.0:
                raise ValueError(f"sparsity must be in [0,1], got {sparsity}.")
            budget = 1.0 - float(sparsity)
        if budget is None or not 0.0 <= float(budget) <= 1.0:
            raise ValueError(f"exact_budget must be in [0,1], got {budget}.")
        count = min(total_blocks, int(math.ceil(float(budget) * total_blocks)))

    if sink_block is not None:
        if sink_block not in (0, -1):
            raise ValueError("sink_block must be 0, -1, or None.")
        count = max(1, count)
    return count, count / total_blocks


def _reference_block_stats(k: torch.Tensor, v: torch.Tensor, block_size: int):
    batch_heads, tokens, head_dim = k.shape
    total_blocks = math.ceil(tokens / block_size)
    k_means = torch.empty((batch_heads, total_blocks, head_dim), device=k.device, dtype=torch.float32)
    v_sums = torch.empty_like(k_means)
    h_sum = torch.zeros((batch_heads, head_dim, head_dim), device=k.device, dtype=torch.float32)
    lengths: list[int] = []

    for block in range(total_blocks):
        start = block * block_size
        end = min(start + block_size, tokens)
        k_block = k[:, start:end].float()
        v_block = v[:, start:end].float()
        k_mean = k_block.mean(dim=1)
        k_means[:, block] = k_mean
        v_sums[:, block] = v_block.sum(dim=1)
        h_sum.add_(torch.bmm((k_block - k_mean[:, None]).transpose(1, 2), v_block))
        lengths.append(end - start)
    return k_means, v_sums, h_sum, lengths


def _select_blocks(
    q: torch.Tensor,
    k_means: torch.Tensor,
    *,
    exact_count: int,
    block_size: int,
    scale: float,
    sink_block: int | None,
) -> torch.Tensor:
    batch_heads, tokens, _ = q.shape
    total_blocks = k_means.shape[1]
    if exact_count == total_blocks:
        return torch.ones((batch_heads, total_blocks, total_blocks), device=q.device, dtype=torch.bool)
    if exact_count == 0:
        return torch.zeros((batch_heads, total_blocks, total_blocks), device=q.device, dtype=torch.bool)

    q_means = torch.stack(
        [q[:, start:min(start + block_size, tokens)].float().mean(dim=1) for start in range(0, tokens, block_size)],
        dim=1,
    )
    route_scores = torch.einsum("bid,bjd->bij", q_means, k_means) * scale
    if sink_block is not None:
        route_scores[..., sink_block] = torch.inf
    indices = torch.topk(route_scores, k=exact_count, dim=-1).indices
    selected = torch.zeros_like(route_scores, dtype=torch.bool)
    selected.scatter_(-1, indices, True)
    return selected


def _online_rescale(m: torch.Tensor, block_m: torch.Tensor):
    new_m = torch.maximum(m, block_m)
    safe_new_m = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
    old_scale = torch.where(torch.isfinite(m), torch.exp(m - safe_new_m), torch.zeros_like(m))
    return new_m, safe_new_m, old_scale


def _piecewise_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_means: torch.Tensor,
    v_sums: torch.Tensor,
    h_sum: torch.Tensor,
    lengths: list[int],
    selected: torch.Tensor,
    *,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    batch_heads, tokens, head_dim = q.shape
    total_blocks = len(lengths)
    output = torch.empty_like(q)

    for query_block in range(total_blocks):
        q_start = query_block * block_size
        q_end = min(q_start + block_size, tokens)
        q_block = q[:, q_start:q_end].float()
        rows = q_end - q_start
        m = torch.full((batch_heads, rows), -torch.inf, device=q.device, dtype=torch.float32)
        denominator = torch.zeros_like(m)
        tail_denominator = torch.zeros_like(m)
        numerator = torch.zeros((batch_heads, rows, head_dim), device=q.device, dtype=torch.float32)

        for key_block in range(total_blocks):
            k_start = key_block * block_size
            k_end = min(k_start + block_size, tokens)
            exact = selected[:, query_block, key_block]
            scores = torch.bmm(q_block, k[:, k_start:k_end].float().transpose(1, 2)) * scale
            masked_scores = torch.where(exact[:, None, None], scores, -torch.inf)
            new_m, safe_new_m, old_scale = _online_rescale(m, masked_scores.amax(dim=-1))
            weights = torch.where(exact[:, None, None], torch.exp(scores - safe_new_m[:, :, None]), 0.0)
            numerator = numerator * old_scale[:, :, None] + torch.bmm(weights, v[:, k_start:k_end].float())
            denominator = denominator * old_scale + weights.sum(dim=-1)
            tail_denominator = tail_denominator * old_scale
            m = new_m

        for key_block in range(total_blocks):
            approximate = ~selected[:, query_block, key_block]
            mean_score = torch.einsum("bmd,bd->bm", q_block, k_means[:, key_block]) * scale
            masked_score = torch.where(approximate[:, None], mean_score, -torch.inf)
            new_m, safe_new_m, old_scale = _online_rescale(m, masked_score)
            weight = torch.where(approximate[:, None], torch.exp(mean_score - safe_new_m), 0.0)
            numerator = numerator * old_scale[:, :, None] + weight[:, :, None] * v_sums[:, key_block, None]
            denominator = denominator * old_scale + weight * lengths[key_block]
            tail_denominator = tail_denominator * old_scale + weight * lengths[key_block]
            m = new_m

        correction = torch.bmm(q_block, h_sum) * (scale / tokens)
        numerator.add_(correction * tail_denominator[:, :, None])
        output[:, q_start:q_end] = (numerator / denominator[:, :, None]).to(q.dtype)

    return output


def _gfx_target(device: torch.device | None = None) -> str | None:
    for name in ("PYTORCH_ROCM_ARCH", "GPU_ARCHS", "AMDGPU_TARGETS", "ROCM_ARCH"):
        value = os.environ.get(name, "")
        if "gfx" in value:
            return value[value.index("gfx"):].split(";")[0].split(",")[0]
    try:
        props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    except Exception:
        return None
    for name in ("gcnArchName", "gfx_version", "name"):
        value = str(getattr(props, name, ""))
        if "gfx" in value:
            return value[value.index("gfx"):].split(":")[0].split()[0]
    return None


def _load_ck_backend(q: torch.Tensor):
    if os.name != "nt":
        return None, "ck_wheel_is_windows_only"
    if q.device.type != "cuda":
        return None, f"ck_requires_cuda_hip_device_not_{q.device.type}"
    if not getattr(torch.version, "hip", None):
        return None, "torch_version_hip_not_detected"
    target = _gfx_target(q.device)
    if target != "gfx1151":
        return None, f"ck_targets_gfx1151_not_{target or 'unknown'}"

    try:
        import rdna35_pisa_ck
    except (ImportError, OSError, RuntimeError) as exc:
        return None, f"ck_wheel_unavailable_{type(exc).__name__}: {exc}"

    try:
        info = rdna35_pisa_ck.build_info()
        capabilities = rdna35_pisa_ck.capabilities()
    except (AttributeError, RuntimeError) as exc:
        return None, f"ck_wheel_metadata_failed_{type(exc).__name__}: {exc}"

    if info.get("api") != 5:
        return None, f"ck_api_5_required_not_{info.get('api')}"
    if info.get("architecture") != "gfx1151" or capabilities.get("architecture") != "gfx1151":
        return None, "ck_wheel_architecture_mismatch"
    if info.get("torch_python_build_version") != torch.__version__:
        return None, f"ck_torch_build_mismatch_{info.get('torch_python_build_version')}_vs_{torch.__version__}"
    if info.get("hip_python_build_version") != torch.version.hip:
        return None, f"ck_hip_build_mismatch_{info.get('hip_python_build_version')}_vs_{torch.version.hip}"
    if q.dtype not in capabilities.get("dtypes", ()):
        return None, f"ck_dtype_{q.dtype}_is_not_supported"
    max_tokens = int(capabilities.get("block_size", 0)) * int(capabilities.get("max_blocks", 0))
    if q.shape[-2] > max_tokens:
        return None, f"ck_tokens_{q.shape[-2]}_exceed_{max_tokens}"
    return rdna35_pisa_ck, None


def _triton_reject_reason(q: torch.Tensor) -> str | None:
    if os.name != "nt":
        return "staged_triton_prototype_is_windows_only"
    if q.device.type != "cuda":
        return f"staged_triton_requires_cuda_hip_device_not_{q.device.type}"
    if not getattr(torch.version, "hip", None):
        return "torch_version_hip_not_detected"
    target = _gfx_target(q.device)
    if target != "gfx1151":
        return f"staged_triton_targets_gfx1151_not_{target or 'unknown'}"
    try:
        import triton  # noqa: F401
    except Exception as exc:
        return f"triton_unavailable_{type(exc).__name__}: {exc}"
    return None


def pisa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    exact_budget: float | None = None,
    sparsity: float | None = None,
    exact_blocks: int | None = None,
    block_size: int = BLOCK_SIZE,
    scale: float | None = None,
    sink_block: int | None = None,
    backend: str = "auto",
    strict_backend: bool = False,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Training-free PISA forward attention for contiguous noncausal [BH,T,128] tensors.

    The reference follows equations 3-10 of arXiv:2602.01077 and the paper variant
    in xie-lab-ml/piecewise-sparse-attention. No full token-level QK matrix is formed.
    """
    _validate_qkv(q, k, v, block_size)
    if backend not in {"auto", "ck", "reference", "triton"}:
        raise ValueError("backend must be one of: auto, ck, reference, triton.")
    if not isinstance(strict_backend, bool):
        raise TypeError("strict_backend must be bool.")

    total_blocks = math.ceil(q.shape[1] / block_size)
    exact_count, resolved_budget = _resolve_exact_blocks(
        total_blocks,
        exact_budget=exact_budget,
        sparsity=sparsity,
        exact_blocks=exact_blocks,
        sink_block=sink_block,
    )
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale_value}.")

    used_backend = "reference"
    fallback_reason = None
    output = None
    if backend in {"auto", "ck"}:
        ck_backend, ck_reject_reason = _load_ck_backend(q)
        if ck_backend is not None:
            output = ck_backend.forward(q, k, v, exact_count, scale=scale_value, sink_block=sink_block)
            used_backend = "ck_flex"
        else:
            fallback_reason = ck_reject_reason
            if backend == "ck" and strict_backend:
                raise RuntimeError(f"PISA CK backend unavailable: {ck_reject_reason}")

    if output is None and backend in {"auto", "triton"}:
        triton_reject_reason = _triton_reject_reason(q)
        if triton_reject_reason is None:
            try:
                from .pisa_kernel import pisa_prepare_triton

                k_means, v_sums, h_sum, lengths = pisa_prepare_triton(k, v, block_size=block_size)
                used_backend = "triton_staged"
            except Exception as exc:
                triton_reject_reason = f"staged_triton_failed_{type(exc).__name__}: {exc}"
        if triton_reject_reason is not None:
            fallback_reason = "; ".join(x for x in (fallback_reason, triton_reject_reason) if x)
            if backend == "triton" and strict_backend:
                raise RuntimeError(f"PISA Triton backend unavailable: {triton_reject_reason}")

    if output is None:
        if used_backend == "reference":
            k_means, v_sums, h_sum, lengths = _reference_block_stats(k, v, block_size)

        selected = _select_blocks(
            q,
            k_means,
            exact_count=exact_count,
            block_size=block_size,
            scale=scale_value,
            sink_block=sink_block,
        )
        output = _piecewise_forward(
            q,
            k,
            v,
            k_means,
            v_sums,
            h_sum,
            lengths,
            selected,
            block_size=block_size,
            scale=scale_value,
        )
    diagnostics: dict[str, Any] = {
        "backend": used_backend,
        "requested_backend": backend,
        "optimized": used_backend in {"ck_flex", "triton_staged"},
        "fallback_reason": fallback_reason,
        "shape": tuple(q.shape),
        "dtype": str(q.dtype),
        "block_size": block_size,
        "total_blocks": total_blocks,
        "exact_blocks_per_query": exact_count,
        "approximate_blocks_per_query": total_blocks - exact_count,
        "exact_budget": resolved_budget,
        "sparsity": 1.0 - resolved_budget,
        "scale": scale_value,
        "selection": "query_centroid_key_centroid_topk",
        "tail_approximation": "block_centered_zeroth_plus_global_first_order",
        "exact_online_softmax": True,
        "shared_numerator_denominator_normalization": True,
        "materialized_token_qk": False,
        "routing_score_elements": q.shape[0] * total_blocks * total_blocks,
        "gfx_target": _gfx_target(q.device) if q.device.type == "cuda" else None,
    }
    if return_diagnostics:
        return output, diagnostics
    return output


def pisa_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs):
    kwargs["backend"] = "reference"
    return pisa_attention(q, k, v, **kwargs)
