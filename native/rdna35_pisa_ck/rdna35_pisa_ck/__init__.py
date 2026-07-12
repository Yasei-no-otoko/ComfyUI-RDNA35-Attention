from __future__ import annotations

import functools
import math
import struct

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention

__version__ = "0.7.0"
_EXPECTED_TORCH_VERSION = "2.14.0a0+rocm7.15.0a20260704"
_EXPECTED_HIP_VERSION = "7.15.26263"
_EXPECTED_NATIVE_API = 5
_EXPECTED_CXX_TORCH_VERSION = "2.14.0"
_EXPECTED_HIP_BUILD_VERSION = (7, 15, 26263)
_EXPECTED_CK_COMMIT = "4975bd0c8e17a54bdc27c746527a385e7383bb07"
if torch.__version__ != _EXPECTED_TORCH_VERSION or torch.version.hip != _EXPECTED_HIP_VERSION:
    raise RuntimeError(
        f"rdna35-pisa-ck {_EXPECTED_TORCH_VERSION}/{_EXPECTED_HIP_VERSION} is incompatible with "
        f"torch {torch.__version__}/ROCm {torch.version.hip}."
    )

from . import _C


BLOCK_SIZE = 64
HEAD_DIM = 128
MAX_BLOCKS = 144
MAX_TOKENS = BLOCK_SIZE * MAX_BLOCKS
SPATIAL_TOKENS = 9216
SPATIAL_SPARSE_EXACT_BLOCKS = 23
_SUPPORTED_DTYPES = {torch.bfloat16}
_BUILD_INFO = dict(_C.build_info())


def _check_extension_abi() -> None:
    expected = {
        "api": _EXPECTED_NATIVE_API,
        "architecture": "gfx1151",
        "ck_commit": _EXPECTED_CK_COMMIT,
        "torch_build_version": _EXPECTED_CXX_TORCH_VERSION,
        "hip_build_version": _EXPECTED_HIP_BUILD_VERSION,
    }
    mismatches = [f"{key}={_BUILD_INFO.get(key)!r}, expected {value!r}" for key, value in expected.items() if _BUILD_INFO.get(key) != value]
    if mismatches:
        raise RuntimeError("rdna35-pisa-ck native extension mismatch: " + "; ".join(mismatches))


_check_extension_abi()

_EXACT_KERNEL_OPTIONS = {
    "fwd_BLOCK_M": 64,
    "fwd_BLOCK_N": 64,
    "fwd_num_stages": 1,
    "fwd_num_warps": 4,
    "ROWS_GUARANTEED_SAFE": True,
}
_SPATIAL_EXACT_KERNEL_OPTIONS = {**_EXACT_KERNEL_OPTIONS, "fwd_BLOCK_N": 32}
_TAIL_KERNEL_OPTIONS = {
    "fwd_BLOCK_M": 64,
    "fwd_BLOCK_N": 32,
    "fwd_num_stages": 1,
    "fwd_num_warps": 4,
    "ROWS_GUARANTEED_SAFE": True,
}


def _check_runtime_abi() -> None:
    _check_extension_abi()
    build_torch = _EXPECTED_TORCH_VERSION
    build_hip = _EXPECTED_HIP_VERSION
    if build_torch != torch.__version__ or build_hip != torch.version.hip:
        raise RuntimeError(
            f"rdna35-pisa-ck was built for torch {build_torch}/ROCm {build_hip}, "
            f"running torch is {torch.__version__}/ROCm {torch.version.hip}."
        )


def _float32(value: float) -> float:
    try:
        converted = struct.unpack("f", struct.pack("f", float(value)))[0]
    except (OverflowError, struct.error, TypeError, ValueError) as exc:
        raise ValueError("scale must be representable as a positive finite float32 value.") from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError("scale must be representable as a positive finite float32 value.")
    return converted


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("q, k, and v must be torch tensors.")
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("PISA CK expects matching [BH,T,D] q/k/v tensors.")
    if q.shape[0] <= 0 or q.shape[1] <= 0 or q.shape[1] > MAX_TOKENS or q.shape[2] != HEAD_DIM:
        raise ValueError(f"PISA CK requires BH>0, 0<T<={MAX_TOKENS}, and D={HEAD_DIM}.")
    if q.device != k.device or q.device != v.device or q.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("PISA CK requires q/k/v on one PyTorch ROCm device.")
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("PISA CK requires matching bfloat16 q/k/v.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("PISA CK requires contiguous q/k/v.")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("PISA CK is forward-only.")


def _validate_spatial_bhtd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("q, k, and v must be torch tensors.")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("PISA spatial CK expects matching [B,H,T,D] q/k/v tensors.")
    batch, heads, tokens, head_dim = q.shape
    if batch <= 0 or heads <= 0 or tokens != SPATIAL_TOKENS or head_dim != HEAD_DIM:
        raise ValueError(f"PISA spatial CK requires B>0, H>0, T={SPATIAL_TOKENS}, and D={HEAD_DIM}.")
    if q.device != k.device or q.device != v.device or q.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("PISA spatial CK requires q/k/v on one PyTorch ROCm device.")
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("PISA spatial CK requires matching bfloat16 q/k/v.")
    expected_stride = (tokens * heads * head_dim, head_dim, heads * head_dim, 1)
    if q.stride() != expected_stride or k.stride() != expected_stride or v.stride() != expected_stride:
        raise ValueError("PISA spatial CK requires BHTD views of contiguous [B,T,H,D] storage.")
    if any(tensor.requires_grad for tensor in (q, k, v)):
        raise ValueError("PISA spatial CK is forward-only.")


def _flex_kernels(scale: float, spatial: bool):
    def exact_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: BlockMask):
        output, auxiliary = flex_attention(
            q[:, None],
            k[:, None],
            v[:, None],
            block_mask=block_mask,
            scale=scale,
            return_aux=AuxRequest(lse=True),
            kernel_options=_SPATIAL_EXACT_KERNEL_OPTIONS if spatial else _EXACT_KERNEL_OPTIONS,
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
            kernel_options=_TAIL_KERNEL_OPTIONS,
        )
        return output[:, 0], auxiliary.lse[:, 0]

    return exact_attention, approximate_attention


@functools.lru_cache(maxsize=16)
def _compiled_flex_kernels(tokens: int, dtype: torch.dtype, device_index: int, scale: float):
    del dtype, device_index
    exact_attention, approximate_attention = _flex_kernels(scale, tokens == SPATIAL_TOKENS)
    return (
        torch.compile(exact_attention, fullgraph=True, dynamic=False),
        torch.compile(approximate_attention, fullgraph=True, dynamic=False),
    )


def _block_lengths(tokens: int, device: torch.device) -> torch.Tensor:
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    values = [BLOCK_SIZE] * blocks
    values[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    return torch.tensor(values, device=device, dtype=torch.float32)


def _centered_h_sum(k: torch.Tensor, v: torch.Tensor, k_centroids: torch.Tensor) -> torch.Tensor:
    tokens = k.shape[1]
    blocks = k_centroids.shape[1]
    centered = torch.empty_like(k)
    if tokens == blocks * BLOCK_SIZE:
        torch.sub(
            k.unflatten(1, (blocks, BLOCK_SIZE)),
            k_centroids[:, :, None],
            out=centered.unflatten(1, (blocks, BLOCK_SIZE)),
        )
    else:
        block_means = torch.repeat_interleave(k_centroids, BLOCK_SIZE, dim=1)[:, :tokens]
        torch.sub(k, block_means, out=centered)
    return torch.bmm(centered.transpose(1, 2), v, out_dtype=torch.float32)


@functools.lru_cache(maxsize=8)
def _full_block_mask(device_index: int) -> BlockMask:
    device = torch.device("cuda", device_index)
    return BlockMask.from_kv_blocks(
        torch.ones((1, 1, 1), device=device, dtype=torch.int32),
        torch.zeros((1, 1, 1, 1), device=device, dtype=torch.int32),
        BLOCK_SIZE=1 << 30,
        seq_lengths=(1, 1),
    )


@torch.library.custom_op("rdna35_pisa_ck::block_stats", mutates_args=(), device_types="cuda")
def _block_stats_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _C.block_stats(q, k, v)


@_block_stats_op.register_fake
def _block_stats_op_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del k, v
    blocks = (q.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    shape = (q.shape[0], blocks, q.shape[2])
    return q.new_empty(shape), q.new_empty(shape, dtype=torch.float32), q.new_empty(shape)


def _forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_blocks: int,
    scale: float,
    sink_block: int,
    spatial_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    tokens = q.shape[1]
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if exact_blocks < 0 or exact_blocks > blocks:
        raise ValueError(f"exact_blocks must be in [0,{blocks}].")
    if sink_block not in (-2, -1, 0):
        raise ValueError("sink_block must be None, -1, or 0.")
    if sink_block != -2 and exact_blocks == 0:
        raise ValueError("sink_block requires at least one exact block.")

    if exact_blocks == blocks:
        return F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], scale=scale).squeeze(1)

    q_centroids, k_centroids_float, v_means = _block_stats_op(q, k, v)
    k_centroids = k_centroids_float.to(q.dtype)
    lengths = _block_lengths(tokens, q.device)
    h_sum = _centered_h_sum(k, v, k_centroids_float)

    if exact_blocks > 0:
        route_scores = torch.bmm(q_centroids, k_centroids.transpose(1, 2), out_dtype=torch.float32)
        route_scores.mul_(scale)
        if sink_block != -2:
            route_scores[..., sink_block] = torch.inf
        indices = torch.topk(route_scores, exact_blocks, dim=-1).indices.sort(dim=-1).values
    else:
        indices = torch.empty((q.shape[0], blocks, 0), device=q.device, dtype=torch.int64)

    selected = torch.zeros((q.shape[0], blocks, blocks), device=q.device, dtype=torch.bool)
    selected.scatter_(2, indices, True)
    log_lengths = lengths.log()
    exact_attention, approximate_attention = _compiled_flex_kernels(tokens, q.dtype, q.get_device(), scale)
    approximate_output, approximate_lse = approximate_attention(
        q,
        k_centroids,
        v_means,
        selected,
        log_lengths,
        _full_block_mask(q.get_device()),
    )

    correction = torch.bmm(q, h_sum.to(q.dtype), out_dtype=torch.float32)
    correction.mul_(scale / tokens)
    if exact_blocks == 0:
        return (approximate_output.float() + correction).to(q.dtype)

    padded_indices = F.pad(indices[:, None].to(torch.int32), (0, blocks - exact_blocks))
    exact_counts = torch.full((q.shape[0], 1, blocks), exact_blocks, device=q.device, dtype=torch.int32)
    empty_counts = torch.zeros_like(exact_counts)
    block_mask = BlockMask.from_kv_blocks(
        empty_counts,
        padded_indices,
        exact_counts,
        padded_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        seq_lengths=(tokens, tokens),
        compute_q_blocks=False,
    )
    exact_output, exact_lse = exact_attention(q, k, v, block_mask)
    if spatial_shape is not None and exact_blocks == SPATIAL_SPARSE_EXACT_BLOCKS:
        batch, heads = spatial_shape
        return _C.fuse_spatial_epilogue(
            exact_output,
            exact_lse,
            approximate_output,
            approximate_lse,
            correction,
            batch,
            heads,
        )
    total_lse = torch.logaddexp(exact_lse, approximate_lse)
    exact_weight = torch.exp(exact_lse - total_lse)
    approximate_weight = torch.exp(approximate_lse - total_lse)
    output = exact_output.float() * exact_weight[..., None]
    output.add_(approximate_output.float() * approximate_weight[..., None])
    output.add_(correction * approximate_weight[..., None])
    return output.to(q.dtype)


@torch.library.custom_op("rdna35_pisa_ck::forward", mutates_args=(), device_types="cuda")
def _forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_blocks: int,
    scale: float,
    sink_block: int,
) -> torch.Tensor:
    return _forward_impl(q, k, v, exact_blocks, scale, sink_block)


@_forward_op.register_fake
def _forward_op_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_blocks: int,
    scale: float,
    sink_block: int,
) -> torch.Tensor:
    del k, v, exact_blocks, scale, sink_block
    return torch.empty_like(q)


@torch.library.custom_op("rdna35_pisa_ck::forward_spatial_bhtd", mutates_args=(), device_types="cuda")
def _forward_spatial_bhtd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_blocks: int,
    scale: float,
    sink_block: int,
) -> torch.Tensor:
    packed_q, packed_k, packed_v = _C.pack_spatial_qkv(q, k, v)
    output = _forward_impl(
        packed_q,
        packed_k,
        packed_v,
        exact_blocks,
        scale,
        sink_block,
        spatial_shape=(q.shape[0], q.shape[1]),
    )
    if exact_blocks == SPATIAL_SPARSE_EXACT_BLOCKS:
        return output
    return _C.unpack_spatial_output(output, q.shape[0], q.shape[1])


@_forward_spatial_bhtd_op.register_fake
def _forward_spatial_bhtd_op_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_blocks: int,
    scale: float,
    sink_block: int,
) -> torch.Tensor:
    del k, v, exact_blocks, scale, sink_block
    return q.new_empty((q.shape[0], q.shape[2], q.shape[1] * q.shape[3]))


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_blocks: int,
    scale: float | None = None,
    sink_block: int | None = None,
) -> torch.Tensor:
    _check_runtime_abi()
    _validate_qkv(q, k, v)
    if isinstance(exact_blocks, bool) or not isinstance(exact_blocks, int):
        raise TypeError("exact_blocks must be an integer.")
    blocks = (q.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    if exact_blocks < 0 or exact_blocks > blocks:
        raise ValueError(f"exact_blocks must be in [0,{blocks}].")
    if sink_block not in (None, -1, 0):
        raise ValueError("sink_block must be None, -1, or 0.")
    if sink_block is not None and exact_blocks == 0:
        raise ValueError("sink_block requires at least one exact block.")
    scale_value = _float32(1.0 / math.sqrt(HEAD_DIM) if scale is None else scale)
    if not torch.compiler.is_compiling():
        _full_block_mask(q.get_device())
    return _forward_op(q, k, v, exact_blocks, scale_value, -2 if sink_block is None else sink_block)


def forward_spatial_bhtd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_blocks: int,
    scale: float | None = None,
    sink_block: int | None = None,
) -> torch.Tensor:
    _check_runtime_abi()
    _validate_spatial_bhtd(q, k, v)
    if isinstance(exact_blocks, bool) or not isinstance(exact_blocks, int):
        raise TypeError("exact_blocks must be an integer.")
    blocks = SPATIAL_TOKENS // BLOCK_SIZE
    if exact_blocks < 0 or exact_blocks > blocks:
        raise ValueError(f"exact_blocks must be in [0,{blocks}].")
    if exact_blocks not in (SPATIAL_SPARSE_EXACT_BLOCKS, blocks):
        raise ValueError(
            f"spatial PISA supports exact_blocks={SPATIAL_SPARSE_EXACT_BLOCKS}; "
            f"use exact_blocks={blocks} only for dense SDPA validation."
        )
    if sink_block not in (None, -1, 0):
        raise ValueError("sink_block must be None, -1, or 0.")
    if sink_block is not None and exact_blocks == 0:
        raise ValueError("sink_block requires at least one exact block.")
    scale_value = _float32(1.0 / math.sqrt(HEAD_DIM) if scale is None else scale)
    if not torch.compiler.is_compiling():
        _full_block_mask(q.get_device())
    return _forward_spatial_bhtd_op(q, k, v, exact_blocks, scale_value, -2 if sink_block is None else sink_block)


def prepare(device_index: int | None = None) -> None:
    _check_runtime_abi()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("rdna35-pisa-ck requires a PyTorch ROCm device.")
    index = torch.cuda.current_device() if device_index is None else int(device_index)
    _full_block_mask(index)


def build_info() -> dict[str, object]:
    return {
        **_BUILD_INFO,
        "torch_python_build_version": _EXPECTED_TORCH_VERSION,
        "hip_python_build_version": _EXPECTED_HIP_VERSION,
        "package_version": __version__,
        "torch_runtime_version": torch.__version__,
        "hip_runtime_version": torch.version.hip,
    }


def capabilities() -> dict[str, object]:
    return {
        "architecture": "gfx1151",
        "block_size": BLOCK_SIZE,
        "head_dim": HEAD_DIM,
        "max_blocks": MAX_BLOCKS,
        "dtypes": (torch.bfloat16,),
        "causal": False,
        "backward": False,
        "first_order_tail": True,
        "flex_attention": True,
        "outer_torch_compile": True,
        "outer_torch_compile_requires_prepare": True,
        "spatial_bhtd": True,
        "spatial_tokens": SPATIAL_TOKENS,
        "spatial_sparse_exact_blocks": (SPATIAL_SPARSE_EXACT_BLOCKS,),
    }


def clear_compile_cache() -> None:
    _compiled_flex_kernels.cache_clear()
    _full_block_mask.cache_clear()


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "MAX_BLOCKS",
    "SPATIAL_SPARSE_EXACT_BLOCKS",
    "SPATIAL_TOKENS",
    "build_info",
    "capabilities",
    "clear_compile_cache",
    "forward",
    "forward_spatial_bhtd",
    "prepare",
]
