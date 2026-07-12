from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class LayoutInfo:
    layout: str
    batch: int | None
    heads: int | None
    tokens: int
    head_dim: int
    restore: Callable[[torch.Tensor], torch.Tensor]


def _canonical_layout(layout: str) -> str:
    value = layout.lower().replace("-", "_").replace(" ", "")
    aliases = {
        "auto": "auto",
        "bhtd": "bhtd",
        "b,h,t,d": "bhtd",
        "bh_t_d": "bh_t_d",
        "bhtd_flat": "bh_t_d",
        "bhtdflat": "bh_t_d",
        "bthd": "bthd",
        "b,t,h*d": "bthd",
        "b_t_hd": "bthd",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported layout '{layout}'.")
    return aliases[value]


def normalize_qkv_to_bh_t_d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layout: str = "auto",
    heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LayoutInfo]:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Fixed block self-attention requires q/k/v to have identical shapes, got {q.shape}, {k.shape}, {v.shape}.")

    selected = _canonical_layout(layout)
    if selected == "auto":
        if q.ndim == 4:
            selected = "bhtd"
        elif q.ndim == 3:
            selected = "bh_t_d"
        else:
            raise ValueError(f"Cannot infer layout for rank-{q.ndim} tensor.")

    if selected == "bhtd":
        if q.ndim != 4:
            raise ValueError(f"layout='bhtd' expects [B,H,T,D], got rank {q.ndim}.")
        batch, n_heads, tokens, head_dim = q.shape

        def restore(out: torch.Tensor) -> torch.Tensor:
            return out.reshape(batch, n_heads, tokens, head_dim)

        return (
            q.contiguous().reshape(batch * n_heads, tokens, head_dim),
            k.contiguous().reshape(batch * n_heads, tokens, head_dim),
            v.contiguous().reshape(batch * n_heads, tokens, head_dim),
            LayoutInfo(selected, batch, n_heads, tokens, head_dim, restore),
        )

    if selected == "bh_t_d":
        if q.ndim != 3:
            raise ValueError(f"layout='bh_t_d' expects [BH,T,D], got rank {q.ndim}.")
        batch_heads, tokens, head_dim = q.shape

        def restore(out: torch.Tensor) -> torch.Tensor:
            return out.reshape(batch_heads, tokens, head_dim)

        return (
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            LayoutInfo(selected, None, None, tokens, head_dim, restore),
        )

    if selected == "bthd":
        if q.ndim != 3:
            raise ValueError(f"layout='bthd' expects [B,T,H*D], got rank {q.ndim}.")
        if heads is None or heads <= 0:
            raise ValueError("layout='bthd' requires a positive heads argument.")
        batch, tokens, channels = q.shape
        if channels % heads != 0:
            raise ValueError(f"channels={channels} is not divisible by heads={heads}.")
        head_dim = channels // heads

        def flatten(x: torch.Tensor) -> torch.Tensor:
            return x.contiguous().reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3).contiguous().reshape(batch * heads, tokens, head_dim)

        def restore(out: torch.Tensor) -> torch.Tensor:
            return out.reshape(batch, heads, tokens, head_dim).permute(0, 2, 1, 3).contiguous().reshape(batch, tokens, channels)

        return (
            flatten(q),
            flatten(k),
            flatten(v),
            LayoutInfo(selected, batch, heads, tokens, head_dim, restore),
        )

    raise AssertionError(f"Unhandled layout {selected}")


def _mask_block(
    mask: torch.Tensor | None,
    start: int,
    end: int,
    *,
    batch_heads: int,
    info: LayoutInfo,
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None

    m = mask.to(device=device)
    if m.ndim == 2:
        if m.shape[-2:] != (info.tokens, info.tokens):
            raise ValueError(f"2D mask must have shape [T,T], got {tuple(m.shape)}.")
        return m[start:end, start:end].unsqueeze(0).expand(batch_heads, -1, -1)

    if m.ndim == 3:
        if m.shape[-2:] != (info.tokens, info.tokens):
            raise ValueError(f"3D mask must end with [T,T], got {tuple(m.shape)}.")
        m = m[:, start:end, start:end]
        if info.batch is not None and m.shape[0] == info.batch:
            return m.repeat_interleave(info.heads, dim=0)
        if m.shape[0] == 1:
            return m.expand(batch_heads, -1, -1)
        if m.shape[0] == batch_heads:
            return m
        raise ValueError(f"3D mask batch dimension {m.shape[0]} is not compatible with BH={batch_heads}.")

    if m.ndim == 4:
        if info.batch is None or info.heads is None:
            raise ValueError("4D mask requires a layout with known B and H.")
        if m.shape[-2:] != (info.tokens, info.tokens):
            raise ValueError(f"4D mask must end with [T,T], got {tuple(m.shape)}.")
        if m.shape[0] not in (1, info.batch):
            raise ValueError(f"4D mask batch dimension {m.shape[0]} is not compatible with B={info.batch}.")
        if m.shape[1] not in (1, info.heads):
            raise ValueError(f"4D mask head dimension {m.shape[1]} is not compatible with H={info.heads}.")
        m = m.expand(info.batch, info.heads, info.tokens, info.tokens)
        return m[:, :, start:end, start:end].reshape(batch_heads, end - start, end - start)

    raise ValueError(f"Unsupported mask rank {m.ndim}.")


def fixed_block_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float | None = None,
    causal: bool = False,
    layout: str = "auto",
    *,
    heads: int | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    q_flat, k_flat, v_flat, info = normalize_qkv_to_bh_t_d(q, k, v, layout=layout, heads=heads)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    batch_heads, tokens, head_dim = q_flat.shape
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q_flat)

    for start in range(0, tokens, block_size):
        end = min(start + block_size, tokens)
        q_block = q_flat[:, start:end, :].to(torch.float32)
        k_block = k_flat[:, start:end, :].to(torch.float32)
        v_block = v_flat[:, start:end, :].to(torch.float32)

        scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale_value
        if causal:
            causal_mask = torch.ones((end - start, end - start), dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0), -torch.inf)

        block_mask = _mask_block(mask, start, end, batch_heads=batch_heads, info=info, device=q.device)
        if block_mask is not None:
            if block_mask.dtype == torch.bool:
                scores = scores.masked_fill(~block_mask, -torch.inf)
            else:
                scores = scores + block_mask.to(dtype=scores.dtype)

        probs = torch.softmax(scores, dim=-1)
        out[:, start:end, :] = torch.matmul(probs, v_block).to(dtype=q_flat.dtype)

    return info.restore(out)


def fixed_block_attention_bthd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    block_size: int = 64,
    scale: float | None = None,
    causal: bool = False,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return fixed_block_attention_ref(
        q,
        k,
        v,
        block_size=block_size,
        scale=scale,
        causal=causal,
        layout="bthd",
        heads=heads,
        mask=mask,
    )


def block_diagonal_sdpa_mask(
    tokens: int,
    *,
    block_size: int = 64,
    causal: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    positions = torch.arange(tokens, device=device)
    block_ids = positions // block_size
    mask = block_ids[:, None] == block_ids[None, :]
    if causal:
        mask = mask & (positions[None, :] <= positions[:, None])
    return mask


def fixed_block_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float | None = None,
    causal: bool = False,
    layout: str = "auto",
    *,
    heads: int | None = None,
) -> torch.Tensor:
    q_flat, k_flat, v_flat, info = normalize_qkv_to_bh_t_d(q, k, v, layout=layout, heads=heads)
    attn_mask = block_diagonal_sdpa_mask(info.tokens, block_size=block_size, causal=causal, device=q_flat.device)
    q_sdpa = q_flat.unsqueeze(1)
    k_sdpa = k_flat.unsqueeze(1)
    v_sdpa = v_flat.unsqueeze(1)
    kwargs = {}
    if scale is not None:
        kwargs["scale"] = float(scale)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        **kwargs,
    ).squeeze(1)
    return info.restore(out.to(dtype=q_flat.dtype))


def full_attention_sdpa_bhtd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    kwargs = {}
    if scale is not None:
        kwargs["scale"] = float(scale)
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=causal,
        **kwargs,
    )
