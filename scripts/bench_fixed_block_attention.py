from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from rdna35_block_attention.diagnostics import detect_runtime
from rdna35_block_attention.dispatch import fixed_block_attention
from rdna35_block_attention.reference import (
    fixed_block_attention_ref,
    fixed_block_attention_sdpa,
    full_attention_sdpa_bhtd,
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_ms(fn, device: torch.device, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fixed 64-token block-diagonal attention.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--mode", choices=("auto", "triton", "reference"), default="auto")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    runtime = detect_runtime()
    device = torch.device("cuda") if runtime["torch_cuda_is_available"] else torch.device("cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    q = torch.randn(args.batch, args.heads, args.tokens, args.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    ref = fixed_block_attention_ref(q, k, v, block_size=64, causal=args.causal, layout="bhtd")
    sdpa_block = fixed_block_attention_sdpa(q, k, v, block_size=64, causal=args.causal, layout="bhtd")
    sdpa_full = full_attention_sdpa_bhtd(q, k, v, causal=args.causal)
    out, info = fixed_block_attention(q, k, v, block_size=64, causal=args.causal, layout="bhtd", mode=args.mode, return_diagnostics=True)
    error = (out.float() - ref.float()).abs().max().item()
    sdpa_block_error = (sdpa_block.float() - ref.float()).abs().max().item()
    sdpa_full_semantic_delta = (sdpa_full.float() - ref.float()).abs().max().item()

    ref_ms = median_ms(lambda: fixed_block_attention_ref(q, k, v, block_size=64, causal=args.causal, layout="bhtd"), device, args.iterations, args.warmup)
    sdpa_block_ms = median_ms(lambda: fixed_block_attention_sdpa(q, k, v, block_size=64, causal=args.causal, layout="bhtd"), device, args.iterations, args.warmup)
    sdpa_full_ms = median_ms(lambda: full_attention_sdpa_bhtd(q, k, v, causal=args.causal), device, args.iterations, args.warmup)
    print("RDNA35 Fixed Block Attention benchmark")
    print(f"device={runtime['device']}")
    print(f"torch={runtime['torch_version']} hip={runtime['torch_version_hip']}")
    print(f"triton={runtime['triton_available']} ({runtime['triton_info']})")
    print(f"shape=B{args.batch} H{args.heads} T{args.tokens} D{args.head_dim} dtype={dtype} causal={args.causal}")
    print(f"reference_median_ms={ref_ms:.3f}")
    print(f"pytorch_sdpa_block_mask_median_ms={sdpa_block_ms:.3f}")
    print(f"pytorch_sdpa_full_attention_median_ms={sdpa_full_ms:.3f}")
    print(f"dispatch_backend={info.get('backend')}")
    print(f"fallback_reason={info.get('fallback_reason')}")
    print(f"max_abs_error={error:.6g}")
    print(f"sdpa_block_mask_max_abs_error_vs_reference={sdpa_block_error:.6g}")
    print(f"full_sdpa_semantic_delta_vs_fixed_block_reference={sdpa_full_semantic_delta:.6g}")

    if info.get("backend") == "triton":
        triton_ms = median_ms(lambda: fixed_block_attention(q, k, v, block_size=64, causal=args.causal, layout="bhtd", mode="triton"), device, args.iterations, args.warmup)
        print(f"triton_median_ms={triton_ms:.3f}")
        print(f"speedup_vs_reference={ref_ms / triton_ms:.3f}x")
        print(f"speedup_vs_pytorch_sdpa_block_mask={sdpa_block_ms / triton_ms:.3f}x")
        print(f"latency_ratio_vs_pytorch_full_sdpa_different_semantics={triton_ms / sdpa_full_ms:.3f}x")
    else:
        print("speedup=unavailable")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
