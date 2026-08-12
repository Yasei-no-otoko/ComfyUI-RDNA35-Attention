from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdna35_block_attention.minimax_h3_gfx1151_attention import (
    MINIMAX_H3_MAX_PACK_TOKENS,
    MINIMAX_H3_MIN_PACK_TOKENS,
    pack_minimax_h3_qkv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiniMax-H3 native and packed QKV attention layouts on gfx1151.")
    parser.add_argument("--mode", choices=("raw", "optimized", "compare"), default="compare")
    parser.add_argument("--tokens", type=int, default=9170)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-aotriton", action="store_true")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def median_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if elapsed <= 0.0 or elapsed > 10000.0:
            raise RuntimeError(f"invalid HIP event duration: {elapsed} ms")
        samples.append(elapsed)
    return float(statistics.median(samples))


def alternating_medians_ms(first_fn, second_fn, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        first_fn()
        second_fn()
    torch.cuda.synchronize()
    first_samples = []
    second_samples = []
    for _ in range(iterations):
        for fn, samples in ((first_fn, first_samples), (second_fn, second_samples)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            if elapsed <= 0.0 or elapsed > 10000.0:
                raise RuntimeError(f"invalid HIP event duration: {elapsed} ms")
            samples.append(elapsed)
    return float(statistics.median(first_samples)), float(statistics.median(second_samples))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("PyTorch ROCm GPU is required")
    target = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":", 1)[0]
    if target != "gfx1151":
        raise RuntimeError(f"gfx1151 is required, got {target or 'unknown'}")
    backend = torch.backends.cuda.preferred_rocm_fa_library()
    if args.require_aotriton and not str(backend).endswith(".AOTriton"):
        raise RuntimeError(f"AOTriton is required for this reproduction run, got {backend}")
    if not MINIMAX_H3_MIN_PACK_TOKENS <= args.tokens <= MINIMAX_H3_MAX_PACK_TOKENS:
        raise ValueError(f"tokens must be between {MINIMAX_H3_MIN_PACK_TOKENS} and {MINIMAX_H3_MAX_PACK_TOKENS}")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("iterations must be positive; warmup must be non-negative")

    torch.manual_seed(args.tokens)
    qkv = torch.randn(1, args.tokens, 3, 56, 128, device="cuda", dtype=torch.bfloat16)
    q = qkv[:, :, 0].permute(0, 2, 1, 3)
    k = qkv[:, :, 1].permute(0, 2, 1, 3)
    v = qkv[:, :, 2].clone().permute(0, 2, 1, 3)

    def raw_attention():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()

    def optimized_attention():
        packed_q, packed_k, packed_v, _ = pack_minimax_h3_qkv(q, k, v)
        return torch.nn.functional.scaled_dot_product_attention(packed_q, packed_k, packed_v).transpose(1, 2).contiguous()

    if args.check:
        expected = raw_attention()
        actual = optimized_attention()
        torch.cuda.synchronize()
        if not torch.isfinite(actual).all():
            raise RuntimeError("optimized attention produced non-finite output")
        if not torch.allclose(actual.float(), expected.float(), atol=args.atol, rtol=args.rtol):
            delta = (actual.float() - expected.float()).abs()
            raise RuntimeError(f"correctness failed: max_abs_error={delta.max().item():.6g}, mean_abs_error={delta.mean().item():.6g}")
        delta = (actual.float() - expected.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()
        print(f"correctness=pass max_abs_error={delta.max().item():.6g} cosine={cosine:.9f}")

    print(f"torch={torch.__version__} hip={torch.version.hip} backend={backend} target={target} shape=B1_H56_T{args.tokens}_D128 dtype=bfloat16")
    timings = {}
    if args.mode == "compare":
        timings["raw_ms"], timings["optimized_ms"] = alternating_medians_ms(
            raw_attention,
            optimized_attention,
            args.warmup,
            args.iterations,
        )
        print(f"raw_ms={timings['raw_ms']:.6f}")
        print(f"optimized_ms={timings['optimized_ms']:.6f}")
    elif args.mode == "raw":
        timings["raw_ms"] = median_ms(raw_attention, args.warmup, args.iterations)
        print(f"raw_ms={timings['raw_ms']:.6f}")
    else:
        timings["optimized_ms"] = median_ms(optimized_attention, args.warmup, args.iterations)
        print(f"optimized_ms={timings['optimized_ms']:.6f}")
    if args.mode == "compare":
        raw_ms = timings["raw_ms"]
        optimized_ms = timings["optimized_ms"]
        print(f"speedup={raw_ms / optimized_ms:.6f}x reduction_pct={(1.0 - optimized_ms / raw_ms) * 100.0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
