from __future__ import annotations

import statistics
import time
from typing import Any

import torch

from .rdna35_block_attention.comfy_patch import patch_model_attention
from .rdna35_block_attention.diagnostics import detect_runtime, explain_dispatch
from .rdna35_block_attention.dispatch import fixed_block_attention
from .rdna35_block_attention.full_attention import full_attention_triton
from .rdna35_block_attention.pisa_attention import pisa_attention
from .rdna35_block_attention.pisa_patch import patch_model_pisa_attention
from .rdna35_block_attention.pisa_runtime import PISA_RUNTIME_ATTACHMENT
from .rdna35_block_attention.reference import (
    fixed_block_attention_ref,
    fixed_block_attention_sdpa,
    full_attention_sdpa_bhtd,
)


class RDNA35BlockAttentionDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("diagnostics",)
    FUNCTION = "run"
    CATEGORY = "RDNA35/Fixed Block Attention"
    OUTPUT_NODE = True

    def run(self):
        return (explain_dispatch(),)


class RDNA35PatchModelAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "mode": (["auto", "reference", "triton"], {"default": "auto"}),
                "semantic_mode": (["exact_only", "experimental_force_block_local"], {"default": "exact_only"}),
                "block_size": ("INT", {"default": 64, "min": 64, "max": 64, "step": 64}),
                "causal": ("BOOLEAN", {"default": False}),
                "verbose_fallbacks": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "patch"
    CATEGORY = "RDNA35/Fixed Block Attention"
    EXPERIMENTAL = True

    def patch(self, model, enabled, mode, semantic_mode, block_size, causal, verbose_fallbacks):
        patched, info = patch_model_attention(
            model,
            enabled=enabled,
            mode=mode,
            semantic_mode=semantic_mode,
            block_size=block_size,
            causal=causal,
            verbose_fallbacks=verbose_fallbacks,
        )
        return patched, info


class RDNA35PatchAnimaPISAAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "verbose_fallbacks": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "patch"
    CATEGORY = "RDNA35/Attention Research"
    EXPERIMENTAL = True

    def patch(self, model, enabled, verbose_fallbacks):
        return patch_model_pisa_attention(
            model,
            enabled=enabled,
            exact_budget=0.15625,
            token_policy="anima_1536_spatial",
            start_layer=4,
            verbose_fallbacks=verbose_fallbacks,
        )


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(fn, device: torch.device, iterations: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    _sync_if_cuda(device)

    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize(device)
        return float(statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends)))

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(samples))


class RDNA35FixedBlockAttentionBenchmark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch": ("INT", {"default": 1, "min": 1, "max": 8}),
                "heads": ("INT", {"default": 4, "min": 1, "max": 32}),
                "tokens": ("INT", {"default": 128, "min": 1, "max": 8192, "step": 64}),
                "head_dim": ("INT", {"default": 64, "min": 32, "max": 128, "step": 32}),
                "dtype": (["float16", "bfloat16"], {"default": "float16"}),
                "mode": (["auto", "triton", "reference"], {"default": "auto"}),
                "causal": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("benchmark",)
    FUNCTION = "run"
    CATEGORY = "RDNA35/Fixed Block Attention"
    OUTPUT_NODE = True

    def run(self, batch, heads, tokens, head_dim, dtype, mode, causal):
        runtime = detect_runtime()
        device = torch.device("cuda") if runtime["torch_cuda_is_available"] else torch.device("cpu")
        compute_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        q = torch.randn(batch, heads, tokens, head_dim, device=device, dtype=compute_dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        ref = fixed_block_attention_ref(q, k, v, block_size=64, causal=causal, layout="bhtd")
        sdpa_block = fixed_block_attention_sdpa(q, k, v, block_size=64, causal=causal, layout="bhtd")
        sdpa_full = full_attention_sdpa_bhtd(q, k, v, causal=causal)
        out, info = fixed_block_attention(q, k, v, block_size=64, causal=causal, layout="bhtd", mode=mode, return_diagnostics=True)
        max_error = (out.float() - ref.float()).abs().max().item()
        sdpa_block_error = (sdpa_block.float() - ref.float()).abs().max().item()
        sdpa_full_semantic_delta = (sdpa_full.float() - ref.float()).abs().max().item()

        ref_ms = _median_ms(lambda: fixed_block_attention_ref(q, k, v, block_size=64, causal=causal, layout="bhtd"), device)
        sdpa_block_ms = _median_ms(lambda: fixed_block_attention_sdpa(q, k, v, block_size=64, causal=causal, layout="bhtd"), device)
        sdpa_full_ms = _median_ms(lambda: full_attention_sdpa_bhtd(q, k, v, causal=causal), device)
        lines = [
            "RDNA35 Fixed Block Attention Benchmark",
            f"device: {runtime['device']}",
            f"torch: {runtime['torch_version']} hip={runtime['torch_version_hip']}",
            f"triton: {runtime['triton_available']} ({runtime['triton_info']})",
            f"shape: B={batch} H={heads} T={tokens} D={head_dim} dtype={compute_dtype} causal={causal}",
            f"reference median: {ref_ms:.3f} ms",
            f"PyTorch SDPA block-mask median: {sdpa_block_ms:.3f} ms",
            f"PyTorch SDPA full-attention median: {sdpa_full_ms:.3f} ms",
            f"dispatch backend: {info.get('backend')}",
            f"fallback reason: {info.get('fallback_reason')}",
            f"max abs error vs reference: {max_error:.6g}",
            f"SDPA block-mask max abs error vs reference: {sdpa_block_error:.6g}",
            f"full SDPA semantic delta vs fixed-block reference: {sdpa_full_semantic_delta:.6g}",
        ]

        if info.get("backend") == "triton":
            triton_ms = _median_ms(lambda: fixed_block_attention(q, k, v, block_size=64, causal=causal, layout="bhtd", mode="triton"), device)
            speedup = ref_ms / triton_ms if triton_ms > 0 else float("inf")
            lines.append(f"triton median: {triton_ms:.3f} ms")
            lines.append(f"measured speedup vs reference: {speedup:.3f}x")
            lines.append(f"measured speedup vs PyTorch SDPA block-mask: {sdpa_block_ms / triton_ms:.3f}x")
            lines.append(f"latency ratio vs PyTorch full SDPA (different semantics): {triton_ms / sdpa_full_ms:.3f}x")
        else:
            lines.append("measured speedup: unavailable because optimized path did not run")

        return ("\n".join(lines),)


class RDNA35FullAttentionBenchmark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch": ("INT", {"default": 1, "min": 1, "max": 4}),
                "heads": ("INT", {"default": 16, "min": 1, "max": 40}),
                "query_tokens": ("INT", {"default": 4096, "min": 64, "max": 9216, "step": 64}),
                "key_tokens": ("INT", {"default": 4096, "min": 64, "max": 9216, "step": 64}),
                "head_dim": ([64, 128], {"default": 128}),
                "dtype": (["float16", "bfloat16"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("benchmark",)
    FUNCTION = "run"
    CATEGORY = "RDNA35/Attention Research"
    OUTPUT_NODE = True
    EXPERIMENTAL = True

    def run(self, batch, heads, query_tokens, key_tokens, head_dim, dtype):
        device = torch.device("cuda")
        compute_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        q = torch.randn(batch, heads, query_tokens, head_dim, device=device, dtype=compute_dtype)
        k = torch.randn(batch, heads, key_tokens, head_dim, device=device, dtype=compute_dtype)
        v = torch.randn_like(k)
        q_bh = q.flatten(0, 1).contiguous()
        k_bh = k.flatten(0, 1).contiguous()
        v_bh = v.flatten(0, 1).contiguous()
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        output, info = full_attention_triton(q_bh, k_bh, v_bh, return_diagnostics=True)
        output = output.unflatten(0, (batch, heads))
        sdpa_ms = _median_ms(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), device)
        triton_ms = _median_ms(lambda: full_attention_triton(q_bh, k_bh, v_bh), device)
        return ("\n".join([
            "RDNA35 Exact Full Attention Benchmark",
            f"shape: B={batch} H={heads} Q={query_tokens} K={key_tokens} D={head_dim} dtype={compute_dtype}",
            f"PyTorch SDPA median: {sdpa_ms:.3f} ms",
            f"gfx1151 Triton median: {triton_ms:.3f} ms",
            f"Triton/SDPA ratio: {triton_ms / sdpa_ms:.3f}x",
            f"max abs error: {(output.float() - reference.float()).abs().max().item():.6g}",
            f"kernel config: {info.get('config')}",
        ]),)


class RDNA35PISAAttentionBenchmark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_heads": ("INT", {"default": 1, "min": 1, "max": 40}),
                "tokens": ("INT", {"default": 9216, "min": 128, "max": 9216, "step": 64}),
                "exact_budget": ("FLOAT", {"default": 0.15625, "min": 0.015625, "max": 1.0, "step": 0.015625}),
                "dtype": (["bfloat16"], {"default": "bfloat16"}),
                "backend": (["ck", "auto", "triton", "reference"], {"default": "ck"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("benchmark",)
    FUNCTION = "run"
    CATEGORY = "RDNA35/Attention Research"
    OUTPUT_NODE = True
    EXPERIMENTAL = True

    def run(self, batch_heads, tokens, exact_budget, dtype, backend):
        device = torch.device("cuda")
        compute_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        q = torch.randn(batch_heads, tokens, 128, device=device, dtype=compute_dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        reference = torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
        _sync_if_cuda(device)
        start = time.perf_counter()
        output, info = pisa_attention(
            q,
            k,
            v,
            exact_budget=exact_budget,
            backend=backend,
            strict_backend=backend in {"ck", "triton"},
            return_diagnostics=True,
        )
        _sync_if_cuda(device)
        cold_ms = (time.perf_counter() - start) * 1000.0
        pisa_ms = _median_ms(
            lambda: pisa_attention(q, k, v, exact_budget=exact_budget, backend=backend, strict_backend=backend in {"ck", "triton"}),
            device,
        )
        sdpa_ms = _median_ms(lambda: torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)), device)
        cosine = torch.nn.functional.cosine_similarity(reference.float().flatten(), output.float().flatten(), dim=0).item()
        return ("\n".join([
            "RDNA35 PISA Attention Benchmark",
            f"shape: BH={batch_heads} T={tokens} D=128 dtype={compute_dtype}",
            f"backend: {info.get('backend')}",
            f"exact blocks: {info.get('exact_blocks_per_query')}/{info.get('total_blocks')}",
            f"PISA cold compile/call: {cold_ms:.3f} ms",
            f"PISA steady-state median: {pisa_ms:.3f} ms",
            f"dense SDPA steady-state median: {sdpa_ms:.3f} ms",
            f"PISA/dense latency ratio: {pisa_ms / sdpa_ms:.3f}x",
            f"cosine vs dense SDPA: {cosine:.9f}",
            f"mean abs error: {(output.float() - reference.float()).abs().mean().item():.6g}",
            f"fallback: {info.get('fallback_reason')}",
            (
                "CK Tile block statistics + PyTorch FlexAttention exact/tail + WMMA first-order correction."
                if info.get("backend") == "ck_flex"
                else "CK/Flex hybrid did not run; see backend and fallback above."
            ),
        ]),)


class RDNA35PISARuntimeReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "run"
    CATEGORY = "RDNA35/Attention Research"
    OUTPUT_NODE = True
    EXPERIMENTAL = True

    def run(self, model, latent):
        state = model.get_attachment(PISA_RUNTIME_ATTACHMENT)
        if state is None:
            raise RuntimeError("RDNA35 PISA runtime state is not attached to this MODEL")
        hits = sum(state.per_layer_hits.values())
        if hits == 0:
            raise RuntimeError("INVALID BENCHMARK: PISA backend was not executed")
        counts = set(state.per_layer_hits.values())
        if set(state.per_layer_hits) != set(range(4, 28)) or len(counts) != 1:
            raise RuntimeError(f"Incomplete PISA layer accounting: {state.report()}")
        forwards = counts.pop()
        state.verify(forwards)
        if not state.verified:
            raise RuntimeError(f"PISA runtime verification failed: {state.report()}")
        report = state.report()
        return {"ui": {"text": [report]}, "result": (latent, report)}


NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "RDNA35BlockAttentionDiagnostics": RDNA35BlockAttentionDiagnostics,
    "RDNA35PatchModelAttention": RDNA35PatchModelAttention,
    "RDNA35PatchAnimaPISAAttention": RDNA35PatchAnimaPISAAttention,
    "RDNA35FixedBlockAttentionBenchmark": RDNA35FixedBlockAttentionBenchmark,
    "RDNA35FullAttentionBenchmark": RDNA35FullAttentionBenchmark,
    "RDNA35PISAAttentionBenchmark": RDNA35PISAAttentionBenchmark,
    "RDNA35PISARuntimeReport": RDNA35PISARuntimeReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RDNA35BlockAttentionDiagnostics": "RDNA35 Block Attention Diagnostics",
    "RDNA35PatchModelAttention": "RDNA35 Patch Model Attention",
    "RDNA35PatchAnimaPISAAttention": "RDNA35 Patch Anima PISA Attention",
    "RDNA35FixedBlockAttentionBenchmark": "RDNA35 Fixed Block Attention Benchmark",
    "RDNA35FullAttentionBenchmark": "RDNA35 Exact Full Attention Benchmark",
    "RDNA35PISAAttentionBenchmark": "RDNA35 PISA Attention Benchmark",
    "RDNA35PISARuntimeReport": "RDNA35 PISA Runtime Report",
}
