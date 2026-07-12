from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdna35_block_attention.dispatch import fixed_block_attention
from rdna35_block_attention.comfy_patch import patch_model_attention


class DummyModel:
    def __init__(self):
        self.model_options = {"transformer_options": {}}

    def clone(self):
        cloned = DummyModel()
        cloned.model_options = {"transformer_options": self.model_options["transformer_options"].copy()}
        return cloned


class DispatchFallbackTests(unittest.TestCase):
    def test_cpu_fallback(self):
        q = torch.randn(1, 1, 64, 32, dtype=torch.float16)
        out, info = fixed_block_attention(q, q, q, layout="bhtd", return_diagnostics=True)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(info["backend"], "reference")
        self.assertIn("not_cuda", info["fallback_reason"])

    def test_unsupported_head_dim_fallback(self):
        q = torch.randn(1, 1, 64, 160, dtype=torch.float16)
        _, info = fixed_block_attention(q, q, q, layout="bhtd", return_diagnostics=True)
        self.assertIn("unsupported_head_dim_160", info["fallback_reason"])

    def test_block_size_fallback(self):
        q = torch.randn(1, 1, 64, 32, dtype=torch.float16)
        _, info = fixed_block_attention(q, q, q, block_size=32, layout="bhtd", return_diagnostics=True)
        self.assertIn("unsupported_block_size_32", info["fallback_reason"])

    def test_requires_grad_fallback(self):
        q = torch.randn(1, 1, 64, 32, dtype=torch.float16, requires_grad=True)
        _, info = fixed_block_attention(q, q, q, layout="bhtd", return_diagnostics=True)
        self.assertIn("requires_grad", info["fallback_reason"])

    def test_fp32_optimized_fallback(self):
        q = torch.randn(1, 1, 64, 32, dtype=torch.float32)
        _, info = fixed_block_attention(q, q, q, layout="bhtd", return_diagnostics=True)
        self.assertIn("unsupported_dtype", info["fallback_reason"])

    def test_arbitrary_mask_reference_fallback(self):
        q = torch.randn(1, 1, 64, 32, dtype=torch.float16)
        mask = torch.zeros(64, 64)
        _, info = fixed_block_attention(q, q, q, layout="bhtd", mask=mask, return_diagnostics=True)
        self.assertIn("arbitrary_mask", info["fallback_reason"])

    def test_model_patch_is_model_local_and_exact_only_is_safe(self):
        model = DummyModel()
        patched, info = patch_model_attention(
            model,
            enabled=True,
            mode="auto",
            semantic_mode="exact_only",
            block_size=64,
            causal=False,
            verbose_fallbacks=False,
        )
        self.assertIsNot(patched, model)
        self.assertNotIn("optimized_attention_override", model.model_options["transformer_options"])
        self.assertIn("optimized_attention_override", patched.model_options["transformer_options"])
        self.assertIn("exact_only", info)

        override = patched.model_options["transformer_options"]["optimized_attention_override"]
        q = torch.randn(1, 64, 32)

        def original_func(*args, **kwargs):
            return "original"

        self.assertEqual(override(original_func, q, q, q, 1), "original")


if __name__ == "__main__":
    unittest.main()
