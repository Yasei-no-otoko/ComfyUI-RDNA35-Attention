from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdna35_block_attention.diagnostics import has_triton, is_rocm_pytorch
from rdna35_block_attention.dispatch import fixed_block_attention
from rdna35_block_attention.full_attention import full_attention_triton, validate_full_attention_triton_bh
from rdna35_block_attention.reference import fixed_block_attention_ref


def rocm_triton_available() -> bool:
    triton_ok, _ = has_triton(import_module=True)
    return bool(torch.cuda.is_available() and is_rocm_pytorch() and triton_ok)


@unittest.skipUnless(rocm_triton_available(), "ROCm/Triton not available")
class TritonForwardTests(unittest.TestCase):
    def test_triton_matches_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            for tokens in (64, 128, 256):
                for head_dim in (32, 64, 128):
                    with self.subTest(dtype=dtype, tokens=tokens, head_dim=head_dim):
                        torch.manual_seed(tokens + head_dim)
                        q = torch.randn(1, 2, tokens, head_dim, device="cuda", dtype=dtype)
                        k = torch.randn_like(q)
                        v = torch.randn_like(q)
                        ref = fixed_block_attention_ref(q, k, v, block_size=64, layout="bhtd")
                        out, info = fixed_block_attention(q, k, v, block_size=64, layout="bhtd", mode="triton", return_diagnostics=True)
                        torch.cuda.synchronize()
                        self.assertEqual(info["backend"], "triton", info)
                        atol = 5e-2 if dtype == torch.float16 else 1e-1
                        rtol = 5e-2 if dtype == torch.float16 else 1e-1
                        self.assertTrue(torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol))

    def test_full_attention_matches_sdpa(self):
        cases = ((17, 31, 64), (64, 64, 128), (65, 137, 128), (129, 33, 64))
        for dtype in (torch.float16, torch.bfloat16):
            for q_tokens, kv_tokens, head_dim in cases:
                with self.subTest(dtype=dtype, q_tokens=q_tokens, kv_tokens=kv_tokens, head_dim=head_dim):
                    torch.manual_seed(q_tokens + kv_tokens + head_dim)
                    q = torch.randn(2, q_tokens, head_dim, device="cuda", dtype=dtype)
                    k = torch.randn(2, kv_tokens, head_dim, device="cuda", dtype=dtype)
                    v = torch.randn_like(k)
                    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    out, info = full_attention_triton(q, k, v, return_diagnostics=True)
                    torch.cuda.synchronize()
                    self.assertTrue(info["supported"], info)
                    self.assertEqual(info["q_tokens"], q_tokens)
                    self.assertEqual(info["kv_tokens"], kv_tokens)
                    atol = 2e-2 if dtype == torch.float16 else 5e-2
                    rtol = 2e-2 if dtype == torch.float16 else 5e-2
                    self.assertTrue(torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol))

    def test_full_attention_custom_scale_matches_sdpa(self):
        q = torch.randn(1, 37, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 71, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        scale = 0.05
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        out = full_attention_triton(q, k, v, scale=scale)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))


class FullAttentionValidationTests(unittest.TestCase):
    def test_cpu_input_reports_fallback_reason_without_importing_triton(self):
        q = torch.randn(2, 5, 128, dtype=torch.float16)
        k = torch.randn(2, 7, 128, dtype=torch.float16)
        v = torch.randn_like(k)
        info = validate_full_attention_triton_bh(q, k, v)
        self.assertFalse(info["supported"])
        self.assertEqual(info["reason"], "not_cuda_or_hip_device_cpu")
        self.assertEqual(info["q_tokens"], 5)
        self.assertEqual(info["kv_tokens"], 7)

    def test_cross_attention_shape_validation(self):
        q = torch.randn(2, 5, 128, dtype=torch.float16)
        k = torch.randn(2, 7, 128, dtype=torch.float16)
        v = torch.randn(2, 8, 128, dtype=torch.float16)
        info = validate_full_attention_triton_bh(q, k, v)
        self.assertEqual(info["reason"], "key_value_length_mismatch")


if __name__ == "__main__":
    unittest.main()
