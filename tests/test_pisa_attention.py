import pathlib
import sys
import unittest

import torch
import torch.nn.functional as F


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rdna35_block_attention.pisa_attention import _gfx_target, _reference_block_stats, pisa_attention
from rdna35_block_attention.pisa_kernel import pisa_prepare_triton


GFX1151_AVAILABLE = torch.cuda.is_available() and bool(torch.version.hip) and _gfx_target() == "gfx1151"


def _inputs(tokens=128, dtype=torch.float16):
    generator = torch.Generator().manual_seed(1234)
    q = torch.randn((2, tokens, 128), generator=generator, dtype=torch.float32).to(dtype).contiguous()
    k = torch.randn((2, tokens, 128), generator=generator, dtype=torch.float32).to(dtype).contiguous()
    v = torch.randn((2, tokens, 128), generator=generator, dtype=torch.float32).to(dtype).contiguous()
    return q, k, v


def _dense_sdpa(q, k, v):
    return F.scaled_dot_product_attention(
        q.float().unsqueeze(1),
        k.float().unsqueeze(1),
        v.float().unsqueeze(1),
        dropout_p=0.0,
        is_causal=False,
    ).squeeze(1)


class PISAAttentionTests(unittest.TestCase):
    @unittest.skipUnless(GFX1151_AVAILABLE, "requires a gfx1151 ROCm device")
    def test_gfx1151_ck_auto_dispatch_is_bfloat16_only(self):
        torch.manual_seed(13)
        q = torch.randn((1, 128, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        expected = pisa_attention(q, k, v, exact_budget=0.5, backend="reference")
        actual, info = pisa_attention(q, k, v, exact_budget=0.5, backend="auto", return_diagnostics=True)
        self.assertEqual(info["backend"], "ck_flex")
        torch.testing.assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)

        fp16 = q.to(torch.float16)
        with self.assertRaisesRegex(RuntimeError, "CK backend unavailable"):
            pisa_attention(fp16, fp16, fp16, exact_budget=0.5, backend="ck", strict_backend=True)

    @unittest.skipUnless(GFX1151_AVAILABLE, "requires a gfx1151 ROCm device")
    def test_gfx1151_triton_stats_match_reference_bf16_and_fp16(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(17)
                k = torch.randn((1, 128, 128), device="cuda", dtype=dtype)
                v = torch.randn_like(k)
                actual = pisa_prepare_triton(k, v)
                expected = _reference_block_stats(k, v, 64)
                for actual_stat, expected_stat in zip(actual[:3], expected[:3]):
                    cosine = F.cosine_similarity(actual_stat.float().flatten(), expected_stat.float().flatten(), dim=0)
                    self.assertGreater(cosine.item(), 0.9999)

    @unittest.skipUnless(GFX1151_AVAILABLE, "requires a gfx1151 ROCm device")
    def test_gfx1151_staged_attention_matches_reference_bf16_and_fp16(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(19)
                q = torch.randn((1, 128, 128), device="cuda", dtype=dtype)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                expected = pisa_attention(q, k, v, exact_budget=0.5, backend="reference")
                actual, info = pisa_attention(
                    q,
                    k,
                    v,
                    exact_budget=0.5,
                    backend="triton",
                    strict_backend=True,
                    return_diagnostics=True,
                )
                cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
                self.assertEqual(info["backend"], "triton_staged")
                self.assertGreater(cosine.item(), 0.9999)
                self.assertTrue(torch.isfinite(actual).all())

    def test_exact_budget_one_matches_dense_sdpa(self):
        q, k, v = _inputs()
        out, info = pisa_attention(q, k, v, exact_budget=1.0, backend="reference", return_diagnostics=True)
        expected = _dense_sdpa(q, k, v)
        torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=2e-3)
        self.assertEqual(info["exact_blocks_per_query"], 2)
        self.assertEqual(info["approximate_blocks_per_query"], 0)
        self.assertTrue(info["exact_online_softmax"])

    def test_centered_tail_uses_numerator_and_denominator(self):
        q, _, v = _inputs()
        generator = torch.Generator().manual_seed(4321)
        block_keys = torch.randn((2, 2, 128), generator=generator, dtype=torch.float32).to(torch.float16)
        k = block_keys.repeat_interleave(64, dim=1).contiguous()
        out, info = pisa_attention(q, k, v, exact_blocks=0, backend="reference", return_diagnostics=True)
        expected = _dense_sdpa(q, k, v)
        torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=2e-3)
        self.assertEqual(info["approximate_blocks_per_query"], 2)
        self.assertTrue(info["shared_numerator_denominator_normalization"])
        self.assertEqual(info["tail_approximation"], "block_centered_zeroth_plus_global_first_order")

    def test_approximation_is_finite_and_tracks_dense_sdpa(self):
        q, _, v = _inputs(tokens=192)
        generator = torch.Generator().manual_seed(99)
        centers = torch.randn((2, 3, 128), generator=generator, dtype=torch.float32)
        noise = torch.randn((2, 192, 128), generator=generator, dtype=torch.float32) * 0.01
        k = (centers.repeat_interleave(64, dim=1) + noise).to(torch.bfloat16).contiguous()
        q = q.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        out = pisa_attention(q, k, v, sparsity=2 / 3, backend="reference")
        expected = _dense_sdpa(q, k, v)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess((out.float() - expected).abs().mean().item(), 0.02)

    def test_global_first_order_matches_paper_formula(self):
        q, k, _ = _inputs()
        v = (0.5 * k.float() + torch.roll(k.float(), 1, dims=-1)).to(torch.float16).contiguous()
        out = pisa_attention(q, k, v, exact_blocks=0, backend="reference")

        scale = 128**-0.5
        k_blocks = k.float().reshape(2, 2, 64, 128)
        v_blocks = v.float().reshape(2, 2, 64, 128)
        k_means = k_blocks.mean(dim=2)
        v_sums = v_blocks.sum(dim=2)
        h_sum = torch.zeros((2, 128, 128), dtype=torch.float32)
        for block in range(2):
            centered = k_blocks[:, block] - k_means[:, block, None]
            h_sum += torch.bmm(centered.transpose(1, 2), v_blocks[:, block])

        alpha = torch.exp(torch.einsum("btd,bnd->btn", q.float(), k_means) * scale)
        denominator = alpha.sum(dim=-1) * 64
        zeroth = torch.einsum("btn,bnd->btd", alpha, v_sums)
        correction = torch.bmm(q.float(), h_sum) * (scale / 128) * denominator[:, :, None]
        expected = (zeroth + correction) / denominator[:, :, None]

        torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=2e-3)
        self.assertGreater(correction.abs().mean().item(), 0.01)

    def test_shape_dtype_and_training_validation(self):
        q, k, v = _inputs()
        with self.assertRaisesRegex(ValueError, "D=128"):
            pisa_attention(q[..., :64].contiguous(), k[..., :64].contiguous(), v[..., :64].contiguous())
        with self.assertRaisesRegex(ValueError, "identical"):
            pisa_attention(q, k[:, :-1].contiguous(), v)
        with self.assertRaisesRegex(ValueError, "fp16/bf16"):
            pisa_attention(q.float(), k.float(), v.float())
        with self.assertRaisesRegex(ValueError, "requiring gradients"):
            pisa_attention(q.requires_grad_(), k, v)
        with self.assertRaisesRegex(ValueError, "block_size=64"):
            pisa_attention(q.detach(), k, v, block_size=32)

    def test_budget_validation_and_triton_fallback_reporting(self):
        q, k, v = _inputs()
        with self.assertRaisesRegex(ValueError, "only one"):
            pisa_attention(q, k, v, exact_budget=0.5, sparsity=0.5)
        with self.assertRaisesRegex(ValueError, "backend"):
            pisa_attention(q, k, v, backend="unknown")
        out, info = pisa_attention(q, k, v, exact_budget=0.5, backend="triton", return_diagnostics=True)
        self.assertEqual(info["backend"], "reference")
        self.assertIn("requires_cuda_hip", info["fallback_reason"])
        self.assertTrue(torch.isfinite(out).all())
        with self.assertRaisesRegex(RuntimeError, "Triton backend unavailable"):
            pisa_attention(q, k, v, backend="triton", strict_backend=True)


if __name__ == "__main__":
    unittest.main()
