from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdna35_block_attention import minimax_h3_gfx1151_attention as h3_attention


class DummyModel:
    def __init__(self, diffusion_model=None):
        self.diffusion_model = diffusion_model
        self.model_options = {"transformer_options": {}}

    def get_model_object(self, name):
        if name != "diffusion_model" or self.diffusion_model is None:
            raise KeyError(name)
        return self.diffusion_model

    def clone(self):
        cloned = DummyModel(self.diffusion_model)
        cloned.model_options = {"transformer_options": self.model_options["transformer_options"].copy()}
        return cloned


def make_minimax_h3_model():
    cls = type("MiniMaxH3Model", (), {})
    cls.__module__ = "comfy.ldm.minimax.model"
    diffusion_model = cls()
    diffusion_model.blocks = []
    diffusion_model.hidden_size = 5376
    return DummyModel(diffusion_model)


class MiniMaxH3PackingTests(unittest.TestCase):
    def test_pack_profiles_keep_values_and_use_one_allocation(self):
        short_qkv = torch.arange(1 * 9170 * 3, dtype=torch.float32).reshape(1, 9170, 3, 1, 1)
        short_q = short_qkv[:, :, 0].permute(0, 2, 1, 3)
        short_k = short_qkv[:, :, 1].permute(0, 2, 1, 3)
        short_v = short_qkv[:, :, 2].permute(0, 2, 1, 3)
        packed_q, packed_k, packed_v, profile = h3_attention.pack_minimax_h3_qkv(short_q, short_k, short_v)
        self.assertEqual(profile, "kv")
        self.assertIs(packed_q, short_q)
        self.assertTrue(packed_k.is_contiguous())
        self.assertTrue(packed_v.is_contiguous())
        self.assertTrue(torch.equal(packed_k, short_k))
        self.assertTrue(torch.equal(packed_v, short_v))
        self.assertIs(packed_k._base, packed_v._base)

        long_qkv = torch.arange(1 * 12000 * 3, dtype=torch.float32).reshape(1, 12000, 3, 1, 1)
        long_q = long_qkv[:, :, 0].permute(0, 2, 1, 3)
        long_k = long_qkv[:, :, 1].permute(0, 2, 1, 3)
        long_v = long_qkv[:, :, 2].permute(0, 2, 1, 3)
        packed_q, packed_k, packed_v, profile = h3_attention.pack_minimax_h3_qkv(long_q, long_k, long_v)
        self.assertEqual(profile, "qkv")
        self.assertTrue(packed_q.is_contiguous())
        self.assertTrue(torch.equal(packed_q, long_q))
        self.assertTrue(torch.equal(packed_k, long_k))
        self.assertTrue(torch.equal(packed_v, long_v))
        self.assertIs(packed_q._base, packed_k._base)
        self.assertIs(packed_q._base, packed_v._base)

    def test_packing_does_not_change_attention_math(self):
        qkv = torch.randn(1, 33, 3, 2, 8)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        packed_q, packed_k, packed_v, _ = h3_attention.pack_minimax_h3_qkv(q, k, v)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        actual = torch.nn.functional.scaled_dot_product_attention(packed_q, packed_k, packed_v)
        self.assertTrue(torch.equal(actual, expected))

    def test_short_h3_sequence_is_left_on_existing_backend(self):
        q = torch.randn(1, 56, 8, 128, dtype=torch.bfloat16)
        reason = h3_attention._reject_reason(q, q, q, 56, None, True, False, 0, {})
        self.assertEqual(reason, "tokens_below_4608")

    def test_unvalidated_larger_sequence_is_left_on_existing_backend(self):
        q = torch.empty(1, 56, 16501, 128, dtype=torch.bfloat16, device="meta")
        reason = h3_attention._reject_reason(q, q, q, 56, None, True, False, 0, {})
        self.assertEqual(reason, "tokens_above_16500")

    def test_override_preserves_arguments_and_chains_after_packing(self):
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        calls = []

        def previous(original_func, packed_q, packed_k, packed_v, heads, **kwargs):
            calls.append((packed_q, packed_k, packed_v, heads, kwargs))
            return "previous"

        override = h3_attention.make_minimax_h3_gfx1151_attention_override(
            device_index=0,
            verbose_fallbacks=False,
            previous_override=previous,
        )
        with mock.patch.object(h3_attention, "_reject_reason", return_value=None):
            result = override(lambda *args, **kwargs: "original", q, k, v, 2, skip_reshape=True)
        self.assertEqual(result, "previous")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][3], 2)
        self.assertTrue(calls[0][4]["skip_reshape"])

    def test_patch_is_h3_only_and_model_local(self):
        unsupported = DummyModel()
        patched, info = h3_attention.patch_minimax_h3_gfx1151_attention(
            unsupported,
            enabled=True,
            verbose_fallbacks=False,
        )
        self.assertIs(patched, unsupported)
        self.assertIn("not found", info)

        model = make_minimax_h3_model()
        with mock.patch.object(h3_attention, "_gfx1151_device_index", return_value=(0, None)):
            patched, info = h3_attention.patch_minimax_h3_gfx1151_attention(
                model,
                enabled=True,
                verbose_fallbacks=False,
            )
        self.assertIsNot(patched, model)
        self.assertNotIn("optimized_attention_override", model.model_options["transformer_options"])
        self.assertIn("optimized_attention_override", patched.model_options["transformer_options"])
        self.assertIn("MiniMax-H3 gfx1151", info)

    def test_patch_does_not_replace_container_aware_override(self):
        model = make_minimax_h3_model()

        def previous_override(*args, **kwargs):
            return None

        previous_override.container_function = lambda *args, **kwargs: None
        model.model_options["transformer_options"]["optimized_attention_override"] = previous_override
        with mock.patch.object(h3_attention, "_gfx1151_device_index", return_value=(0, None)):
            patched, info = h3_attention.patch_minimax_h3_gfx1151_attention(
                model,
                enabled=True,
                verbose_fallbacks=False,
            )
        self.assertIs(patched, model)
        self.assertIn("container-aware", info)


def gfx1151_available() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    target = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":", 1)[0]
    return target == "gfx1151"


@unittest.skipUnless(gfx1151_available(), "gfx1151 ROCm device not available")
class MiniMaxH3GFX1151IntegrationTests(unittest.TestCase):
    def test_bf16_h3_layout_matches_existing_attention(self):
        torch.manual_seed(4608)
        qkv = torch.randn(1, 4608, 3, 56, 128, device="cuda", dtype=torch.bfloat16)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].clone().permute(0, 2, 1, 3)
        packed_q, packed_k, packed_v, profile = h3_attention.pack_minimax_h3_qkv(q, k, v)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        actual = torch.nn.functional.scaled_dot_product_attention(packed_q, packed_k, packed_v)
        torch.cuda.synchronize()
        self.assertEqual(profile, "kv")
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=5e-2, rtol=5e-2))
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
