from __future__ import annotations

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rdna35_block_attention.anima_pisa_integration import (
    ANIMA_PISA_HEAD_DIM,
    ANIMA_PISA_HEADS,
    ANIMA_PISA_TOKENS,
    install_anima_pisa_attention,
    make_anima_pisa_attn_op,
)
from rdna35_block_attention.pisa_runtime import PISARuntimeState


class FakeCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor):
        return torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)

    @property
    def device(self):
        return torch.device("cuda", 0)


def _qkv(*, tokens=ANIMA_PISA_TOKENS, dtype=torch.bfloat16, requires_grad=False):
    tensor = torch.zeros((1, tokens, ANIMA_PISA_HEADS, ANIMA_PISA_HEAD_DIM), dtype=dtype, requires_grad=requires_grad)
    tensor = FakeCudaTensor(tensor)
    return tensor, tensor, tensor


class DummyPatcher:
    def __init__(self):
        self.model = SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=SimpleNamespace(n_heads=16, head_dim=128, attn_op=Mock())) for _ in range(28)]
        )
        self.object_patches = {}

    def get_model_object(self, name):
        self.assert_name = name
        return self.model

    def add_object_patch(self, name, value):
        self.object_patches[name] = value


class AnimaPISAIntegrationTests(unittest.TestCase):
    def test_direct_op_hits_native_without_markers(self):
        q, k, v = _qkv()
        native = Mock(return_value=torch.zeros((1, ANIMA_PISA_TOKENS, ANIMA_PISA_HEADS * ANIMA_PISA_HEAD_DIM), dtype=torch.bfloat16))
        original = Mock()
        state = PISARuntimeState(armed=True)
        op = make_anima_pisa_attn_op(original, native_forward=native, exact_blocks=23, device_index=0, layer_index=4, runtime_state=state)

        result = op(q, k, v)

        self.assertEqual(result.shape, (1, ANIMA_PISA_TOKENS, ANIMA_PISA_HEADS * ANIMA_PISA_HEAD_DIM))
        native.assert_called_once()
        original.assert_not_called()
        self.assertEqual(native.call_args.kwargs["exact_blocks"], 23)
        self.assertEqual(native.call_args.args[0].shape, (1, ANIMA_PISA_HEADS, ANIMA_PISA_TOKENS, ANIMA_PISA_HEAD_DIM))
        self.assertEqual(state.per_layer_hits, {4: 1})
        self.assertEqual(state.shape_counts[(1, ANIMA_PISA_HEADS, ANIMA_PISA_TOKENS, ANIMA_PISA_HEAD_DIM)], 1)

    def test_other_profile_chains_original_op(self):
        q, k, v = _qkv(tokens=64)
        original = Mock(return_value="fallback")
        state = PISARuntimeState(armed=True)
        op = make_anima_pisa_attn_op(original, native_forward=Mock(), exact_blocks=23, device_index=0, layer_index=4, runtime_state=state)

        self.assertEqual(op(q, k, v), "fallback")
        original.assert_called_once_with(q, k, v, transformer_options=None)
        self.assertEqual(state.fallback_reasons["shape_(64, 16, 128)_is_not_t9216_h16_d128"], 1)

    def test_eligible_native_failure_is_loud(self):
        q, k, v = _qkv()
        state = PISARuntimeState(armed=True)
        op = make_anima_pisa_attn_op(Mock(), native_forward=Mock(side_effect=ValueError("native failure")), exact_blocks=23, device_index=0, layer_index=4, runtime_state=state)

        with self.assertRaisesRegex(RuntimeError, "eligible Anima"):
            op(q, k, v)
        self.assertTrue(state.failed)

    def test_installs_requested_layer_range(self):
        patcher = DummyPatcher()

        patched = install_anima_pisa_attention(
            patcher,
            native_forward=Mock(),
            exact_blocks=23,
            device_index=0,
            first_layer=8,
            last_layer=19,
        )

        self.assertEqual(patched, 12)
        self.assertEqual(len(patcher.object_patches), 12)
        self.assertNotIn("diffusion_model.blocks.7.self_attn.attn_op", patcher.object_patches)
        self.assertIn("diffusion_model.blocks.8.self_attn.attn_op", patcher.object_patches)
        self.assertIn("diffusion_model.blocks.19.self_attn.attn_op", patcher.object_patches)
        self.assertNotIn("diffusion_model.blocks.20.self_attn.attn_op", patcher.object_patches)


if __name__ == "__main__":
    unittest.main()
