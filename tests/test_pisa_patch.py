from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import Mock, patch

import torch


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rdna35_block_attention.pisa_patch import make_pisa_attention_override, patch_model_pisa_attention


TOKENS = 64
HEAD_DIM = 128


class FakeCudaTensor(torch.Tensor):
    device_index = 0

    @staticmethod
    def __new__(cls, tensor):
        return torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)

    @property
    def device(self):
        return torch.device("cuda", self.device_index)

    @property
    def is_cuda(self):
        return True


class OtherFakeCudaTensor(FakeCudaTensor):
    device_index = 1


def _tensor(shape=(1, 2, TOKENS, HEAD_DIM), *, value=0.0, dtype=torch.bfloat16, requires_grad=False, cls=FakeCudaTensor):
    batch, heads, tokens, head_dim = shape
    storage = torch.full((batch, tokens, heads, head_dim), value, dtype=dtype, requires_grad=requires_grad)
    return cls(storage.permute(0, 2, 1, 3))


def _matching_tensors(**kwargs):
    tensor = _tensor(**kwargs)
    return tensor, tensor, tensor


class DummyModel:
    def __init__(self, transformer_options=None):
        self.model_options = {"transformer_options": dict(transformer_options or {})}

    def clone(self):
        return DummyModel(self.model_options["transformer_options"])


def _make_override(previous_override=None, native_forward=None):
    return make_pisa_attention_override(
        exact_budget=0.25,
        allowed_tokens=frozenset((TOKENS,)),
        expected_token_shape=(1, 8, 8),
        device_index=0,
        native_forward=native_forward or Mock(name="native_forward"),
        verbose_fallbacks=False,
        previous_override=previous_override,
    )


class PISAPatchTests(unittest.TestCase):
    def test_patch_is_installed_only_on_cloned_model(self):
        previous = Mock(name="previous_override")
        model = DummyModel({"optimized_attention_override": previous})

        ck_module = Mock()
        ck_module.build_info.return_value = {"api": 5}
        ck_module.capabilities.return_value = {"spatial_sparse_exact_blocks": (23,)}
        with patch("rdna35_block_attention.pisa_patch._gfx1151_device_index", return_value=(0, None)), patch.dict(sys.modules, {"rdna35_pisa_ck": ck_module}):
            patched, _ = patch_model_pisa_attention(
                model,
                enabled=True,
                exact_budget=0.15625,
                token_policy="auto_9216",
                verbose_fallbacks=False,
            )

        self.assertIsNot(patched, model)
        self.assertIs(model.model_options["transformer_options"]["optimized_attention_override"], previous)
        self.assertIsNot(patched.model_options["transformer_options"]["optimized_attention_override"], previous)

    def test_generic_patch_accepts_non_anima_sparse_budget(self):
        model = DummyModel()
        ck_module = Mock()
        ck_module.build_info.return_value = {"api": 5}
        ck_module.capabilities.return_value = {"spatial_sparse_exact_blocks": (23,)}

        with patch("rdna35_block_attention.pisa_patch._gfx1151_device_index", return_value=(0, None)), patch.dict(sys.modules, {"rdna35_pisa_ck": ck_module}):
            patched, info = patch_model_pisa_attention(
                model,
                enabled=True,
                exact_budget=0.25,
                token_policy="auto_9216",
                verbose_fallbacks=False,
            )

        self.assertIsNot(patched, model)
        self.assertIn("generic gfx1151 PISA", info)

    def test_explicit_self_attention_marker_is_required(self):
        q = _tensor(value=1.0)
        k = _tensor(value=2.0)
        v = _tensor(value=3.0)
        native_output = torch.zeros((1, TOKENS, 2 * HEAD_DIM), dtype=q.dtype)
        previous = Mock(return_value="previous")
        native = Mock(return_value=native_output)
        override = _make_override(previous, native)

        self.assertEqual(override(Mock(), q, k, v, 2, skip_reshape=True), "previous")
        self.assertEqual(override(Mock(), q, k, v, 2, skip_reshape=True, is_self_attention=False), "previous")
        override(
            Mock(),
            q,
            k,
            v,
            2,
            skip_reshape=True,
            is_self_attention=True,
            is_initial_transformer_block=False,
            attention_token_shape=(1, 8, 8),
        )

        self.assertEqual(previous.call_count, 2)
        native.assert_called_once()

    def test_unsupported_calls_chain_previous_override_or_original(self):
        q = torch.zeros((1, 2, 64, HEAD_DIM), dtype=torch.bfloat16)
        previous = Mock(return_value="previous")
        original = Mock(return_value="original")

        chained = _make_override(previous)
        self.assertEqual(chained(original, q, q, q, 2, skip_reshape=True, is_self_attention=False), "previous")
        previous.assert_called_once()
        original.assert_not_called()

        unchained = _make_override()
        self.assertEqual(unchained(original, q, q, q, 2, skip_reshape=True, is_self_attention=False), "original")
        original.assert_called_once()

    def test_ineligible_calls_never_enter_native_pisa(self):
        q = _tensor(value=1.0)
        k = _tensor(value=2.0)
        v = _tensor(value=3.0)
        cases = (
            ("mask", lambda: ((q, k, v), {"mask": object(), "skip_reshape": True})),
            ("reshape_required", lambda: ((q, k, v), {"skip_reshape": False})),
            ("unmerged_output", lambda: ((q, k, v), {"skip_reshape": True, "skip_output_reshape": True})),
            ("mismatched_shape", lambda: ((q, _tensor(shape=(1, 2, TOKENS - 1, HEAD_DIM)), v), {"skip_reshape": True})),
            ("heads", lambda: ((q, k, v), {"heads": 1, "skip_reshape": True})),
            ("head_dim", lambda: (_matching_tensors(shape=(1, 2, TOKENS, 64)), {"skip_reshape": True})),
            ("dtype", lambda: (_matching_tensors(dtype=torch.float16), {"skip_reshape": True})),
            ("different_dtype", lambda: ((q, _tensor(dtype=torch.float16), v), {"skip_reshape": True})),
            ("different_device", lambda: ((q, _tensor(cls=OtherFakeCudaTensor), v), {"skip_reshape": True})),
            ("requires_grad", lambda: ((_tensor(requires_grad=True), k, v), {"skip_reshape": True})),
            ("token_count", lambda: (_matching_tensors(shape=(1, 2, 128, HEAD_DIM)), {"skip_reshape": True})),
            ("attention_precision", lambda: ((q, k, v), {"attn_precision": torch.float32, "skip_reshape": True})),
            ("gqa", lambda: ((q, k, v), {"enable_gqa": True, "skip_reshape": True})),
            ("contiguous_bhtd", lambda: (tuple(tensor.contiguous() for tensor in (q, k, v)), {"skip_reshape": True, "attention_token_shape": (1, 8, 8)})),
        )

        for name, make_case in cases:
            with self.subTest(name=name):
                tensors, kwargs = make_case()
                heads = kwargs.pop("heads", 2)
                previous = Mock(return_value="previous")
                native = Mock()
                override = _make_override(previous, native)
                result = override(Mock(), *tensors, heads, is_self_attention=True, is_initial_transformer_block=False, **kwargs)
                self.assertEqual(result, "previous")
                previous.assert_called_once()
                native.assert_not_called()

        cpu = torch.zeros((1, 2, TOKENS, HEAD_DIM), dtype=torch.bfloat16)
        previous = Mock(return_value="previous")
        native = Mock()
        override = _make_override(previous, native)
        result = override(Mock(), cpu, cpu, cpu, 2, skip_reshape=True, is_self_attention=True, is_initial_transformer_block=False)
        self.assertEqual(result, "previous")
        previous.assert_called_once()
        native.assert_not_called()

    def test_eligible_call_uses_native_bhtd_and_merged_output(self):
        batch, heads = 2, 2
        shape = (batch, heads, TOKENS, HEAD_DIM)
        q = _tensor(shape, value=1.0)
        k = _tensor(shape, value=2.0)
        v = _tensor(shape, value=3.0)
        native_output = torch.arange(batch * TOKENS * heads * HEAD_DIM, dtype=q.dtype).reshape(batch, TOKENS, heads * HEAD_DIM)
        native = Mock(return_value=native_output)
        override = _make_override(native_forward=native)

        output = override(
            Mock(),
            q,
            k,
            v,
            heads,
            skip_reshape=True,
            is_self_attention=True,
            is_initial_transformer_block=False,
            attention_token_shape=(1, 8, 8),
        )

        native.assert_called_once()
        native_q, native_k, native_v = native.call_args.args[:3]
        for tensor, value in zip((native_q, native_k, native_v), (1.0, 2.0, 3.0)):
            self.assertIs(tensor, (q, k, v)[int(value) - 1])
            self.assertEqual(tensor[0, 0, 0, 0].item(), value)
        self.assertEqual(native.call_args.args[3], 1)

        self.assertIs(output, native_output)

    def test_skip_output_reshape_falls_back(self):
        batch, heads = 1, 2
        q = _tensor((batch, heads, TOKENS, HEAD_DIM), value=1.0)
        native = Mock()
        previous = Mock(return_value="previous")
        override = _make_override(previous, native)

        output = override(
            Mock(),
            q,
            q,
            q,
            heads,
            skip_reshape=True,
            skip_output_reshape=True,
            is_self_attention=True,
            is_initial_transformer_block=False,
        )

        self.assertEqual(output, "previous")
        native.assert_not_called()

    def test_native_exception_propagates_without_fallback(self):
        q = _tensor(value=1.0)
        previous = Mock(return_value="previous")
        original = Mock(return_value="original")
        native = Mock(side_effect=RuntimeError("native failure"))
        override = _make_override(previous, native)

        with self.assertRaisesRegex(RuntimeError, "native failure"):
            override(
                original,
                q,
                q,
                q,
                2,
                skip_reshape=True,
                is_self_attention=True,
                is_initial_transformer_block=False,
                attention_token_shape=(1, 8, 8),
            )

        previous.assert_not_called()
        original.assert_not_called()

    def test_layers_before_start_layer_chain_to_previous_override(self):
        q = _tensor(value=1.0)
        previous = Mock(return_value="previous")
        native = Mock()
        override = _make_override(previous, native)

        result = override(
            Mock(),
            q,
            q,
            q,
            2,
            skip_reshape=True,
            is_self_attention=True,
            is_initial_transformer_block=True,
            attention_token_shape=(1, 8, 8),
        )

        self.assertEqual(result, "previous")
        native.assert_not_called()

    def test_bhtd_view_is_forwarded_without_python_layout_copy(self):
        tokens = 256
        sequence = torch.arange(tokens, dtype=torch.bfloat16).reshape(1, tokens, 1, 1)
        q = FakeCudaTensor(sequence.expand(1, tokens, 1, HEAD_DIM).clone().permute(0, 2, 1, 3))
        merged = torch.zeros((1, tokens, HEAD_DIM), dtype=q.dtype)
        native = Mock(return_value=merged)
        override = make_pisa_attention_override(
            exact_budget=0.25,
            allowed_tokens=frozenset((tokens,)),
            expected_token_shape=(1, 16, 16),
            device_index=0,
            native_forward=native,
            verbose_fallbacks=False,
            previous_override=None,
        )

        output = override(
            Mock(),
            q,
            q,
            q,
            1,
            skip_reshape=True,
            is_self_attention=True,
            is_initial_transformer_block=False,
            attention_token_shape=(1, 16, 16),
        )

        self.assertIs(native.call_args.args[0], q)
        self.assertIs(output, merged)


if __name__ == "__main__":
    unittest.main()
