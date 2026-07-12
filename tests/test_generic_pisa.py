import sys
import types
import unittest
from unittest import mock

import torch

from rdna35_block_attention.generic_pisa import MIN_TOKENS, _block_stats, make_generic_pisa_override
from rdna35_block_attention.pisa_runtime import PISARuntimeState


class FakeCudaTensor(torch.Tensor):
    device_index = 0

    @staticmethod
    def __new__(cls, tensor):
        return torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)

    @property
    def device(self):
        return torch.device("cuda", self.device_index)


class OtherFakeCudaTensor(FakeCudaTensor):
    device_index = 1


def _cuda_tensor(shape, dtype=torch.float16, tensor_type=FakeCudaTensor):
    return tensor_type(torch.zeros(shape, dtype=dtype))


class GenericPISADispatchTests(unittest.TestCase):
    def test_validated_ck_hyd_profile_has_priority(self):
        q = torch.randn(2, 128, 128, dtype=torch.bfloat16)
        outputs = (
            torch.randn(2, 2, 128, dtype=q.dtype),
            torch.randn(2, 2, 128),
            torch.randn(2, 2, 128, dtype=q.dtype),
            torch.randn(2, 128, 128),
        )
        ck = types.SimpleNamespace(
            capabilities=lambda: {"hyd_stats_head_dims": (128,), "hyd_stats_dtypes": (torch.bfloat16,)},
            block_stats_hyd=mock.Mock(return_value=outputs),
        )
        with mock.patch.dict(sys.modules, {"rdna35_pisa_ck": ck}):
            result = _block_stats(q, q, q, return_backend=True)
        ck.block_stats_hyd.assert_called_once_with(q, q, q)
        self.assertEqual(result[:4], outputs)
        self.assertEqual(result[4].tolist(), [64.0, 64.0])
        self.assertEqual(result[5], "ck_hyd")

    def test_unvalidated_head_dimension_uses_triton(self):
        q = torch.randn(2, 128, 64, dtype=torch.float16)
        ck = types.SimpleNamespace(
            capabilities=lambda: {"hyd_stats_head_dims": (128,), "hyd_stats_dtypes": (torch.bfloat16,)},
            block_stats_hyd=mock.Mock(),
        )
        triton_outputs = (
            torch.randn(2, 2, 64),
            torch.randn(2, 2, 64),
            torch.randn(2, 64, 64),
            [64, 64],
        )
        with mock.patch.dict(sys.modules, {"rdna35_pisa_ck": ck}), mock.patch(
            "rdna35_block_attention.pisa_kernel.pisa_prepare_triton", return_value=triton_outputs
        ) as triton_stats:
            q_centroids, k_centroids, v_means, h_sum, lengths, backend = _block_stats(q, q, q, return_backend=True)
        ck.block_stats_hyd.assert_not_called()
        triton_stats.assert_called_once_with(q, q, block_size=64)
        self.assertEqual(q_centroids.shape, (2, 2, 64))
        self.assertIs(k_centroids, triton_outputs[0])
        self.assertIs(h_sum, triton_outputs[2])
        self.assertEqual(v_means.shape, triton_outputs[1].shape)
        self.assertEqual(lengths.tolist(), [64.0, 64.0])
        self.assertEqual(backend, "triton_hyd")

    def test_image_and_video_flattened_self_attention_use_generic_pisa(self):
        profiles = (
            ("sd15_d40", 8, 40),
            ("sd15_d80", 8, 80),
            ("sd15_d160", 8, 160),
            ("sdxl_d64", 10, 64),
            ("wan_d128", 12, 128),
            ("ltx_d64", 16, 64),
        )
        for profile, heads, head_dim in profiles:
            with self.subTest(profile=profile, heads=heads, head_dim=head_dim):
                q = _cuda_tensor((1, MIN_TOKENS, heads * head_dim))
                state = PISARuntimeState(armed=True)
                override = make_generic_pisa_override(
                    exact_budget=0.25,
                    device_index=0,
                    previous_override=None,
                    runtime_state=state,
                )
                pisa_output = torch.zeros((heads, MIN_TOKENS, head_dim), dtype=q.dtype)
                with mock.patch(
                    "rdna35_block_attention.generic_pisa.generic_pisa_attention",
                    return_value=(pisa_output, "triton_hyd"),
                ) as pisa:
                    output = override(mock.Mock(), q, q, q, heads, is_self_attention=True)

                self.assertEqual(output.shape, q.shape)
                self.assertEqual(pisa.call_args.kwargs["exact_budget"], 0.25)
                self.assertEqual(state.backend_counts["triton_hyd"], 1)
                self.assertEqual(state.per_layer_hits[-1], 1)

    def test_cross_masked_short_and_cached_video_calls_fall_back(self):
        heads, head_dim = 4, 16
        eligible = _cuda_tensor((1, MIN_TOKENS, heads * head_dim))
        short = _cuda_tensor((1, MIN_TOKENS - 64, heads * head_dim))
        cached_kv = _cuda_tensor((1, MIN_TOKENS + 64, heads * head_dim))
        other_device = _cuda_tensor(eligible.shape, tensor_type=OtherFakeCudaTensor)
        bf16 = _cuda_tensor(eligible.shape, dtype=torch.bfloat16)
        previous = mock.Mock(return_value="fallback")
        state = PISARuntimeState(armed=True)
        override = make_generic_pisa_override(
            exact_budget=0.25,
            device_index=0,
            previous_override=previous,
            runtime_state=state,
        )

        cases = (
            (eligible, eligible, eligible, {"is_self_attention": False}, "not_explicit_self_attention"),
            (eligible, eligible, eligible, {"is_self_attention": True, "mask": torch.ones(1)}, "attention_mask_is_not_supported"),
            (eligible, eligible, eligible, {"is_self_attention": True, "enable_gqa": True}, "gqa_is_not_supported"),
            (eligible, other_device, other_device, {"is_self_attention": True}, "qkv_must_share_one_device"),
            (eligible, bf16, bf16, {"is_self_attention": True}, "matching_fp16_or_bf16_qkv_are_required"),
            (short, short, short, {"is_self_attention": True}, f"tokens_{MIN_TOKENS - 64}_below_{MIN_TOKENS}"),
            (eligible, cached_kv, cached_kv, {"is_self_attention": True}, "matching_btc_qkv_divisible_by_heads_are_required"),
        )
        for q, k, v, kwargs, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(override(mock.Mock(), q, k, v, heads, **kwargs), "fallback")
                self.assertEqual(state.fallback_reasons[reason], 1)

        self.assertEqual(previous.call_count, len(cases))

    def test_backend_compile_failure_falls_back_but_oom_is_loud(self):
        heads, head_dim = 4, 16
        q = _cuda_tensor((1, MIN_TOKENS, heads * head_dim))
        previous = mock.Mock(return_value="fallback")
        state = PISARuntimeState(armed=True)
        override = make_generic_pisa_override(
            exact_budget=0.25,
            device_index=0,
            previous_override=previous,
            runtime_state=state,
        )

        with mock.patch(
            "rdna35_block_attention.generic_pisa.generic_pisa_attention",
            side_effect=RuntimeError("compile failed"),
        ):
            self.assertEqual(override(mock.Mock(), q, q, q, heads, is_self_attention=True), "fallback")
        self.assertEqual(state.fallback_reasons["pisa_backend_error_RuntimeError"], 1)

        with mock.patch(
            "rdna35_block_attention.generic_pisa.generic_pisa_attention",
            side_effect=torch.OutOfMemoryError("out of memory"),
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                override(mock.Mock(), q, q, q, heads, is_self_attention=True)


if __name__ == "__main__":
    unittest.main()
