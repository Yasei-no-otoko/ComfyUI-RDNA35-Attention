import sys
import types
import unittest
from unittest import mock

import torch

from rdna35_block_attention.generic_pisa import _block_stats


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
            result = _block_stats(q, q, q)
        ck.block_stats_hyd.assert_called_once_with(q, q, q)
        self.assertEqual(result[:4], outputs)
        self.assertEqual(result[4].tolist(), [64.0, 64.0])

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
            q_centroids, k_centroids, v_means, h_sum, lengths = _block_stats(q, q, q)
        ck.block_stats_hyd.assert_not_called()
        triton_stats.assert_called_once_with(q, q, block_size=64)
        self.assertEqual(q_centroids.shape, (2, 2, 64))
        self.assertIs(k_centroids, triton_outputs[0])
        self.assertIs(h_sum, triton_outputs[2])
        self.assertEqual(v_means.shape, triton_outputs[1].shape)
        self.assertEqual(lengths.tolist(), [64.0, 64.0])


if __name__ == "__main__":
    unittest.main()
