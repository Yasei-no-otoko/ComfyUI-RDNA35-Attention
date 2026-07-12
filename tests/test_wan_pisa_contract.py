import unittest
from unittest import mock

import torch

from comfy.ldm.wan.ar_model import CausalWanSelfAttention


class WanPISAContractTests(unittest.TestCase):
    def test_causal_wan_marks_kv_cached_attention(self):
        attention = CausalWanSelfAttention.__new__(CausalWanSelfAttention)
        torch.nn.Module.__init__(attention)
        attention.num_heads = 2
        attention.head_dim = 4
        attention.q = torch.nn.Identity()
        attention.k = torch.nn.Identity()
        attention.v = torch.nn.Identity()
        attention.o = torch.nn.Identity()
        attention.norm_q = torch.nn.Identity()
        attention.norm_k = torch.nn.Identity()
        x = torch.randn(1, 8, 8)
        cache = {
            "k": torch.empty(1, 16, 2, 4),
            "v": torch.empty(1, 16, 2, 4),
            "end": 0,
        }

        with mock.patch("comfy.ldm.wan.ar_model.apply_rope1", side_effect=lambda value, freqs: value), mock.patch(
            "comfy.ldm.wan.ar_model.optimized_attention", return_value=x
        ) as optimized_attention:
            attention(x, freqs=None, kv_cache=cache, transformer_options={"optimized_attention_override": object()})

        self.assertTrue(optimized_attention.call_args.kwargs["is_self_attention"])
        self.assertTrue(optimized_attention.call_args.kwargs["is_kv_cached_attention"])
        self.assertEqual(cache["end"], 8)


if __name__ == "__main__":
    unittest.main()
