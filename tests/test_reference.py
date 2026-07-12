from __future__ import annotations

import math
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdna35_block_attention.reference import (
    fixed_block_attention_bthd,
    fixed_block_attention_ref,
    fixed_block_attention_sdpa,
)


def torch_block_loop(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int, causal: bool) -> torch.Tensor:
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for start in range(0, q.shape[-2], block_size):
        end = min(start + block_size, q.shape[-2])
        qb = q[:, :, start:end, :].float()
        kb = k[:, :, start:end, :].float()
        vb = v[:, :, start:end, :].float()
        scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale
        if causal:
            mask = torch.ones((end - start, end - start), dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        out[:, :, start:end, :] = torch.matmul(torch.softmax(scores, dim=-1), vb).to(q.dtype)
    return out


class ReferenceTests(unittest.TestCase):
    def test_bhtd_matches_torch_block_loop(self):
        for batch in (1, 2):
            for heads in (1, 4):
                for tokens in (64, 128, 192):
                    for head_dim in (32, 64, 128):
                        for causal in (False, True):
                            with self.subTest(batch=batch, heads=heads, tokens=tokens, head_dim=head_dim, causal=causal):
                                torch.manual_seed(batch * 100000 + heads * 1000 + tokens + head_dim)
                                q = torch.randn(batch, heads, tokens, head_dim, dtype=torch.float32)
                                k = torch.randn_like(q)
                                v = torch.randn_like(q)
                                expected = torch_block_loop(q, k, v, 64, causal)
                                actual = fixed_block_attention_ref(q, k, v, block_size=64, causal=causal, layout="bhtd")
                                self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_bh_t_d_layout(self):
        q = torch.randn(8, 65, 32, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        actual = fixed_block_attention_ref(q, k, v, block_size=64, layout="bh_t_d")
        expected = fixed_block_attention_ref(q.reshape(2, 4, 65, 32), k.reshape(2, 4, 65, 32), v.reshape(2, 4, 65, 32), block_size=64, layout="bhtd").reshape(8, 65, 32)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_bthd_wrapper(self):
        batch, heads, tokens, head_dim = 2, 4, 127, 32
        q = torch.randn(batch, tokens, heads * head_dim, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        actual = fixed_block_attention_bthd(q, k, v, heads=heads, block_size=64)
        q4 = q.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k4 = k.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v4 = v.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3).contiguous()
        expected = fixed_block_attention_ref(q4, k4, v4, block_size=64, layout="bhtd").permute(0, 2, 1, 3).contiguous().reshape(batch, tokens, heads * head_dim)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_sdpa_block_mask_matches_reference(self):
        for tokens in (64, 127, 128):
            for causal in (False, True):
                with self.subTest(tokens=tokens, causal=causal):
                    q = torch.randn(2, 3, tokens, 32, dtype=torch.float32)
                    k = torch.randn_like(q)
                    v = torch.randn_like(q)
                    expected = fixed_block_attention_ref(q, k, v, block_size=64, causal=causal, layout="bhtd")
                    actual = fixed_block_attention_sdpa(q, k, v, block_size=64, causal=causal, layout="bhtd")
                    self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
