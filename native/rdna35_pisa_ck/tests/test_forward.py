from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

import rdna35_pisa_ck


BLOCK_SIZE = 64
HEAD_DIM = 128


def _gfx1151_available() -> bool:
    if not torch.cuda.is_available() or not torch.version.hip:
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0] == "gfx1151"


GFX1151_AVAILABLE = _gfx1151_available()


def _inputs(tokens: int, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn((1, tokens, HEAD_DIM), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def _pisa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    exact_blocks: int,
    scale: float | None = None,
    sink_block: int | None = None,
) -> torch.Tensor:
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    batch_heads, tokens, _ = q.shape
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_value = 1.0 / math.sqrt(HEAD_DIM) if scale is None else float(scale)

    lengths = [min(BLOCK_SIZE, tokens - block * BLOCK_SIZE) for block in range(blocks)]
    lengths_tensor = torch.tensor(lengths, device=q.device, dtype=torch.float32)
    q_centroids = torch.stack(
        [q_float[:, start : start + length].mean(dim=1) for start, length in zip(range(0, tokens, BLOCK_SIZE), lengths)],
        dim=1,
    ).to(q.dtype).float()
    k_centroids_float = torch.stack(
        [k_float[:, start : start + length].mean(dim=1) for start, length in zip(range(0, tokens, BLOCK_SIZE), lengths)],
        dim=1,
    )
    k_centroids = k_centroids_float.to(k.dtype).float()
    v_sums = torch.stack(
        [v_float[:, start : start + length].sum(dim=1) for start, length in zip(range(0, tokens, BLOCK_SIZE), lengths)],
        dim=1,
    ).to(v.dtype).float()
    v_means = v_sums / lengths_tensor[None, :, None]

    selected = torch.zeros((batch_heads, blocks, blocks), device=q.device, dtype=torch.bool)
    if exact_blocks:
        route_scores = torch.bmm(q_centroids, k_centroids.transpose(1, 2)) * scale_value
        if sink_block is not None:
            route_scores[..., sink_block] = torch.inf
        indices = torch.topk(route_scores, exact_blocks, dim=-1).indices
        selected.scatter_(2, indices, True)

    h_sum = torch.zeros((batch_heads, HEAD_DIM, HEAD_DIM), device=q.device, dtype=torch.float32)
    for block, (start, length) in enumerate(zip(range(0, tokens, BLOCK_SIZE), lengths)):
        centered = (k_float[:, start : start + length] - k_centroids_float[:, block, None]).to(k.dtype).float()
        h_sum.add_(torch.bmm(centered.transpose(1, 2), v_float[:, start : start + length]))
    correction = torch.bmm(q_float, h_sum) * (scale_value / tokens)
    values = torch.cat((v_float, v_means), dim=1)
    output = torch.empty_like(q_float)

    for query_block in range(blocks):
        start = query_block * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, tokens)
        q_block = q_float[:, start:end]

        exact_logits = torch.bmm(q_block, k_float.transpose(1, 2)) * scale_value
        exact_token_mask = selected[:, query_block].repeat_interleave(BLOCK_SIZE, dim=1)[:, :tokens]
        exact_logits.masked_fill_(~exact_token_mask[:, None, :], -torch.inf)

        approximate_logits = torch.bmm(q_block, k_centroids.transpose(1, 2)) * scale_value
        approximate_logits.add_(lengths_tensor.log()[None, None, :])
        approximate_logits.masked_fill_(selected[:, query_block, None, :], -torch.inf)

        probabilities = torch.softmax(torch.cat((exact_logits, approximate_logits), dim=-1), dim=-1)
        approximate_mass = probabilities[..., tokens:].sum(dim=-1, keepdim=True)
        output[:, start:end] = torch.bmm(probabilities, values) + correction[:, start:end] * approximate_mass

    return output


def _compile_target(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return rdna35_pisa_ck.forward(q, k, v, exact_blocks=1)


@unittest.skipUnless(GFX1151_AVAILABLE, "gfx1151 ROCm device required")
class TestForward(unittest.TestCase):
    def test_partial_matches_python_reference(self):
        q, k, v = _inputs(128, torch.bfloat16, seed=19)
        actual = rdna35_pisa_ck.forward(q, k, v, exact_blocks=1)
        expected = _pisa_reference(q, k, v, exact_blocks=1)
        torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)

    def test_partial_tail_matches_python_reference(self):
        q, k, v = _inputs(65, torch.bfloat16, seed=23)
        for sink_block in (0, -1):
            with self.subTest(sink_block=sink_block):
                actual = rdna35_pisa_ck.forward(q, k, v, exact_blocks=1, sink_block=sink_block)
                expected = _pisa_reference(q, k, v, exact_blocks=1, sink_block=sink_block)
                torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)

    def test_first_order_correction_centers_before_gemm(self):
        torch.manual_seed(37)
        tokens = 256
        q = (torch.randn((1, tokens, HEAD_DIM), device="cuda") * 16.0).to(torch.bfloat16)
        offsets = torch.randn((1, tokens // BLOCK_SIZE, 1, HEAD_DIM), device="cuda") * 64.0
        noise = torch.randn((1, tokens // BLOCK_SIZE, BLOCK_SIZE, HEAD_DIM), device="cuda") * 0.125
        k = (offsets + noise).flatten(1, 2).to(torch.bfloat16)
        v = torch.randn_like(k)
        actual = rdna35_pisa_ck.forward(q, k, v, exact_blocks=0)
        expected = _pisa_reference(q, k, v, exact_blocks=0)
        torch.testing.assert_close(actual.float(), expected, atol=6e-2, rtol=5e-2)

    def test_exact_matches_dense_sdpa(self):
        scale = 0.125
        q, k, v = _inputs(128, torch.bfloat16, seed=17)
        actual = rdna35_pisa_ck.forward(q, k, v, exact_blocks=2, scale=scale)
        expected = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], scale=scale).squeeze(1)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_invalid_scale_and_shape(self):
        q, k, v = _inputs(128, torch.bfloat16, seed=29)
        for scale in (0.0, -1.0, math.inf, math.nan, 1e100):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "scale"):
                rdna35_pisa_ck.forward(q, k, v, exact_blocks=1, scale=scale)

        with self.assertRaisesRegex(ValueError, r"matching \[BH,T,D\]"):
            rdna35_pisa_ck.forward(q[0], k[0], v[0], exact_blocks=1)
        with self.assertRaisesRegex(ValueError, r"matching \[BH,T,D\]"):
            rdna35_pisa_ck.forward(q, k[:, :-1].contiguous(), v, exact_blocks=1)
        with self.assertRaisesRegex(ValueError, "D=128"):
            rdna35_pisa_ck.forward(q[..., :64].contiguous(), k[..., :64].contiguous(), v[..., :64].contiguous(), exact_blocks=1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            rdna35_pisa_ck.forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), exact_blocks=1)
        fp16 = q.to(torch.float16)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            rdna35_pisa_ck.forward(fp16, fp16, fp16, exact_blocks=1)

    def test_large_finite_bfloat16_values_remain_finite(self):
        q = torch.zeros((1, 128, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.full_like(q, 2000.0)
        output = rdna35_pisa_ck.forward(q, k, v, exact_blocks=1)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, v, atol=0.0, rtol=0.0)

    def test_block_stats_average_bfloat16_extrema_without_overflow(self):
        maximum = torch.finfo(torch.bfloat16).max
        q = torch.full((1, 65, HEAD_DIM), maximum, device="cuda", dtype=torch.bfloat16)
        q_centroids, k_centroids, v_means = rdna35_pisa_ck._C.block_stats(q, q, q)

        for tensor in (q_centroids, k_centroids, v_means):
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertTrue((tensor > 0).all())

    def test_outer_torch_compile_fullgraph(self):
        q, k, v = _inputs(64, torch.bfloat16, seed=31)
        rdna35_pisa_ck.prepare()
        expected = _compile_target(q, k, v)
        compiled = torch.compile(_compile_target, backend="inductor", fullgraph=True)
        actual = compiled(q, k, v)
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    def test_spatial_pack_and_unpack_round_trip(self):
        batch, heads, tokens = 1, 2, rdna35_pisa_ck.SPATIAL_TOKENS
        raster_tokens = torch.arange(tokens, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        base = raster_tokens.view(batch, tokens, 1, 1).expand(batch, tokens, heads, HEAD_DIM).contiguous()
        bhtd = base.permute(0, 2, 1, 3)
        packed_q, packed_k, packed_v = rdna35_pisa_ck._C.pack_spatial_qkv(bhtd, bhtd, bhtd)

        expected_prefix = list(range(8)) + list(range(96, 104))
        self.assertEqual(packed_q[0, :16, 0].float().tolist(), expected_prefix)
        torch.testing.assert_close(packed_q, packed_k, atol=0.0, rtol=0.0)
        torch.testing.assert_close(packed_q, packed_v, atol=0.0, rtol=0.0)

        restored = rdna35_pisa_ck._C.unpack_spatial_output(packed_q, batch, heads)
        torch.testing.assert_close(restored, base.flatten(2), atol=0.0, rtol=0.0)

    def test_spatial_sparse_profile_is_finite_on_current_stream(self):
        batch, heads, tokens = 1, 2, rdna35_pisa_ck.SPATIAL_TOKENS
        current_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()

        for seed in (41, 47):
            torch.manual_seed(seed)
            q_bthd = torch.randn((batch, tokens, heads, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
            k_bthd = torch.randn_like(q_bthd)
            v_bthd = torch.randn_like(q_bthd)
            q, k, v = (tensor.permute(0, 2, 1, 3) for tensor in (q_bthd, k_bthd, v_bthd))

            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                output = rdna35_pisa_ck.forward_spatial_bhtd(q, k, v, exact_blocks=23)
            current_stream.wait_stream(stream)

            self.assertEqual(output.shape, (batch, tokens, heads * HEAD_DIM))
            self.assertTrue(torch.isfinite(output).all())

    def test_spatial_exact_matches_dense_sdpa(self):
        batch, heads, tokens = 1, 2, rdna35_pisa_ck.SPATIAL_TOKENS
        torch.manual_seed(43)
        q_bthd = torch.randn((batch, tokens, heads, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
        k_bthd = torch.randn_like(q_bthd)
        v_bthd = torch.randn_like(q_bthd)
        q, k, v = (tensor.permute(0, 2, 1, 3) for tensor in (q_bthd, k_bthd, v_bthd))

        actual = rdna35_pisa_ck.forward_spatial_bhtd(q, k, v, exact_blocks=tokens // BLOCK_SIZE)
        expected = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(batch, tokens, heads * HEAD_DIM)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

        with self.assertRaisesRegex(ValueError, "views of contiguous"):
            rdna35_pisa_ck.forward_spatial_bhtd(q.contiguous(), k.contiguous(), v.contiguous(), exact_blocks=1)

        for exact_blocks in (0, 22, 24, 143):
            with self.subTest(exact_blocks=exact_blocks), self.assertRaisesRegex(ValueError, "supports exact_blocks=23"):
                rdna35_pisa_ck.forward_spatial_bhtd(q, k, v, exact_blocks=exact_blocks)


if __name__ == "__main__":
    unittest.main()
