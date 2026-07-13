import types
import unittest

import torch

from comfy.ldm.lightricks.model import LTXBaseModel


class LTXPISAContractTests(unittest.TestCase):
    def _run_forward(self, token_count, transformer_options):
        original = torch.zeros(1, 4, 2, 64, 64)
        tokens = torch.zeros(1, token_count, 64)
        observed = {}
        model = types.SimpleNamespace(
            _process_input=lambda x, keyframe_idxs, denoise_mask, **kwargs: (tokens, None, {"orig_shape": list(original.shape)}),
            _prepare_timestep=lambda timestep, batch_size, dtype, **kwargs: (timestep, timestep, None),
            _prepare_context=lambda context, batch_size, x, attention_mask: (context, attention_mask),
            _prepare_attention_mask=lambda attention_mask, dtype: attention_mask,
            _prepare_positional_embeddings=lambda pixel_coords, frame_rate, dtype: None,
            _build_guide_self_attention_mask=lambda x, options, merged_args: None,
            _process_transformer_blocks=lambda x, context, attention_mask, timestep, pe, transformer_options, **kwargs: observed.setdefault("grid", transformer_options.get("attention_token_grid")) or x,
            _process_output=lambda x, embedded_timestep, keyframe_idxs, **kwargs: x,
        )

        LTXBaseModel._forward(model, original, torch.zeros(1), torch.zeros(1, 1, 64), None, transformer_options=transformer_options)
        return observed.get("grid")

    def test_unfiltered_ltx_video_exposes_token_grid(self):
        self.assertEqual(self._run_forward(2 * 64 * 64, {}), (2, 64, 64))

    def test_filtered_ltx_video_clears_stale_token_grid(self):
        options = {"attention_token_grid": (9, 9, 9)}
        self.assertIsNone(self._run_forward(1024, options))
        self.assertNotIn("attention_token_grid", options)


if __name__ == "__main__":
    unittest.main()
