from __future__ import annotations

import pathlib
import sys
import unittest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rdna35_block_attention.pisa_runtime import PISARuntimeState


class PISARuntimeStateTests(unittest.TestCase):
    def test_expected_calls_uses_the_pisa_layer_range(self):
        self.assertEqual(PISARuntimeState.expected_calls(17), 408)
        self.assertEqual(PISARuntimeState.expected_calls(3, start_layer=2, total_layers=5), 9)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PISARuntimeState.expected_calls(-1)
        with self.assertRaisesRegex(ValueError, "start_layer"):
            PISARuntimeState.expected_calls(1, start_layer=5, total_layers=4)

    def test_record_tracks_hits_fallbacks_calls_and_shapes(self):
        state = PISARuntimeState(armed=True)
        state.record(layer=4, is_self_attention=True, shape=(1, 16, 9216, 128))
        state.record(is_self_attention=False, shape=(1, 16, 9216, 512), fallback_reason="call_is_not_explicitly_self_attention")
        state.record(is_self_attention=True, fallback_reason="token_shape_is_not_expected")

        self.assertTrue(state.executed)
        self.assertEqual(state.per_layer_hits, {4: 1})
        self.assertEqual(state.self_calls, 2)
        self.assertEqual(state.cross_calls, 1)
        self.assertEqual(state.shape_counts[(1, 16, 9216, 128)], 1)
        self.assertEqual(state.fallback_reasons["call_is_not_explicitly_self_attention"], 1)
        self.assertIn("hits=1/?", state.report())

    def test_verify_requires_every_expected_layer_for_every_forward(self):
        state = PISARuntimeState(armed=True)
        for _ in range(2):
            for layer in range(4, 28):
                state.record(layer=layer, is_self_attention=True)

        self.assertTrue(state.verify(2))
        self.assertTrue(state.verified)
        self.assertFalse(state.failed)
        self.assertIn("hits=48/48", state.report())

    def test_verify_marks_incomplete_accounting_as_failed(self):
        state = PISARuntimeState(armed=True)
        for layer in range(4, 27):
            state.record(layer=layer)

        self.assertFalse(state.verify(1))
        self.assertTrue(state.failed)
        self.assertFalse(state.verified)
        self.assertEqual(state.first_error, "PISA hits=23, expected=24")

    def test_first_error_is_preserved_and_reset_clears_sample_state(self):
        state = PISARuntimeState(armed=True)
        state.record(error=RuntimeError("first failure"))
        state.record(error="later failure")

        self.assertTrue(state.failed)
        self.assertEqual(state.first_error, "first failure")
        self.assertFalse(state.verify(0))

        state.reset(armed=False)
        self.assertFalse(state.armed)
        self.assertFalse(state.executed)
        self.assertFalse(state.verified)
        self.assertFalse(state.failed)
        self.assertEqual(state.per_layer_hits, {})
        self.assertEqual(state.fallback_reasons, {})
        self.assertIsNone(state.first_error)


if __name__ == "__main__":
    unittest.main()
