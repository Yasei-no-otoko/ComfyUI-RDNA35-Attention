from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_NAME = "rdna35_attention_test_package"

if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)

from rdna35_block_attention.pisa_runtime import PISA_RUNTIME_ATTACHMENT, PISARuntimeState


RDNA35PISARuntimeReport = sys.modules[f"{PACKAGE_NAME}.nodes"].RDNA35PISARuntimeReport


class DummyModel:
    def __init__(self, state):
        self.state = state

    def get_attachment(self, name):
        if name == PISA_RUNTIME_ATTACHMENT:
            return self.state
        return None


class PISARuntimeReportTests(unittest.TestCase):
    def test_success_returns_the_original_latent_and_verified_report(self):
        state = PISARuntimeState(armed=True)
        for _ in range(2):
            for layer in range(4, 28):
                state.record(layer=layer, is_self_attention=True)
        latent = {"samples": object()}

        output = RDNA35PISARuntimeReport().run(DummyModel(state), latent)
        output_latent, report = output["result"]

        self.assertIs(output_latent, latent)
        self.assertEqual(output["ui"]["text"], [report])
        self.assertTrue(state.verified)
        self.assertIn("verified=1", report)
        self.assertIn("hits=48/48", report)

    def test_zero_hits_are_rejected_as_an_invalid_benchmark(self):
        state = PISARuntimeState(armed=True)

        with self.assertRaisesRegex(RuntimeError, "INVALID BENCHMARK: PISA backend was not executed"):
            RDNA35PISARuntimeReport().run(DummyModel(state), {"samples": object()})

    def test_incomplete_layer_accounting_is_rejected(self):
        state = PISARuntimeState(armed=True)
        for layer in range(4, 27):
            state.record(layer=layer, is_self_attention=True)

        with self.assertRaisesRegex(RuntimeError, "Incomplete PISA layer accounting"):
            RDNA35PISARuntimeReport().run(DummyModel(state), {"samples": object()})


if __name__ == "__main__":
    unittest.main()
