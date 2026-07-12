from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterable


PISA_RUNTIME_ATTACHMENT = "rdna35_pisa_runtime_state"


@dataclass
class PISARuntimeState:
    """Per-sample accounting for the model-local PISA override."""

    armed: bool = False
    executed: bool = False
    verified: bool = False
    failed: bool = False
    per_layer_hits: Counter[int] = field(default_factory=Counter)
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    self_calls: int = 0
    cross_calls: int = 0
    shape_counts: Counter[tuple[int, ...]] = field(default_factory=Counter)
    first_error: str | None = None
    _expected: int | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @staticmethod
    def expected_calls(actual_forwards: int, start_layer: int = 4, total_layers: int = 28) -> int:
        if actual_forwards < 0:
            raise ValueError("actual_forwards must be non-negative")
        if start_layer < 0 or total_layers < start_layer:
            raise ValueError("expected 0 <= start_layer <= total_layers")
        return actual_forwards * (total_layers - start_layer)

    def reset(self, *, armed: bool | None = None) -> None:
        with self._lock:
            if armed is not None:
                self.armed = armed
            self.executed = False
            self.verified = False
            self.failed = False
            self.per_layer_hits.clear()
            self.fallback_reasons.clear()
            self.self_calls = 0
            self.cross_calls = 0
            self.shape_counts.clear()
            self.first_error = None
            self._expected = None

    def record(
        self,
        *,
        layer: int | None = None,
        is_self_attention: bool | None = None,
        shape: Iterable[int] | None = None,
        fallback_reason: str | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Record one attention dispatch without retaining any tensor data."""
        with self._lock:
            if is_self_attention is True:
                self.self_calls += 1
            elif is_self_attention is False:
                self.cross_calls += 1

            if shape is not None:
                self.shape_counts[tuple(int(value) for value in shape)] += 1

            if fallback_reason is not None:
                self.fallback_reasons[str(fallback_reason)] += 1

            if error is not None:
                self.failed = True
                if self.first_error is None:
                    self.first_error = str(error)

            if layer is not None:
                self.executed = True
                self.per_layer_hits[int(layer)] += 1

    def verify(self, actual_forwards: int, start_layer: int = 4, total_layers: int = 28) -> bool:
        expected = self.expected_calls(actual_forwards, start_layer, total_layers)
        expected_layers = range(start_layer, total_layers)
        with self._lock:
            self._expected = expected
            if self.failed:
                self.verified = False
                return False

            unexpected_layers = sorted(set(self.per_layer_hits).difference(expected_layers))
            missing_layers = [layer for layer in expected_layers if self.per_layer_hits[layer] != actual_forwards]
            actual = sum(self.per_layer_hits.values())
            if unexpected_layers or missing_layers or actual != expected:
                self.failed = True
                self.verified = False
                if self.first_error is None:
                    self.first_error = f"PISA hits={actual}, expected={expected}"
                return False

            self.verified = True
            return True

    def report(self) -> str:
        with self._lock:
            hits = sum(self.per_layer_hits.values())
            expected = "?" if self._expected is None else str(self._expected)
            fallbacks = ",".join(f"{reason}:{count}" for reason, count in sorted(self.fallback_reasons.items())) or "-"
            shapes = ",".join(f"{shape}:{count}" for shape, count in sorted(self.shape_counts.items())) or "-"
            return (
                f"PISA armed={int(self.armed)} executed={int(self.executed)} verified={int(self.verified)} "
                f"failed={int(self.failed)} hits={hits}/{expected} self={self.self_calls} cross={self.cross_calls} "
                f"fallbacks={fallbacks} shapes={shapes} first_error={self.first_error or '-'}"
            )
