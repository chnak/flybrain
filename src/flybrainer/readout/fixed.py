"""Readout 1: the fixed "biological" decoder, kept as a control.

Mean right DNp20 rate minus mean left DNp20 rate over the observation, gated
by any DNpe017 spike, with a 2 Hz threshold: positive difference maps to
ENTER, negative to EXIT, otherwise HOLD. The reported realized-over-implied
value is 1 minus difference/10, clipped to [0, 2]; confidence comes from the
gate (0 without a gate spike).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from flybrainer.interfaces import Decision, Prediction, SensorFrame


class ColumnBinding:
    """Maps neuron ids to positions in a cached column subset."""

    def __init__(self, neuron_ids: np.ndarray):
        self.columns = np.ascontiguousarray(neuron_ids, dtype=np.int64)
        self._order = np.argsort(self.columns, kind="stable")
        self._sorted = self.columns[self._order]

    def positions(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        idx = np.searchsorted(self._sorted, ids)
        idx = np.clip(idx, 0, max(0, len(self._sorted) - 1))
        ok = len(self._sorted) > 0 and np.all(self._sorted[idx] == ids)
        if not ok:
            missing = ids[self._sorted[idx] != ids] if len(self._sorted) else ids
            raise KeyError(f"{len(missing)} neuron ids are not in the bound columns")
        return self._order[idx]


class FixedDecoder:
    name = "fixed"

    def __init__(
        self,
        neural_ms: float = 200.0,
        threshold_hz: float = 2.0,
        left: str = "DNp20_L",
        right: str = "DNp20_R",
        gate: str = "DNpe017",
        confidence_spikes: float = 5.0,
    ):
        self.neural_ms = float(neural_ms)
        self.threshold_hz = float(threshold_hz)
        self.left = left
        self.right = right
        self.gate = gate
        self.confidence_spikes = float(confidence_spikes)
        self._binding: ColumnBinding | None = None

    # -- configuration -------------------------------------------------------

    def params(self) -> dict:
        return {
            "name": self.name,
            "version": 1,
            "neural_ms": self.neural_ms,
            "threshold_hz": self.threshold_hz,
            "left": self.left,
            "right": self.right,
            "gate": self.gate,
            "confidence_spikes": self.confidence_spikes,
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.params(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def bind_columns(self, neuron_ids: np.ndarray) -> None:
        self._binding = ColumnBinding(neuron_ids)

    # -- decoding ------------------------------------------------------------

    def _select(self, counts: np.ndarray, brain, population: str) -> np.ndarray:
        ids = np.asarray(brain.populations[population], dtype=np.int64)
        counts = np.asarray(counts)
        if counts.shape[-1] == brain.n:
            return counts[..., ids]
        if self._binding is not None:
            return counts[..., self._binding.positions(ids)]
        raise ValueError("counts are not full length and no columns are bound; call bind_columns")

    def decode(self, counts: np.ndarray, brain, neural_ms: float | None = None) -> dict:
        ms = float(neural_ms or self.neural_ms)
        seconds = max(ms, 1e-9) / 1000.0
        left = self._select(counts, brain, self.left).astype(np.float64)
        right = self._select(counts, brain, self.right).astype(np.float64)
        gate = self._select(counts, brain, self.gate).astype(np.float64)
        left_hz = float(left.mean() / seconds) if left.size else 0.0
        right_hz = float(right.mean() / seconds) if right.size else 0.0
        gate_spikes = int(gate.sum()) if gate.size else 0
        diff = right_hz - left_hz
        if gate_spikes > 0 and diff > self.threshold_hz:
            side = Decision.ENTER
        elif gate_spikes > 0 and diff < -self.threshold_hz:
            side = Decision.EXIT
        else:
            side = Decision.HOLD
        return {
            "left_hz": left_hz,
            "right_hz": right_hz,
            "difference_hz": diff,
            "gate_spikes": gate_spikes,
            "side": side.value,
        }

    def realized_over_implied(self, difference_hz: float) -> float:
        return float(np.clip(1.0 - difference_hz / 10.0, 0.0, 2.0))

    def predict(self, counts: np.ndarray, brain, observation: SensorFrame | None = None) -> Prediction:
        d = self.decode(counts, brain)
        side = Decision(d["side"])
        in_position = bool(observation is not None and observation.position_lots != 0)
        decision = side
        if (side == Decision.ENTER and in_position) or (
            side == Decision.EXIT and not in_position
        ):
            decision = Decision.HOLD
        confidence = min(1.0, d["gate_spikes"] / self.confidence_spikes) if d["gate_spikes"] > 0 else 0.0
        details = dict(d)
        details.update({"threshold_hz": self.threshold_hz, "signal": side.value, "readout": self.name})
        return Prediction(
            realized_over_implied=self.realized_over_implied(d["difference_hz"]),
            confidence=float(confidence),
            decision=decision,
            details=details,
        )

    def predict_batch(self, counts: np.ndarray, brain, neural_ms: float | None = None) -> dict[str, np.ndarray]:
        """Vectorized decode over rows. Returns arrays: roi, signal (str), difference_hz, gate_spikes."""
        ms = float(neural_ms or self.neural_ms)
        seconds = max(ms, 1e-9) / 1000.0
        counts = np.asarray(counts)
        left = self._select(counts, brain, self.left).astype(np.float64)
        right = self._select(counts, brain, self.right).astype(np.float64)
        gate = self._select(counts, brain, self.gate).astype(np.float64)
        left_hz = left.mean(axis=-1) / seconds if left.shape[-1] else np.zeros(counts.shape[0])
        right_hz = right.mean(axis=-1) / seconds if right.shape[-1] else np.zeros(counts.shape[0])
        gate_spikes = gate.sum(axis=-1) if gate.shape[-1] else np.zeros(counts.shape[0])
        diff = right_hz - left_hz
        signal = np.full(counts.shape[0], Decision.HOLD.value, dtype=object)
        gated = gate_spikes > 0
        signal[gated & (diff > self.threshold_hz)] = Decision.ENTER.value
        signal[gated & (diff < -self.threshold_hz)] = Decision.EXIT.value
        roi = np.clip(1.0 - diff / 10.0, 0.0, 2.0)
        return {
            "roi": roi,
            "signal": signal,
            "difference_hz": diff,
            "gate_spikes": gate_spikes,
            "left_hz": left_hz,
            "right_hz": right_hz,
        }

    # -- persistence (stateless, kept for a uniform interface) ---------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "readout.json").write_text(json.dumps(self.params(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> FixedDecoder:
        params = json.loads((Path(directory) / "readout.json").read_text(encoding="utf-8"))
        params.pop("name", None)
        params.pop("version", None)
        return cls(**params)
