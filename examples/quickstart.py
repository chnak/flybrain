"""End-to-end flybrainer example using a stub Brain (no MaleCNS download).

This file is runnable as:

    python examples/quickstart.py

It demonstrates the full pipeline:
    SensorFrame --[Encoder]--> Stimulus --[Brain]--> counts --[Decoder]--> Prediction

The "Brain" here is a 256-neuron stub that satisfies ``BrainProtocol`` and
produces deterministic-ish Poisson counts.  Replace it with the real
``Brain.load_graph(...)`` once you have compiled a graph from the
MaleCNS dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from flybrainer import (
    REQUIRED_POPULATIONS,
    BrainProtocol,
    Decision,
    ObservationResult,
    Prediction,
    SensorFrame,
    Stimulus,
)
from flybrainer.encoders import make_encoder
from flybrainer.readout.fixed import FixedDecoder

# ---------------------------------------------------------------------------
# Stub implementation of BrainProtocol.  Replace with:
#     from flybrainer import Brain, load_graph
#     brain = Brain(graph_path="connectome/...")
# once you have a real compiled graph.
# ---------------------------------------------------------------------------


@dataclass
class StubBrain(BrainProtocol):
    """Tiny stub that satisfies BrainProtocol without a real connectome."""

    n_neurons: int = 256
    seed: int = 42

    def __post_init__(self) -> None:
        self.n: int = int(self.n_neurons)
        self._rng = np.random.default_rng(self.seed)
        self.populations: dict[str, np.ndarray] = {}
        # Carve the neuron range into contiguous slices, one per population.
        cursor = 0
        for name in REQUIRED_POPULATIONS:
            size = max(8, self.n // 32)
            if cursor + size > self.n:
                size = max(1, self.n - cursor)
            self.populations[name] = np.arange(cursor, cursor + size, dtype=np.int32)
            cursor += size
            if cursor >= self.n:
                cursor = self.n - 1

    def observe(self, stimulus: Stimulus, neural_ms: float) -> ObservationResult:
        """Fake 'spike' counts proportional to the stimulus drive."""
        r16 = float(np.mean(stimulus.r16)) if stimulus.r16.size else 0.0
        r8 = float(np.mean(stimulus.r8)) if stimulus.r8.size else 0.0
        drive = 0.5 * (r16 + r8)
        seconds = max(neural_ms, 1e-9) / 1000.0
        lam = (2.0 + 25.0 * drive) * seconds
        counts = self._rng.poisson(lam, size=self.n).astype(np.int32)
        return ObservationResult(
            counts=counts,
            neural_ms=float(neural_ms),
            sim_ms=0.0,
            compute_seconds=0.0,
        )

    def checkpoint(self, path: str) -> None:
        # Stub: nothing to do
        pass

    def restore(self, path: str) -> None:
        pass

    def provenance(self) -> dict[str, Any]:
        return {"stub": True, "n": self.n}


# ---------------------------------------------------------------------------
# Build a synthetic SensorFrame (past-only, frozen).
# ---------------------------------------------------------------------------


def make_sensor_frame() -> SensorFrame:
    IST = timezone(timedelta(hours=5, minutes=30))
    bars = tuple(
        Bar(  # type: ignore[name-defined]
            timestamp=datetime(2024, 1, 15, 9, 25 + i, tzinfo=IST),
            open=21600 + i * 5,
            high=21650 + i * 5,
            low=21580 + i * 5,
            close=21630 + i * 5,
            volume=1000 + i * 50,
        )
        for i in range(3)
    )
    return SensorFrame(
        timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=IST),
        index_bars=bars,
        vix=14.2,
        vix_bars=bars,
        straddle_premium=120.5,
        entry_credit=None,
        days_to_expiry=4.0,
        minutes_since_open=10,
        position_lots=0,
    )


# Need Bar from interfaces to build the sensor frame
from flybrainer.interfaces import Bar  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> Prediction:
    # ---- 1.  Set up the brain ----------------------------------------------
    brain = StubBrain(n_neurons=256)

    # ---- 2.  Encode the market snapshot into photoreceptor currents --------
    encoder = make_encoder("bars")
    sf = make_sensor_frame()
    stimulus = encoder.encode(sf, brain)
    print(f"Stimulus: r16={stimulus.r16.shape} r8={stimulus.r8.shape}")

    # ---- 3.  Run 200 ms of neural dynamics --------------------------------
    result = brain.observe(stimulus, neural_ms=200.0)
    total_spikes = int(result.counts.sum())
    print(f"Observed: {total_spikes} spikes across {brain.n} neurons "
          f"in {result.neural_ms:.0f} ms")

    # ---- 4.  Decode into a Decision ---------------------------------------
    decoder = FixedDecoder(neural_ms=200.0, threshold_hz=2.0)
    pred = decoder.predict(result.counts, brain, sf)
    print(f"Prediction: decision={pred.decision.value} "
          f"roi={pred.realized_over_implied:.3f} "
          f"confidence={pred.confidence:.3f}")

    # Sanity check on the protocol types
    assert isinstance(pred, Prediction)
    assert pred.decision in {Decision.ENTER, Decision.EXIT, Decision.HOLD}

    return pred


if __name__ == "__main__":
    main()
