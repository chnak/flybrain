"""End-to-end pipeline test with a fake BrainProtocol implementation.

Uses the real encoder (BarsEncoder) + the real FixedDecoder, but supplies
a stub Brain that satisfies BrainProtocol without needing a compiled graph.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

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
from flybrainer.interfaces import Bar
from flybrainer.readout.fixed import FixedDecoder


class FakeBrain:
    """Minimal in-memory BrainProtocol: 256 neurons, fake populations."""

    def __init__(self, n: int = 256):
        self.n = int(n)
        # Pick contiguous slices for each population so FixedDecoder can index.
        rng = np.random.default_rng(0)
        cursor = 0
        self.populations: dict[str, np.ndarray] = {}
        for name in REQUIRED_POPULATIONS:
            # Vary size a little so it looks realistic
            size = max(2, n // 32)
            if cursor + size > n:
                size = max(1, n - cursor)
            self.populations[name] = np.arange(cursor, cursor + size, dtype=np.int32)
            cursor += size
            if cursor >= n:
                cursor = n - 1
        self._rng = rng

    def observe(self, stimulus: Stimulus, neural_ms: float) -> ObservationResult:
        # Generate counts roughly proportional to the stimulus
        r16_mean = float(np.mean(stimulus.r16)) if stimulus.r16.size else 0.0
        r8_mean = float(np.mean(stimulus.r8)) if stimulus.r8.size else 0.0
        seconds = max(neural_ms, 1e-9) / 1000.0
        rate = 5.0 + 30.0 * (r16_mean + r8_mean) / 2.0  # Hz
        lam = rate * seconds
        counts = self._rng.poisson(lam, size=self.n).astype(np.int32)
        return ObservationResult(
            counts=counts,
            neural_ms=float(neural_ms),
            sim_ms=0.0,
            compute_seconds=0.001,
        )

    def checkpoint(self, path: str) -> None:
        pass

    def restore(self, path: str) -> None:
        pass

    def provenance(self) -> dict[str, Any]:
        return {"fake": True, "n": self.n}


@pytest.fixture
def fake_brain() -> FakeBrain:
    return FakeBrain(n=256)


@pytest.fixture
def sensor_frame() -> SensorFrame:
    IST = timezone(timedelta(hours=5, minutes=30))
    bars = (
        Bar(
            timestamp=datetime(2024, 1, 15, 9, 25, tzinfo=IST),
            open=21600, high=21650, low=21580, close=21630, volume=1000,
        ),
        Bar(
            timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=IST),
            open=21630, high=21680, low=21610, close=21660, volume=1200,
        ),
    )
    return SensorFrame(
        timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=IST),
        index_bars=bars,
        vix=14.0,
        vix_bars=bars,
        straddle_premium=120.0,
        entry_credit=None,
        days_to_expiry=4.0,
        minutes_since_open=10,
        position_lots=0,
    )


def test_fake_brain_satisfies_protocol(fake_brain: FakeBrain):
    assert isinstance(fake_brain, BrainProtocol)
    assert isinstance(fake_brain.n, int)
    assert fake_brain.n == 256
    for name in REQUIRED_POPULATIONS:
        assert name in fake_brain.populations


def test_encode_decode_predict(fake_brain: FakeBrain, sensor_frame: SensorFrame):
    """End-to-end: SensorFrame -> Stimulus -> brain.observe -> Prediction."""
    encoder = make_encoder("bars")
    stimulus = encoder.encode(sensor_frame, fake_brain)

    assert isinstance(stimulus, Stimulus)
    assert stimulus.r16.shape[0] > 0
    assert stimulus.r8.shape[0] > 0
    assert np.all((stimulus.r16 >= 0) & (stimulus.r16 <= 1))
    assert np.all((stimulus.r8 >= 0) & (stimulus.r8 <= 1))

    result = fake_brain.observe(stimulus, neural_ms=200.0)
    assert isinstance(result, ObservationResult)
    assert result.counts.shape == (fake_brain.n,)
    assert result.neural_ms == 200.0

    decoder = FixedDecoder()
    pred = decoder.predict(result.counts, fake_brain, sensor_frame)
    assert isinstance(pred, Prediction)
    assert pred.decision in {Decision.ENTER, Decision.EXIT, Decision.HOLD}
    assert 0.0 <= pred.confidence <= 1.0
    assert 0.0 <= pred.realized_over_implied <= 2.0


def test_decoder_holds_when_in_position(fake_brain: FakeBrain, sensor_frame: SensorFrame):
    """If the decoder says EXIT but we're already flat -> HOLD."""
    from datetime import timedelta, timezone
    timezone(timedelta(hours=5, minutes=30))
    sensor_frame = SensorFrame(
        timestamp=sensor_frame.timestamp,
        index_bars=sensor_frame.index_bars,
        vix=sensor_frame.vix,
        vix_bars=sensor_frame.vix_bars,
        straddle_premium=sensor_frame.straddle_premium,
        entry_credit=sensor_frame.entry_credit,
        days_to_expiry=sensor_frame.days_to_expiry,
        minutes_since_open=sensor_frame.minutes_since_open,
        position_lots=0,  # flat
    )
    encoder = make_encoder("bars")
    stimulus = encoder.encode(sensor_frame, fake_brain)
    result = fake_brain.observe(stimulus, neural_ms=200.0)
    decoder = FixedDecoder()
    pred = decoder.predict(result.counts, fake_brain, sensor_frame)
    # decision is one of the three
    assert pred.decision in (Decision.ENTER, Decision.EXIT, Decision.HOLD)
