"""Profile the MaleCNS kernel to find the inner hot spot."""
import cProfile
import pstats
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from flybrainer import Brain
from flybrainer.encoders import make_encoder
from flybrainer.interfaces import Bar, SensorFrame
from scripts._bench_factories import make_malecns_brain, make_toy_brain


def make_frame():
    IST = timezone(timedelta(hours=5, minutes=30))
    bars = tuple(
        Bar(
            timestamp=datetime(2024, 1, 15, 9, 25 + i, tzinfo=IST),
            open=21600.0 + i * 5, high=21650.0 + i * 5,
            low=21580.0 + i * 5, close=21630.0 + i * 5, volume=1000 + i * 50,
        ) for i in range(3)
    )
    return SensorFrame(
        timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=IST),
        index_bars=bars, vix=14.2, vix_bars=bars,
        straddle_premium=120.5, entry_credit=None,
        days_to_expiry=4.0, minutes_since_open=10, position_lots=0,
    )


def profile_one(label: str, brain_factory, repeats: int = 5) -> None:
    print(f"\n========== profile {label} ==========")
    brain = brain_factory()
    encoder = make_encoder("bars", eye_map=brain.eye_map())
    stim = encoder.encode(make_frame(), brain)
    # JIT warm
    brain.observe(stim, neural_ms=200.0)
    brain.reset()

    # Profile one observe
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(repeats):
        brain.reset()
        brain.observe(stim, neural_ms=200.0)
    pr.disable()

    stats = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
    print(f"\n-- top 20 by cumulative time ({repeats} runs) --")
    stats.print_stats(20)

    stats = pstats.Stats(pr).strip_dirs().sort_stats("tottime")
    print(f"\n-- top 20 by self time --")
    stats.print_stats(20)


def main():
    profile_one("100-neuron toy", make_toy_brain, repeats=200)
    profile_one("MaleCNS (166,700)", make_malecns_brain, repeats=5)


if __name__ == "__main__":
    main()
