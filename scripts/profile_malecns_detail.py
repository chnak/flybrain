"""Profile MaleCNS with cProfile breakdown."""
import time
import sys
import cProfile
import pstats
import io
import numpy as np

sys.path.insert(0, ".")
from flybrainer import Brain
from flybrainer.encoders import make_encoder
from flybrainer.interfaces import Bar, SensorFrame
from datetime import datetime, timedelta, timezone


def main():
    IST = timezone(timedelta(hours=5, minutes=30))
    bars = tuple(
        Bar(
            timestamp=datetime(2024, 1, 15, 9, 25 + i, tzinfo=IST),
            open=21600.0 + i * 5,
            high=21650.0 + i * 5,
            low=21580.0 + i * 5,
            close=21630.0 + i * 5,
            volume=1000 + i * 50,
        )
        for i in range(3)
    )
    sf = SensorFrame(
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

    brain = Brain(plastic=False)
    encoder = make_encoder("bars", eye_map=brain.eye_map())
    stim = encoder.encode(sf, brain)
    brain.observe(stim, neural_ms=200.0)  # warm

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        brain.reset()
        brain.observe(stim, neural_ms=200.0)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue())

    print("\n--- tottime breakdown ---")
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(15)
    print(s2.getvalue())

    # Per-bin timing inside observe
    print("\n--- Per-call time at observation start vs end ---")
    times = []
    for i in range(5):
        brain.reset()
        t0 = time.perf_counter()
        brain.observe(stim, neural_ms=200.0)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"5 runs: {[f'{t:.0f}ms' for t in times]}, mean={np.mean(times):.0f}ms")


if __name__ == "__main__":
    main()
