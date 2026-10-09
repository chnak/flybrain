"""Profile the real MaleCNS pipeline."""
import time
import sys
import cProfile
import pstats
import io
import numpy as np

sys.path.insert(0, ".")
from flybrainer import Brain
from flybrainer.encoders import make_encoder
from flybrainer.paths import PATHS
from flybrainer.readout.fixed import FixedDecoder
from flybrainer.interfaces import Bar, SensorFrame
from datetime import datetime, timedelta, timezone


def make_frame():
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


def main():
    print("--- Load Brain (MaleCNS) ---", flush=True)
    t0 = time.perf_counter()
    brain = Brain(plastic=False)
    print(f"load: {time.perf_counter() - t0:.2f}s, n={brain.n:,}, edges={brain.edges:,}", flush=True)

    # Warm JIT
    print("--- JIT warmup ---", flush=True)
    sf = make_frame()
    encoder = make_encoder("bars", eye_map=brain.eye_map())
    stim = encoder.encode(sf, brain)
    brain.observe(stim, neural_ms=200.0)
    print("JIT done", flush=True)

    decoder = FixedDecoder(neural_ms=200.0)

    # Time single observe
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        brain.reset()
        result = brain.observe(stim, neural_ms=200.0)
    t_200 = (time.perf_counter() - t0) / N * 1000
    print(f"\n[MaleCNS] observe(200ms): {t_200:.2f} ms/iter ({200.0 / t_200:.1f}x realtime)", flush=True)

    t0 = time.perf_counter()
    for _ in range(N):
        brain.reset()
        result = brain.observe(stim, neural_ms=500.0)
    t_500 = (time.perf_counter() - t0) / N * 1000
    print(f"[MaleCNS] observe(500ms): {t_500:.2f} ms/iter ({500.0 / t_500:.1f}x realtime)", flush=True)

    # Encoder
    t0 = time.perf_counter()
    for _ in range(100):
        stim = encoder.encode(sf, brain)
    t_enc = (time.perf_counter() - t0) / 100 * 1000
    print(f"[MaleCNS] encoder:        {t_enc:.4f} ms/iter", flush=True)

    # Decoder
    t0 = time.perf_counter()
    for _ in range(N):
        pred = decoder.predict(result.counts, brain, sf)
    t_dec = (time.perf_counter() - t0) / N * 1000
    print(f"[MaleCNS] decoder:        {t_dec:.4f} ms/iter", flush=True)

    # cProfile
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10):
        brain.reset()
        brain.observe(stim, neural_ms=200.0)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print("\n--- cProfile: top 20 by cumtime (10x observe(200ms)) ---")
    print(s.getvalue())


if __name__ == "__main__":
    main()
