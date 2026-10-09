"""Benchmark ensemble decision: serial vs 2/4/8/16 workers, 100-neuron vs MaleCNS."""
import time
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from flybrainer import Brain
from flybrainer.encoders import make_encoder
from flybrainer.interfaces import Bar, SensorFrame
from flybrainer.ensemble import BatchBrain
from scripts._bench_factories import make_toy_brain, make_malecns_brain


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


def make_varied_stimuli(brain, encoder, base_sf, n: int) -> list:
    stimuli = []
    for i in range(n):
        s = SensorFrame(
            timestamp=base_sf.timestamp,
            index_bars=base_sf.index_bars,
            vix=10.0 + (10.0 * i / max(1, n - 1)),
            vix_bars=base_sf.vix_bars,
            straddle_premium=100.0 + i,
            entry_credit=None,
            days_to_expiry=4.0,
            minutes_since_open=10,
            position_lots=0,
        )
        stimuli.append(encoder.encode(s, brain))
    return stimuli


def bench_100_neuron(workers_list, repeats=30):
    print("\n========== 100-neuron toy graph ==========", flush=True)

    # Warm in main process
    brain = make_toy_brain()
    encoder = make_encoder("bars")
    sf = make_frame()
    stim = encoder.encode(sf, brain)
    brain.observe(stim, neural_ms=200.0)  # JIT
    print("JIT done (main process)", flush=True)

    stimuli = make_varied_stimuli(brain, encoder, sf, repeats)

    # Serial baseline
    for _ in range(3):
        brain.reset()
        brain.observe(stimuli[0], neural_ms=200.0)
    t0 = time.perf_counter()
    for s in stimuli:
        brain.reset()
        brain.observe(s, neural_ms=200.0)
    t_serial = (time.perf_counter() - t0) * 1000
    per_call_serial = t_serial / repeats
    print(f"  serial:  {per_call_serial:7.2f} ms/call  ({repeats} calls in {t_serial:.0f} ms)", flush=True)

    for w in workers_list:
        if w > (os.cpu_count() or 1):
            continue
        try:
            with BatchBrain(num_workers=w, brain_factory=make_toy_brain) as ens:
                # Warm pool (worker JIT + first import)
                warm = make_varied_stimuli(brain, encoder, sf, w)
                ens.observe_batch(warm, neural_ms=200.0)
                # Real
                t0 = time.perf_counter()
                results = ens.observe_batch(stimuli, neural_ms=200.0)
                t_par = (time.perf_counter() - t0) * 1000
                per_call = t_par / repeats
                speedup = per_call_serial / per_call
                wall_p50 = float(np.percentile(ens.last_wall_times, 50) * 1000)
                wall_p99 = float(np.percentile(ens.last_wall_times, 99) * 1000)
                print(
                    f"  {w:2d}-proc:  {per_call:7.2f} ms/call  "
                    f"(wall_p50={wall_p50:6.1f} ms, p99={wall_p99:6.1f} ms, speedup {speedup:4.2f}x)",
                    flush=True,
                )
        except Exception as e:
            print(f"  {w:2d}-proc:  FAILED: {e}", flush=True)


def bench_malecns(workers_list, repeats=8):
    print("\n========== MaleCNS v1.0 (166,700 neurons) ==========", flush=True)

    # Warm in main process
    brain = make_malecns_brain()
    encoder = make_encoder("bars", eye_map=brain.eye_map())
    sf = make_frame()
    stim = encoder.encode(sf, brain)
    brain.observe(stim, neural_ms=200.0)  # JIT
    print("JIT done (main process)", flush=True)

    stimuli = make_varied_stimuli(brain, encoder, sf, repeats)

    # Serial baseline
    for _ in range(2):
        brain.reset()
        brain.observe(stimuli[0], neural_ms=200.0)
    t0 = time.perf_counter()
    for s in stimuli:
        brain.reset()
        brain.observe(s, neural_ms=200.0)
    t_serial = (time.perf_counter() - t0) * 1000
    per_call_serial = t_serial / repeats
    print(f"  serial:  {per_call_serial:7.0f} ms/call  ({repeats} calls in {t_serial:.0f} ms)", flush=True)

    for w in workers_list:
        if w > (os.cpu_count() or 1):
            continue
        try:
            with BatchBrain(num_workers=w, brain_factory=make_malecns_brain) as ens:
                warm = make_varied_stimuli(brain, encoder, sf, w)
                ens.observe_batch(warm, neural_ms=200.0)
                t0 = time.perf_counter()
                results = ens.observe_batch(stimuli, neural_ms=200.0)
                t_par = (time.perf_counter() - t0) * 1000
                per_call = t_par / repeats
                speedup = per_call_serial / per_call
                wall_p50 = float(np.percentile(ens.last_wall_times, 50) * 1000)
                wall_p99 = float(np.percentile(ens.last_wall_times, 99) * 1000)
                print(
                    f"  {w:2d}-proc:  {per_call:7.0f} ms/call  "
                    f"(wall_p50={wall_p50:6.0f} ms, p99={wall_p99:6.0f} ms, speedup {speedup:4.2f}x)",
                    flush=True,
                )
        except Exception as e:
            print(f"  {w:2d}-proc:  FAILED: {e}", flush=True)


def main():
    print(f"CPU count: {os.cpu_count()}", flush=True)
    workers = [1, 2, 4, 8, 16]
    if os.cpu_count():
        workers = [w for w in workers if w <= os.cpu_count()]

    bench_100_neuron(workers, repeats=30)
    bench_malecns(workers, repeats=8)


if __name__ == "__main__":
    main()
