"""End-to-end flybrain example using the REAL MaleCNS v1.0 connectome.

This is the same pipeline as examples/real_brain.py but loads the FULL
166,700-neuron MaleCNS v1.0 connectome compiled from the 1 GB feather files,
not a synthetic 100-neuron toy graph.

Run:
    python examples/real_malecns_demo.py

Requirements:
    - MaleCNS v1.0 feather files in  ~/.flybrain/malecns/
    - Compiled graph.npz at      ~/.flybrain/graph.npz
    (Both are produced by flybrain.connectome.compile.compile_graph().)

What this demo does
-------------------
1. Compiles the 3 feather files into graph.npz if not already done.
2. Loads the full 166,700-neuron Brain.
3. Synthesizes a fake NIFTY-style 5-minute-bar SensorFrame.
4. Encodes it with BarsEncoder into a retinal Stimulus.
5. Runs 500 ms of LIF dynamics through the Numba JIT kernel.
6. Reads out per-population firing rates, runs FixedDecoder, prints kernel
   parameters and graph hashes, does checkpoint / restore round-trip.
"""
from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from flybrain import (
    REQUIRED_POPULATIONS,
    Brain,
    SensorFrame,
)
from flybrain.interfaces import Bar
from flybrain.connectome.compile import compile_graph
from flybrain.encoders import make_encoder
from flybrain.paths import PATHS
from flybrain.readout.fixed import FixedDecoder

print("=" * 70)
print("REAL MaleCNS v1.0 Brain demo")
print("=" * 70)

# -----------------------------------------------------------------
# 1. Compile from sources if graph.npz doesn't exist yet.
# -----------------------------------------------------------------
if not PATHS.graph.exists():
    print("\n[graph] not found, compiling from sources ...")
    t0 = time.perf_counter()
    compile_graph()
    print(f"[graph] compiled in {time.perf_counter() - t0:.1f} s")
print(f"[graph] {PATHS.graph}")

# -----------------------------------------------------------------
# 2. Load the Brain with the REAL MaleCNS graph (166,700 neurons,
#    25,582,938 edges, 124M contacts).
# -----------------------------------------------------------------
print("\n[brain] loading MaleCNS v1.0 ...")
t0 = time.perf_counter()
brain = Brain(plastic=False)
load_s = time.perf_counter() - t0

provenance = brain.provenance()
pop_sizes = provenance["population_sizes"]
print(f"[brain] loaded in {load_s:.2f} s")
print(f"[brain] n_neurons = {provenance['n']:,}")
print(f"[brain] n_edges   = {provenance['edges']:,}")
print(f"[brain] n_populations = {len(pop_sizes)}")
for k in REQUIRED_POPULATIONS:
    print(f"[brain]   {k:14} = {pop_sizes.get(k, 0):,}")

# -----------------------------------------------------------------
# 3. Build a synthetic NIFTY-style SensorFrame (60 5-min bars).
# -----------------------------------------------------------------
N_BARS = 60
base_price = 24_500.0
rng = np.random.default_rng(0)
log_rets = rng.normal(0.0, 0.0008, size=N_BARS)  # ~ 1.3% daily vol
closes = base_price * np.exp(np.cumsum(log_rets))
ts0 = datetime.now(timezone.utc) - timedelta(minutes=5 * N_BARS)
bars = tuple(
    Bar(
        timestamp=ts0 + timedelta(minutes=5 * i),
        open=closes[i - 1] if i > 0 else closes[0],
        high=closes[i] * (1.0 + abs(rng.normal(0, 0.0005))),
        low=closes[i] * (1.0 - abs(rng.normal(0, 0.0005))),
        close=closes[i],
        volume=1.0,
    )
    for i in range(N_BARS)
)
vix_bars = tuple(
    Bar(
        timestamp=ts0 + timedelta(minutes=5 * i),
        open=14.0, high=14.5, low=13.5, close=14.0 + rng.normal(0, 0.3), volume=0.0,
    )
    for i in range(N_BARS)
)
frame = SensorFrame(
    timestamp=ts0 + timedelta(minutes=5 * N_BARS),
    index_bars=bars,
    vix=14.2,
    vix_bars=vix_bars,
    straddle_premium=80.0,
    entry_credit=None,
    days_to_expiry=5.0,
    minutes_since_open=60,
    position_lots=0,
)
print(f"\n[frame] {len(bars)} x 5min bars, last close={closes[-1]:.2f}, VIX=14.2")

# -----------------------------------------------------------------
# 4. Encode to retinal Stimulus.
# -----------------------------------------------------------------
encoder = make_encoder("bars", eye_map=brain.eye_map())
stim = encoder.encode(frame, brain=brain)
print(f"[stim]  r16.shape={stim.r16.shape}  r8.shape={stim.r8.shape}")
print(f"[stim]  r16.mean={stim.r16.mean():.3f}  r8.mean={stim.r8.mean():.3f}")

# -----------------------------------------------------------------
# 5. Observe 500 ms of neural dynamics through the JIT kernel.
# -----------------------------------------------------------------
print("\n[observe] running 500 ms of LIF dynamics on 166,700 neurons ...")
t0 = time.perf_counter()
obs = brain.observe(stim, neural_ms=500.0)
wall_s = time.perf_counter() - t0
n_spikes = int(obs.counts.sum())
print(f"[observe] {n_spikes:,} spikes across {brain.n:,} neurons")
print(f"[observe] wall time = {wall_s*1000:.1f} ms   sim time = 500 ms")
print(f"[observe] speed-up  = {500.0 / (wall_s*1000):.1f}x realtime")

# -----------------------------------------------------------------
# 6. Per-population firing rates.
# -----------------------------------------------------------------
print("\n[rates]")
rates = brain.population_rates(obs.counts, obs.neural_ms)
for k in REQUIRED_POPULATIONS:
    hz = rates.get(k, 0.0)
    bar_len = min(60, max(0, int(round(hz))))
    print(f"  {k:14} {hz:8.3f} Hz  {'#' * bar_len}")

# -----------------------------------------------------------------
# 7. Decoder prediction.
# -----------------------------------------------------------------
print("\n[decode]")
decoder = FixedDecoder()
prediction = decoder.predict(obs.counts, brain=brain, observation=frame)
print(f"  decision     = {prediction.decision.name}")
print(f"  confidence   = {prediction.confidence:.3f}")
print(f"  roi          = {prediction.realized_over_implied:.3f}")
signal = prediction.details.get("signal", "?")
diff = prediction.details.get("difference_hz")
print(f"  signal       = {signal}")
print(f"  diff_hz      = {diff if diff is not None else 'n/a'}")

# -----------------------------------------------------------------
# 8. Introspection (kernel + graph hashes).
# -----------------------------------------------------------------
print("\n[introspect]")
p = brain.parameters()
print(f"  kernel    = dt_ms={p['dt_ms']}  tau_m_ms={p['tau_m_ms']}  "
      f"v_thresh_mv={p['v_thresh_mv']}")
print(f"  decoder   = bin_ms={p['bin_ms']}  half_saturation={p['half_saturation']}  "
      f"plastic={p['plastic']}")
hashes = brain.graph_hashes()
sample = next(iter(hashes.items()))
print(f"  graph     = {len(hashes)} arrays hashed, e.g. "
      f"{sample[0]}={sample[1][:16]}...")

# -----------------------------------------------------------------
# 9. Save + restore a checkpoint.
# -----------------------------------------------------------------
print("\n[ckpt]")
with tempfile.TemporaryDirectory() as tmpdir:
    from pathlib import Path
    ckpt = Path(tmpdir) / "brain.ckpt.npz"
    brain.checkpoint(ckpt)
    size = ckpt.stat().st_size

    brain.reset()
    brain.restore(ckpt)
    obs2 = brain.observe(stim, neural_ms=500.0)
    n_post = int(obs2.counts.sum())
    print(f"  wrote + restored -> {size:,} bytes")
    print(f"  pre-checkpoint spikes  = {n_spikes:,}")
    print(f"  post-restore   spikes  = {n_post:,}")
    print(f"  deterministic         = {n_spikes == n_post}")

print("\n" + "=" * 70)
print("DONE: real MaleCNS v1.0 connectome, end-to-end")
print("=" * 70)