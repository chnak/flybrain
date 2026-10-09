"""Profile the decision pipeline: encoder, observe, decoder."""
import time
import sys
import cProfile
import pstats
import io

import numpy as np

sys.path.insert(0, ".")
from examples.real_brain import build_synthetic_graph, make_sensor_frame
from flybrainer import Brain
from flybrainer.encoders import make_encoder
from flybrainer.readout.fixed import FixedDecoder


def main():
    # ---- JIT warmup ----
    print("--- JIT warmup ---", flush=True)
    g0 = build_synthetic_graph()
    b0 = Brain(graph_path="<x>", graph=g0, check_counts=False)
    sf0 = make_sensor_frame()
    enc0 = make_encoder("bars")
    s0 = enc0.encode(sf0, b0)
    b0.observe(s0, neural_ms=200.0)
    print("JIT done", flush=True)

    # ---- Real measurement ----
    graph = build_synthetic_graph()
    brain = Brain(graph_path="<x>", graph=graph, check_counts=False, plastic=False)
    encoder = make_encoder("bars")
    sf = make_sensor_frame()
    decoder = FixedDecoder(neural_ms=200.0)

    stim = encoder.encode(sf, brain)
    print(f"\nbrain.n = {brain.n}, edges = {brain.edges}, plastic = {brain.plastic}")

    # ---- Time encode ----
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        stim = encoder.encode(sf, brain)
    t_enc = (time.perf_counter() - t0) / N * 1000
    print(f"\n[100-neuron]  encoder:        {t_enc:.4f} ms/iter (mean of {N})")

    # ---- Time single observe ----
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        brain.reset()
        result = brain.observe(stim, neural_ms=200.0)
    t_single = (time.perf_counter() - t0) / N * 1000
    print(f"[100-neuron]  observe(200ms): {t_single:.4f} ms/iter (mean of {N})")
    print(f"[100-neuron]  decoder:        ", end="")
    t0 = time.perf_counter()
    for _ in range(N):
        pred = decoder.predict(result.counts, brain, sf)
    t_dec = (time.perf_counter() - t0) / N * 1000
    print(f"{t_dec:.4f} ms/iter (mean of {N})")

    # ---- Time plastic observe ----
    brain_p = Brain(graph_path="<x>", graph=graph, check_counts=False, plastic=True)
    brain_p.observe(stim, neural_ms=200.0)  # warm
    t0 = time.perf_counter()
    for _ in range(N):
        brain_p.reset()
        result = brain_p.observe(stim, neural_ms=200.0)
    t_plastic = (time.perf_counter() - t0) / N * 1000
    print(f"[100-neuron]  plastic observe:{t_plastic:.4f} ms/iter (mean of {N})")

    # ---- Time 500ms observe (10x neural) ----
    t0 = time.perf_counter()
    for _ in range(N):
        brain.reset()
        result = brain.observe(stim, neural_ms=500.0)
    t_500 = (time.perf_counter() - t0) / N * 1000
    print(f"[100-neuron]  observe(500ms): {t_500:.4f} ms/iter (mean of {N})")

    # ---- cProfile breakdown of one observe ----
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        brain.reset()
        brain.observe(stim, neural_ms=200.0)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print("\n--- cProfile: top 25 by cumtime (20 observe(200ms) calls) ---")
    print(s.getvalue())


if __name__ == "__main__":
    main()
