"""Head-to-head wake-pass benchmark: measure per-call time of MaleCNS observe."""
import sys
import os
import time
import numpy as np

sys.path.insert(0, ".")
from flybrainer import Brain
from flybrainer.interfaces import Stimulus
from scripts._bench_factories import make_toy_brain, make_malecns_brain


def bench(factory, name, n_iter=20, neural_ms=200.0):
    brain = factory()
    rng = np.random.default_rng(0)
    n16 = len(brain.populations["R1-R6"])
    n8 = len(brain.populations["R8"])
    stim = Stimulus(r16=rng.random(n16), r8=rng.random(n8))
    # JIT warm
    for _ in range(3):
        brain.reset()
        brain.observe(stim, neural_ms=neural_ms)
    # Measure
    samples = []
    for _ in range(n_iter):
        brain.reset()
        t0 = time.perf_counter()
        brain.observe(stim, neural_ms=neural_ms)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p10 = samples[len(samples) // 10]
    print(f"  {name:8s} neural_ms={neural_ms:5.0f}:  "
          f"p10={p10:6.1f}ms  p50={p50:6.1f}ms  "
          f"min={min(samples):6.1f}ms  max={max(samples):6.1f}ms")
    return p50


if __name__ == "__main__":
    print(f"Kernel version: {__import__('flybrainer').kernel.KERNEL_VERSION}")
    print(f"Numba threads:  {__import__('numba').config.NUMBA_NUM_THREADS}")
    print()
    print("=== Toy (100 neurons) ===")
    bench(make_toy_brain, "toy", n_iter=50)
    bench(make_toy_brain, "toy", n_iter=50, neural_ms=500.0)
    print()
    print("=== MaleCNS (166,700 neurons) ===")
    bench(make_malecns_brain, "male", n_iter=20)
    bench(make_malecns_brain, "male", n_iter=20, neural_ms=500.0)
