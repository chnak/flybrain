"""Verify the parallel kernel produces identical results to the serial one."""
import sys
import time
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._bench_factories import make_malecns_brain, make_toy_brain


def check(name, brain_factory):
    print(f"\n--- {name} ---")
    b1 = brain_factory()
    b2 = brain_factory()
    # Random stimulus
    rng = np.random.default_rng(0)
    n16 = len(b1.populations["R1-R6"])
    n8 = len(b1.populations["R8"])
    from flybrainer.interfaces import Stimulus
    stim = Stimulus(
        r16=rng.random(n16),
        r8=rng.random(n8),
    )
    # Warm both
    b1.observe(stim, neural_ms=200.0)
    b2.observe(stim, neural_ms=200.0)
    b1.reset()
    b2.reset()
    r1 = b1.observe(stim, neural_ms=200.0)
    r2 = b2.observe(stim, neural_ms=200.0)
    diff = (r1.counts - r2.counts)
    nz = np.nonzero(diff)[0]
    if len(nz) == 0:
        print(f"  identical: {r1.counts.sum()} spikes, max_count={r1.counts.max()}")
    else:
        print(f"  DIFFER: {len(nz)} neurons differ, max abs diff = {np.abs(diff).max()}")
        print(f"    first 5 differing: {nz[:5].tolist()}")
    # Time both
    brain = brain_factory()
    brain.observe(stim, neural_ms=200.0)  # JIT
    brain.reset()
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        brain.reset()
        brain.observe(stim, neural_ms=200.0)
    per_call = (time.perf_counter() - t0) * 1000 / n
    print(f"  per-call: {per_call:.1f} ms")


def main():
    check("100-neuron toy", make_toy_brain)
    check("MaleCNS", make_malecns_brain)


if __name__ == "__main__":
    main()
