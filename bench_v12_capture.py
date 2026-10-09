"""Capture v1.2 (or current) spike counts for later bit-identical comparison."""
import sys
import os
import numpy as np

sys.path.insert(0, ".")
from flybrainer.interfaces import Stimulus
from scripts._bench_factories import make_toy_brain, make_malecns_brain

os.makedirs("bench_v12", exist_ok=True)


def capture(factory, name, seed=0):
    b1 = factory()
    b2 = factory()
    rng = np.random.default_rng(seed)
    n16 = len(b1.populations["R1-R6"])
    n8 = len(b1.populations["R8"])
    stim = Stimulus(r16=rng.random(n16), r8=rng.random(n8))
    b1.observe(stim, neural_ms=200.0)
    b2.observe(stim, neural_ms=200.0)
    b1.reset()
    b2.reset()
    r1 = b1.observe(stim, neural_ms=200.0)
    r2 = b2.observe(stim, neural_ms=200.0)
    np.save(f"bench_v12/{name}_a.npy", r1.counts)
    np.save(f"bench_v12/{name}_b.npy", r2.counts)
    det = bool(np.array_equal(r1.counts, r2.counts))
    print(f"{name}: total={int(r1.counts.sum())}, max={int(r1.counts.max())}, det={det}")
    sys.stdout.flush()


if __name__ == "__main__":
    capture(make_toy_brain, "toy")
    capture(make_malecns_brain, "malecns")
