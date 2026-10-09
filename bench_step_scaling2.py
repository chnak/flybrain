"""Compare chunked vs unchunked for the same number of steps, isolating per-call overhead."""
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from flybrainer import Brain
from flybrainer.interfaces import Stimulus


def main():
    brain = Brain(plastic=False)
    rng = np.random.default_rng(0)
    n16 = len(brain.populations["R1-R6"])
    n8 = len(brain.populations["R8"])
    stim = Stimulus(r16=rng.random(n16), r8=rng.random(n8))
    # JIT warm
    brain.observe(stim, neural_ms=200.0)

    # Build constant drive
    drive = np.zeros(brain.n, dtype=np.float64)
    drive[brain.lamina] = 12.0
    drive[brain.r16] = 15.0
    drive[brain.r8] = 15.0
    bin_counts = np.zeros(brain.n, dtype=np.int32)

    # Total sim: 2000 steps
    # Test A: 20 calls of 100 steps
    # Test B: 1 call of 2000 steps
    # But state diverges! To compare same state, just time ONE observation.

    # For test A: do one full observation
    brain.reset()
    brain.kernel.set_drive(drive)
    times = []
    for _ in range(5):
        brain.reset()
        brain.kernel.set_drive(drive)
        t0 = time.perf_counter()
        for _ in range(20):
            bin_counts.fill(0)
            brain.kernel.run(100, bin_counts)
        times.append((time.perf_counter() - t0) * 1000)
    a = sorted(times)[2]
    print(f"20 calls x 100 steps: {a:.1f} ms (median of 5)")

    # For test B: do one big call
    times = []
    for _ in range(5):
        brain.reset()
        brain.kernel.set_drive(drive)
        t0 = time.perf_counter()
        bin_counts.fill(0)
        brain.kernel.run(2000, bin_counts)
        times.append((time.perf_counter() - t0) * 1000)
    b = sorted(times)[2]
    print(f"1 call x 2000 steps:  {b:.1f} ms (median of 5)")
    print(f"Ratio: {a/b:.2f}x (smaller / bigger)")
    print(f"Spikes A: {int(bin_counts.sum())}")

    # To be fair, the A case allows state to be observed between calls.
    # But the bin_counts is the same, and the work is the same.
    # If per-call overhead is significant, A should be slower than B.
    # If state divergence matters, A and B should be different.


if __name__ == "__main__":
    main()
