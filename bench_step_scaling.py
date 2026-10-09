"""Test: per-call fixed overhead vs per-step cost in run_steps (with drive)."""
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

    # Build a constant drive (lamina + photoreceptors at 50% lum)
    drive = np.zeros(brain.n, dtype=np.float64)
    drive[brain.lamina] = 12.0
    r16_drive = 30.0 * 0.5 / (0.5 + 0.5)
    r8_drive = 30.0 * 0.5 / (0.5 + 0.5)
    drive[brain.r16] = r16_drive
    drive[brain.r8] = r8_drive
    bin_counts = np.zeros(brain.n, dtype=np.int32)
    counts = np.zeros(brain.n, dtype=np.int32)

    print("Simulating 200ms (2000 steps) at constant drive with varying steps/call:\n")
    for n_steps_per_call in (50, 100, 200, 500, 1000, 2000):
        total_steps = 2000
        n_calls = total_steps // n_steps_per_call
        brain.reset()
        brain.kernel.set_drive(drive)
        t0 = time.perf_counter()
        for _ in range(n_calls):
            bin_counts.fill(0)
            brain.kernel.run(n_steps_per_call, bin_counts)
        elapsed = (time.perf_counter() - t0) * 1000
        per_call = elapsed / n_calls
        per_step = elapsed / total_steps
        total_spikes = int(bin_counts.sum())
        print(f"steps/call={n_steps_per_call:5d}  calls={n_calls:3d}  "
              f"total={elapsed:7.1f}ms  per_call={per_call:6.2f}ms  per_step={per_step:6.3f}ms  "
              f"spikes={total_spikes}")


if __name__ == "__main__":
    main()
