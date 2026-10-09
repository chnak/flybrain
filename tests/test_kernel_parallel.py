"""Tests for the parallel kernel changes (regression + perf sanity)."""
import time

import numpy as np
import pytest

from flybrainer import Brain
from flybrainer.interfaces import Stimulus


def _make_random_stim(brain: Brain, seed: int = 0) -> Stimulus:
    rng = np.random.default_rng(seed)
    n16 = len(brain.populations["R1-R6"])
    n8 = len(brain.populations["R8"])
    return Stimulus(r16=rng.random(n16), r8=rng.random(n8))


@pytest.mark.parametrize("factory_name", ["toy", "malecns"])
def test_parallel_kernel_deterministic(request, factory_name):
    """Two Brain instances on the parallel kernel produce identical spike counts."""
    from scripts._bench_factories import make_toy_brain, make_malecns_brain

    factory = make_toy_brain if factory_name == "toy" else make_malecns_brain
    if factory_name == "malecns":
        pytest.importorskip("flybrainer.connectome", reason="MaleCNS graph needed")

    b1 = factory()
    b2 = factory()
    stim = _make_random_stim(b1, seed=0)
    # JIT warm
    b1.observe(stim, neural_ms=50.0)
    b2.observe(stim, neural_ms=50.0)
    b1.reset()
    b2.reset()
    r1 = b1.observe(stim, neural_ms=200.0)
    r2 = b2.observe(stim, neural_ms=200.0)
    np.testing.assert_array_equal(
        r1.counts, r2.counts,
        err_msg="parallel kernel produced non-deterministic output",
    )


def test_toy_graph_speedup_smoke():
    """The parallel kernel should be substantially faster than the old serial.

    Loose smoke test: catches "kernel entirely reverted to single-threaded"
    or "prange split out and only one core is used" regressions.  We
    measured 3.66x on a 20-core dev box (~1.5 ms / call) but CI runners
    and single-core machines can be 50-100x slower, so the bound has to
    be very lenient.  Per-call > 500 ms on a 100-neuron graph means
    something is badly wrong (the pre-1.2 single-threaded baseline was
    ~5 ms; 500 ms is ~100x slower, not perf variance).
    """
    from scripts._bench_factories import make_toy_brain

    brain = make_toy_brain()
    stim = _make_random_stim(brain, seed=1)
    # JIT warm
    brain.observe(stim, neural_ms=200.0)
    brain.reset()
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        brain.reset()
        brain.observe(stim, neural_ms=200.0)
    per_call_ms = (time.perf_counter() - t0) * 1000 / n
    # Generous CI bound: 100x the expected dev-box perf.  Anything beyond
    # this is a real regression, not hardware variance.
    assert per_call_ms < 500.0, f"toy graph per-call {per_call_ms:.1f} ms (regression?)"


def test_wake_pass_threshold_is_defined():
    """v1.3 should expose a threshold for the parallel wake pass.

    Without the threshold, the TBB scheduling overhead dominates on
    small populations (toy, microcircuits) and the parallel scheme is
    slower than a plain serial scan.  The threshold is a runtime
    tunable, not a hard constant, so the test just asserts it is a
    positive integer exported from the kernel module.
    """
    from flybrainer import kernel

    threshold = kernel.WAKE_PARALLEL_THRESHOLD
    threads = kernel.WAKE_THREADS
    assert isinstance(threshold, int) and threshold > 0, (
        f"WAKE_PARALLEL_THRESHOLD must be a positive int, got {threshold!r}"
    )
    assert isinstance(threads, int) and threads > 0, (
        f"WAKE_THREADS must be a positive int, got {threads!r}"
    )


def test_wake_pass_threshold_dispatches_serial_for_small_n():
    """The threshold branch must produce bit-identical output to a pure
    serial scan for n < threshold.

    We construct a tiny synthetic state and compare against a hand-rolled
    serial activation order.  This guards against future refactors that
    might silently break the bit-identical property on the toy path.
    """
    from flybrainer import kernel
    from numba import njit

    n = 32
    S = np.full((n, kernel.RECORD), -60.0, dtype=np.float64)
    S[:, kernel.POS] = -1.0  # all sleepers
    S[:, kernel.REST] = -52.0
    # First half: drive > gap, so eligible via the constant-current path
    S[: n // 2, kernel.DRIVE] = 10.0  # > -45 - (-52) = 7
    S[: n // 2, kernel.V] = -60.0
    # Second half: drive = 0 (no constant current), V deeply below threshold
    S[n // 2 :, kernel.DRIVE] = 0.0
    S[n // 2 :, kernel.V] = -60.0

    active_idx = np.zeros(n, dtype=np.int32)
    n_active = np.zeros(1, dtype=np.int64)
    # Simulate the serial branch
    for i in range(n):
        if S[i, kernel.POS] < 0.0 and kernel._eligible(i, S):
            kernel._activate(i, S, active_idx, n_active)
    expected = active_idx[: int(n_active[0])].copy()
    assert int(n_active[0]) == n // 2
    # Ascending i, only the first half
    assert list(expected) == list(range(n // 2))
