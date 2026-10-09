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
