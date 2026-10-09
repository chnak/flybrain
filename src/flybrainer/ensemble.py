"""Multi-CPU ensemble: run N independent Brain.observe() calls in parallel.

The flybrainer kernel is single-threaded by design (deterministic LIF with
a global ring buffer). Naively parallelising the inner Numba loop is hard
because the synaptic conductance G[j] += weight[e] and the active-list
manipulation have cross-neuron dependencies.  The embarrassingly-parallel
axis is the *ensemble*: run N brains (or the same brain with N different
stimuli) on N CPU cores via multiprocessing.

Usage (keep-alive, recommended)::

    from flybrainer.ensemble import BatchBrain

    def my_brain_factory():
        return Brain(plastic=False)

    with BatchBrain(num_workers=8, brain_factory=my_brain_factory) as ens:
        results = ens.observe_batch(stimuli, neural_ms=200.0)
        # results[i].counts for input stimuli[i], in input order

Usage (pickle mode, brain is small or call rate is low)::

    brain = Brain(...)  # small graph
    with BatchBrain(num_workers=4, keep_alive=False) as ens:
        results = ens.observe_batch(stimuli, neural_ms=200.0, brain=brain)

Each worker process constructs its own brain on first use (keep-alive) and
reuses it for the lifetime of the pool.  No pickling of the graph happens
on the hot path.  The factory MUST be a module-level function (not a
closure) so the spawn-imported worker can import it.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np

from flybrainer.interfaces import BrainProtocol, ObservationResult, Stimulus


# ---------------------------------------------------------------------------
# Worker-side state and functions
# ---------------------------------------------------------------------------

# Brain instances installed per worker process, keyed by os.getpid().
# Each worker process gets one brain; no sharing across processes.
_BRAINS: dict[int, BrainProtocol] = {}


def _init_worker_with_factory(brain_factory: Callable[[], BrainProtocol]) -> None:
    """initializer for ProcessPoolExecutor.  Called once per worker.

    Stores the factory in module state; the brain is constructed lazily
    on first observe() so the worker startup is fast and we don't waste
    memory if the worker is never asked to do work.
    """
    global _BRAIN_FACTORY
    _BRAIN_FACTORY = brain_factory
    _BRAINS.clear()


_BRAIN_FACTORY: Callable[[], BrainProtocol] | None = None


def _get_brain() -> BrainProtocol:
    pid = os.getpid()
    if pid not in _BRAINS:
        if _BRAIN_FACTORY is None:
            raise RuntimeError("worker: no brain_factory registered")
        _BRAINS[pid] = _BRAIN_FACTORY()
    return _BRAINS[pid]


def _worker_observe_keep_alive(
    stimulus: Stimulus,
    neural_ms: float,
) -> tuple[int, np.ndarray, float, float, float]:
    """Worker-side observe() using the lazy-installed brain. No graph pickling."""
    t_start = time.perf_counter()
    brain = _get_brain()
    result = brain.observe(stimulus, neural_ms=neural_ms)
    wall_s = time.perf_counter() - t_start
    return (
        os.getpid(),
        np.asarray(result.counts, dtype=np.int32),
        float(result.sim_ms),
        float(result.compute_seconds),
        float(wall_s),
    )


def _worker_observe_pickle(
    brain_pickle: bytes,
    stimulus: Stimulus,
    neural_ms: float,
) -> tuple[int, np.ndarray, float, float, float]:
    """Pickle-mode worker: unpickle brain per call.  Slow for big graphs."""
    t_start = time.perf_counter()
    brain: BrainProtocol = pickle.loads(brain_pickle)
    result = brain.observe(stimulus, neural_ms=neural_ms)
    wall_s = time.perf_counter() - t_start
    return (
        os.getpid(),
        np.asarray(result.counts, dtype=np.int32),
        float(result.sim_ms),
        float(result.compute_seconds),
        float(wall_s),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BatchBrain:
    """ProcessPoolExecutor-backed ensemble of N brains.

    Parameters
    ----------
    num_workers : int | None
        Number of worker processes.  Default: ``cpu_count() - 1``.
    keep_alive : bool
        If True (default), one brain per worker, constructed lazily by
        ``brain_factory`` and reused across calls.  The factory MUST be a
        module-level function (or a class with __call__) that returns a
        ``BrainProtocol`` instance.  Pass it via ``brain_factory=``.

        If False, every call pickles the brain and ships it.  Use this
        only for small graphs (sub-MB).
    brain_factory : callable | None
        Required when ``keep_alive=True``.  Called once per worker with
        no arguments, must return a ``BrainProtocol``.

    Notes
    -----
    On Windows / macOS the spawn start method is used (the only safe
    cross-platform choice).  The worker re-imports the main module; the
    factory must therefore be importable by name, not a local closure.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        keep_alive: bool = True,
        brain_factory: Callable[[], BrainProtocol] | None = None,
    ):
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)
        if keep_alive and brain_factory is None:
            raise ValueError("keep_alive=True requires brain_factory")
        self.num_workers = int(num_workers)
        self.keep_alive = bool(keep_alive)
        self.brain_factory = brain_factory
        self._executor: ProcessPoolExecutor | None = None
        self.last_wall_times: np.ndarray = np.zeros(0, dtype=np.float64)

    def __enter__(self) -> BatchBrain:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def start(self) -> None:
        if self._executor is not None:
            return
        ctx = mp.get_context("spawn")
        if self.keep_alive:
            assert self.brain_factory is not None
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=ctx,
                initializer=_init_worker_with_factory,
                initargs=(self.brain_factory,),
            )
        else:
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=ctx,
            )

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def observe_batch(
        self,
        stimuli: list[Stimulus],
        neural_ms: float = 200.0,
        brain: BrainProtocol | None = None,
    ) -> list[ObservationResult]:
        """Run N observe() calls in parallel, return results in input order."""
        if not stimuli:
            return []
        if self._executor is None:
            self.start()

        n = len(stimuli)
        future_to_index: dict[Any, int] = {}
        for i, stim in enumerate(stimuli):
            if self.keep_alive:
                fut = self._executor.submit(
                    _worker_observe_keep_alive, stim, float(neural_ms)
                )
            else:
                if brain is None:
                    raise ValueError("pickle mode requires `brain` argument")
                brain_bytes = pickle.dumps(brain)
                fut = self._executor.submit(
                    _worker_observe_pickle, brain_bytes, stim, float(neural_ms)
                )
            future_to_index[fut] = i

        results: list[ObservationResult | None] = [None] * n
        wall_times = np.zeros(n, dtype=np.float64)
        for fut, idx in future_to_index.items():
            wid, counts, sim_ms, compute_s, wall_s = fut.result()
            results[idx] = ObservationResult(
                counts=counts,
                neural_ms=float(neural_ms),
                sim_ms=sim_ms,
                compute_seconds=compute_s,
            )
            wall_times[idx] = wall_s
        self.last_wall_times = wall_times
        return results  # type: ignore[return-value]


__all__ = ["BatchBrain"]
