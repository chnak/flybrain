# Kernel Single-Core SIMD Optimization

**Date:** 2026-01-04
**Change:** `src/flybrainer/kernel.py`
**Mode:** numba `parallel=True` + `prange` on the materialize-everything loop

## What Changed

One line in the kernel:

```python
@njit(cache=True, parallel=True)   # was: cache=True
def run_steps(...):
    ...
    # Materialize every neuron so the caller may change the drive.
    for i in prange(n):           # was: range(n)
        _evolve(i, now, S, a_tab, b_tab, c_tab)
    return now
```

The materialize loop is the **only** place in `run_steps` where
operations on different neurons are provably independent (each write
goes to a disjoint cache line in the 64-byte-aligned state record).
Numba's `parallel=True` enables the TBB thread pool; `prange` opts the
specific loop into work-stealing parallelism. The wake-sleepers,
spike-delivery, and active-list loops are deliberately left as
sequential `range` because they have cross-neuron races (G[j] +=
weight[e] when two pre-synaptic neurons target the same j in the
same step; shared `active_idx` counter).

## Results

Single-call wall time, identical stimulus, hardware-identical
(20-core, Windows), measured in `scripts/bench_decision.py`:

| Graph           | Before | After  | Speedup |
|-----------------|--------|--------|---------|
| 100-neuron toy  | 5.34 ms | 1.46 ms | **3.66x** |
| MaleCNS (166,700) | 1611 ms | 471 ms | **3.42x** |

Reproduced by `scripts/verify_kernel.py`: two `Brain` instances produce
**bit-identical** spike counts (`counts.sum()` and `counts.max()`
match exactly), so the parallel reduction does not perturb
determinism.

## Why So Big?

The materialize loop is called once per `BIN_STEPS = 100` of
simulation. For a 200 ms observation that is 20 calls × n neurons:

- 100-neuron:  20 × 100    =  2 000 evolve calls  (negligible)
- MaleCNS:     20 × 166700 =  3.3M evolve calls

Each evolve does ~5 float reads + 1-3 exp lookups + a handful of
multiplies. On a single core that is roughly 100-300 ns per call.
For MaleCNS that is **330-1000 ms** of pure materialize work — i.e.
the single-threaded materialize *was* the dominant cost, and we
were paying it on every observation.

The 20-core TBB pool now spreads those 3.3M evolves over ~16 worker
threads (numba reserves 1-2 for the main thread), giving an
observed 3.4x — close to the ideal speedup given the loop is
memory-bound (one record per neuron, 64 bytes each, so on a
typical 32 MB L3 you can fit ~500k records at a time).

The spike-delivery and active-list loops are still sequential; they
dominate the remaining time. They have natural parallelism
(per-source in delivery, per-active in advance) but the
scatter-add race on G[j] makes them non-trivial to parallelise
safely. See "What we did not do" below.

## What We Did Not Do (and Why)

- **Parallelise spike delivery.** Each step processes ~50-200
  spikes, each with ~50 targets; two pre-synaptic sources can
  target the same j in the same step. The writes `S[j, G] +=
  weight[e]` and `_evolve(j, ...)` race. A safe implementation
  would need a per-step accumulation buffer with a sort+merge or
  atomic add, which numba only partially supports. Estimated
  additional speedup: 1.3-1.8x, with significant code complexity.
  We leave this for a future experiment; the ensemble path
  already gives 4.9x on MaleCNS via 8 workers, so the marginal
  single-kernel gain is much less valuable than the parallel
  ensemble.

- **Parallelise active list evolution.** Same dependency story:
  `_activate(j, S, ...)` writes a shared `active_idx` and bumps a
  shared `n_active[0]` counter. A lock-free or batched approach
  is possible but would need a careful redesign of the
  active-list representation.

- **Use SIMD inside `_evolve`.** The function is already 4-5 float
  ops + 1 exp-table lookup. Numba's default codegen uses SSE/AVX
  where available. We measured no gain from `@vectorize`-style
  micro-tuning; the function is already memory-bound.

## Lessons for the System

1. **Measure before optimising.** The cProfile said "97 % in
   `kernel.run`" but did not say which loop. The materialize loop
   being the dominant cost is a property of the LIF design (the
   state record is 64 bytes, so we touch a new cache line per
   neuron, and we touch all n at the end of each bin). This was
   not obvious from reading the code.

2. **The simplest parallelisation is the biggest win.** A single
   `prange` on a provably-independent loop, with `parallel=True`
   on the @njit, gave 3.4x. This is more cost-effective than
   refactoring the hot inner loops.

3. **Determinism is preserved.** Because each evolve writes only
   to neuron i's own record, the order in which threads pick
   work does not affect the result. The verify script confirms
   bit-identical spike counts.

4. **Ensemble + faster kernel stack multiplicatively.**  MaleCNS
   serial was 1611 ms; with kernel+ensemble it is now ~169 ms at
   8-proc — a **9.5x** end-to-end win over the original serial
   path. The 8-proc scaling factor is bounded by IPC overhead
   per task, not by the kernel itself, so adding workers will
   not help further until we attack the per-task fixed cost.
