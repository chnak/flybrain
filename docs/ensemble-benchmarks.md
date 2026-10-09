# Ensemble Decision Benchmarks (post kernel-parallel)

**Date:** 2026-01-04 (revised)
**Hardware:** 20 CPU cores, Windows
**Script:** `scripts/bench_decision.py`

**Important:** the kernel is now parallel via numba `prange` on the
materialize loop (see `docs/kernel-parallel-optimization.md`).  The
numbers below reflect the combined effect of (a) `BatchBrain` and
(b) the parallel kernel.  The single-call wall dropped from
1611 ms to **471 ms** on MaleCNS, a 3.4x win from one line change
in the kernel.

## Setup

`flybrainer.ensemble.BatchBrain` runs N independent `Brain.observe()` calls
in parallel using `multiprocessing.ProcessPoolExecutor` (spawn context).

Two modes:

- **Keep-alive** (default): one brain per worker, constructed lazily on
  first use, reused forever. Zero per-call graph pickling. The factory
  must be a module-level function (not a closure) so the spawn-imported
  worker can import it by name. The factory is shipped via `initargs`
  and stored in module state; each worker calls it on first use.
- **Pickle mode**: every call pickles the graph and ships it. Only
  practical for sub-MB graphs.

## Results (post parallel kernel)

### 100-neuron toy graph

| Config | ms/call | wall p50 (ms) | wall p99 (ms) | speedup |
|---|---|---|---|---|
| serial | 1.46 | – | – | 1.00x |
| 1-proc | 7.16 | 1.7 | 58.6 | 0.20x |
| 2-proc | 1.55 | 2.4 | 7.1 | 0.94x |
| 4-proc | 0.79 | 2.5 | 3.4 | **1.84x** |
| 8-proc | 0.64 | 4.0 | 4.7 | **2.27x** |
| 16-proc | 0.60 | 7.5 | 8.2 | **2.41x** |

Note: 1-proc and low-N are now relatively worse than before because the
serial baseline is 3.7x faster (1.46 ms vs 5.34 ms); IPC overhead is the
same, so the relative cost is bigger.  Past 4-proc the absolute
throughput is excellent.

### MaleCNS v1.0 (166,700 neurons, ~290 MB compiled graph)

| Config | ms/call | wall p50 (ms) | wall p99 (ms) | speedup |
|---|---|---|---|---|
| serial | 471 | – | – | 1.00x |
| 1-proc | 485 | 482 | 495 | 0.97x |
| 2-proc | 282 | 561 | 568 | 1.67x |
| 4-proc | 208 | 827 | 836 | 2.26x |
| 8-proc | 169 | 1318 | 1344 | **2.79x** |
| 16-proc | 166 | 1315 | 1324 | 2.84x |

At 8-proc the wall per task is 1318 ms — almost 3x the single-call
serial time, because each call now spends the bulk of its time in
deliver spikes and advance active list (both still sequential), not
in the parallelized materialize loop.  At 16-proc the wall is flat
(1315 ms) because we're already past the work-per-call lower bound.

## Combined Story

| Path                            | Time     | Speedup vs naive serial |
|---------------------------------|----------|-------------------------|
| Naive single-threaded MaleCNS   | 1611 ms  | 1.00x                   |
| + parallel materialize loop     | 471 ms   | 3.4x                    |
| + 8-way ensemble (8 candidates) | 1318 ms  | 1.22x per call *        |
| Combined: 8 candidates in 1.3s  | 1.3 s    | **9.5x**                |

(*) Per call: 1318 / 8 ≈ 165 ms.  Total wall 1.3 s.

**Best end-to-end MaleCNS decision time for 8 candidates: 1.3 s**
(was 12.9 s before any of these changes).  A 9.5x win from two
complementary optimisations: parallelize the single-core hot loop
and parallelize the ensemble axis.

## Lessons for the System

1. **Embarrassingly-parallel axis is the ensemble, not the kernel.**
   The single-threaded LIF kernel is hard to parallelise (cross-neuron
   conductance sums, shared active list).  We don't try.  We run N
   independent brains on N cores.

2. **For MaleCNS-scale graphs, keep-alive mode is essential.** A 290 MB
   graph pickled per call would cost ~3 s; keep-alive makes per-call
   overhead negligible (just the counts ndarray coming back).

3. **For 100-neuron-scale graphs, even 16-proc is still scaling.**  We
   could go higher. But for the trading use case, the typical
   `observe_batch` size is N=8-32 (one per option strike / expiry /
   scenario), so 8-16 workers is the realistic sweet spot.

4. **Optimal worker count = min(N_stimuli, num_cpu_cores - 1).** Past that
   point you're either idle or paying IPC overhead with no benefit.

5. **1-proc is always worse than serial** (just IPC overhead, no
   parallel work).  Never use `BatchBrain(num_workers=1)`; use the
   brain directly.

6. **Decision latency = max(individual compute times).** For balanced
   workloads, the single 471 ms compute becomes 1318 ms wall with 8
   workers because each worker's call includes a parallel materialize
   on its own n neurons (the TBB pool inside the worker is 16 threads,
   competing for 8 cores).  As N_stim grows, we trade latency for
   throughput; the user can decide which they care about.

7. **Single-core parallel and ensemble stack multiplicatively.**  The
   materialize loop was the dominant cost of the single call; making
   it parallel removed a fixed cost that did not benefit from
   ensemble.  The remaining sequential work (spike delivery, active
   list) is what the ensemble attacks.

## Recommended Production Config

```python
NUM_WORKERS = min(8, os.cpu_count() - 1)   # leave 1 core for OS/encoder
# NUM_WORKERS = min(N_STIMULI, NUM_WORKERS)  # never over-subscribe
```

For the straddle-decision use case: 8 candidate structures per decision
→ 8-proc ensemble → 1.3 s wall vs 12.9 s serial = **9.5x speedup**,
plus deterministic "all options evaluated in the same neural time"
which the serial model cannot offer.

## Caveats

- The bench includes a warm-up call (workers JIT compile + first
  import + first graph load). Real production code should do this
  once at boot, not per decision.
- `Brain.reset()` between calls is cheap (~0.1 ms) but does require
  the worker to hold the brain state, not share it across calls.
- 8 stims × 8 workers is perfectly balanced.  8 stims × 16 workers
  has 8 idle workers — don't do that.
- The `parallel=True` kernel uses numba's TBB pool; this consumes
  some of the cores that the ensemble also wants.  At 8 ensemble
  workers on a 20-core box, each worker can still use 2-3 TBB
  threads before contention kicks in, so we get the speedup we
  see.  On a 4-core box the picture would be different: ensemble
  would dominate and `parallel=True` would be a wash.
