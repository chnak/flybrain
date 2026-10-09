# Kernel Performance Investigation (2026-02)

## TL;DR
The MaleCNS simulation is at **0.4x realtime** (495 ms per 200 ms sim). The bottleneck is the **inner simulation loop itself**, not the per-call overhead. Schedule-fusion (batching N bins into one numba call) was tested and **does not help** — only ~5% of time is in call boundary / Python<->JIT marshaling.

To reach 1x realtime requires ~2.5x speedup on the kernel itself. That needs deeper changes (SoA layout, lower precision, parallelizing the delivery loop). The 5% overhead is real but irrelevant to the goal.

## v1.3 (2026-03): parallel wake pass
Added a two-stage parallel scheme for the wake pass in `run_steps`. Each
of `WAKE_THREADS` (= 20 by default) prange-chunks writes the indices
of newly eligible sleepers into a private per-thread buffer slot, then
a sequential compaction step rebuilds the active set in ascending
neuron-id order. Because thread `tid` owns the disjoint chunk
`[tid*chunk, (tid+1)*chunk)`, the final iteration order matches the
serial scan, so the kernel stays bit-identical (`tests/test_kernel_parallel.py`).

A runtime threshold (`WAKE_PARALLEL_THRESHOLD = 1024`) dispatches to a
plain serial scan for small populations: the TBB scheduling overhead
of `prange` is ~20 us, so it only pays off above a few hundred
neurons. On MaleCNS (n=166,700, 20 bins per `observe(200ms)`) the
parallel wake pass is ~3-7% faster than serial on the cold path; on
toy (n=100) the serial path is ~50% faster than the parallel one
because the work is too small to amortize the prange.

Bit-identical: verified by `bench_v13_compare.py` against a v1.2
baseline (`toy: 860/860, malecns: 104828/104828, |delta|=0`).

## Where time goes
cProfile of 10x `observe(200ms)` on MaleCNS (n=166,700, 25.6M edges):

| Component | Total | Per-iter | Per-call | % |
|---|---|---|---|---|
| `kernel.run` | 4.63s | 463 ms | 23.2 ms (2000 steps) | 95% |
| `kernel.set_drive` | 0.15s | 14.5 ms | 0.72 ms | 3% |
| `kernel.reset` | 0.05s | 5.5 ms | 5.5 ms | 1% |
| `kernel.fresh` (alloc) | 0.05s | 4.9 ms | — | 1% |
| encoder | 0.025s | 0.26 ms | — | 0.5% |
| decoder | 0.003s | 0.03 ms | — | 0.06% |

Per call of `run(2000)`: 23.2 ms.
- Active-set evolves: ~166K × 2000 = 332M (4 cache-line touches each = 1.3G mem refs)
- Edge deliveries: 113K spikes × ~225 edges = 25.6M
- Wake-check (1× per call): 166K sequential
- Materialize (1× per call, parallel): 166K

**Effective throughput**: 358M operations in 23.2 ms ≈ 15 GOps/s, ≈1.4 ns/op. This is memory-bound (S array is 9.3 MB, doesn't fit in L2/L3).

## What was tried: schedule fusion
**Hypothesis**: each `set_drive; run(100)` pair crosses the Python<->numba boundary twice. If the host loop is collapsed to a single `run_schedule(drive_schedule, steps_per_bin, per_bin_counts)` call, the boundary cost vanishes.

**Result** (`scripts/test_3way.py`, since removed):
- 1× `run(2000)`: 553.8 ms
- 20× `run(100)` loop: 579.3 ms (1.05× slower)
- 1× `run_schedule(20 bins)`: 589.6 ms (1.06× of 1-call)

**Conclusion**: The per-call overhead is real but only ~5% of total. The schedule function is bit-identical (verified) but slightly slower than the loop because:
1. The inner `run_steps` still calls materialize + wake_check per bin.
2. Numba's slice assignments (`S[:, DRIVE] = drive_schedule[b]`, `per_bin_counts[b, :] = bin_counts`) cost the same as the host-side `set_drive` they replace.

To get the 5% back, the materialize in `run_steps` would need a `do_materialize=False` mode for intermediate bins, but 5% is not the bottleneck.

## Where the real time is
The 463 ms / iter in `kernel.run` is the actual simulation. To cut it in half we need to change the inner loop.

**Not the boundary** — verified above.

**Not the host loop in `observe()`** — it's only 2000 lines of trivial Python, negligible.

**Not the encoder/decoder** — sub-millisecond.

**Not reset** — 5 ms, dominated by `aligned_records` allocation; the alloc itself is needed (state shape can be big).

The remaining 463 ms is the work the kernel must do: 358M operations. At 1.4 ns/op we are close to the memory bandwidth ceiling. Further gains require one of:

1. **Lower precision (float32)** — halves memory traffic. Values fit comfortably in f32 (V ~ -70 to -40, G up to ~10s). Risk: need to verify plasticity/decoder precision is unaffected.

2. **SoA layout** — split `S` into 6 separate arrays (V, G, A, REST, DRIVE, LAST). Numba's prange over a tile then streams each array linearly, improving cache behavior. Risk: requires rewriting all the access patterns.

3. **Parallelize the delivery loop** — currently serial per-spike because multiple spikes can converge on the same postsynaptic target. A per-thread delta buffer + final merge pattern would let `prange` over the edge list. Risk: complex; merge cost may eat the gain.

4. **Skip the `_eligible` check in the hot loop** — only check when `S[j, G] + S[j, DRIVE]` plausibly crosses the threshold. Most G updates are far from threshold. Risk: subtle; can lose activation.

5. **GPU / explicit SIMD** — beyond numba's auto-vectorization. Out of scope for this codebase.

## Recommendation
The 0.4x realtime is *the* bottleneck. To get to 1x realtime, the kernel itself needs to be ~2.5x faster. The cheapest, lowest-risk change is **(1) float32**. The most architectural is **(2) SoA + (3) parallel delivery**.

Before investing in any of these, profile with `numba -p` (or `nsys`/`vtune`) to confirm the memory-bound hypothesis on a real workload — toy network timings are misleading because the active set is tiny and the kernel is compute-bound, not memory-bound.
