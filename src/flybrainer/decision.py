"""OptionDecider: high-level business API on top of the ensemble.

Use case: given the current market state (a SensorFrame) and N candidate
option structures / position scenarios, run N independent brains in
parallel and return the brain's response to each.  The caller is free
to map "brain response" → "trading decision" however it likes (which
neurons spiked, which MBON, ratio of realized-vs-implied, etc.).

Typical use::

    from flybrainer.decision import OptionDecider, make_iv_scenarios
    from scripts._bench_factories import make_malecns_brain

    decider = OptionDecider(make_malecns_brain, num_workers=8)

    base = sensor_frame_now()
    scenarios = make_iv_scenarios(base, iv_offsets=(0.5, 0.75, 1.0, 1.25, 1.5))
    responses = decider.evaluate(scenarios)
    best = max(responses, key=lambda r: r.score(population="KC"))
    print("most active KC under high IV:", best.scenario_id)

Design notes
------------
- Encoding happens on the main process (cheap, just numpy ops). The
  eye_map is read from a template brain built once at construction.
- The actual `Brain.observe` runs in the worker pool. Each worker
  constructs its own brain on first use (the keep-alive pattern).
- The decider holds one template brain for the lifetime of the
  process; reset it explicitly if you want fresh state across many
  decisions.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from flybrainer.encoders import make_encoder
from flybrainer.ensemble import BatchBrain
from flybrainer.interfaces import (
    BrainProtocol,
    ObservationResult,
    SensorFrame,
    Stimulus,
)


@dataclass
class CandidateResponse:
    """Brain response to one scenario, plus a few derived summaries."""

    scenario_id: str
    counts: np.ndarray
    sim_ms: float
    compute_seconds: float
    wall_seconds: float
    population_rates: dict[str, float] = field(default_factory=dict)

    def score(
        self,
        population: str,
        brain: BrainProtocol | None = None,
        metric: str = "mean",
    ) -> float:
        """Default ranking metric.  mean counts in a population, else
        ``max``, ``sum``."""
        if brain is None:
            raise ValueError("score() needs the brain to look up the population indices")
        idx = brain.populations.get(population)
        if idx is None or len(idx) == 0:
            raise KeyError(f"unknown population: {population!r}")
        c = self.counts[idx]
        if metric == "mean":
            return float(c.mean())
        if metric == "max":
            return float(c.max())
        if metric == "sum":
            return float(c.sum())
        raise ValueError(f"metric must be mean|max|sum, got {metric!r}")


def make_iv_scenarios(
    base: SensorFrame,
    iv_offsets: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
    premium_scale: Callable[[float], float] | None = None,
    id_prefix: str = "iv",
) -> list[SensorFrame]:
    """N candidate scenarios parametrised by IV offset from `base.straddle_premium`.

    The VIX value is shifted in proportion; the straddle premium is scaled
    (or set via `premium_scale`).  All other fields are inherited from `base`.
    Useful for "what if realised vol is X% higher / lower" sweeps.
    """
    out: list[SensorFrame] = []
    base_vix = base.vix
    base_prem = base.straddle_premium or 0.0
    for off in iv_offsets:
        scaled = (premium_scale or (lambda o: base_prem * o))(off)
        out.append(
            SensorFrame(
                timestamp=base.timestamp,
                index_bars=base.index_bars,
                vix=base_vix * off,
                vix_bars=base.vix_bars,
                straddle_premium=scaled,
                entry_credit=base.entry_credit,
                days_to_expiry=base.days_to_expiry,
                minutes_since_open=base.minutes_since_open,
                position_lots=base.position_lots,
            )
        )
    return out


def make_strike_scenarios(
    base: SensorFrame,
    strikes: Sequence[float],
    id_prefix: str = "strike",
) -> list[SensorFrame]:
    """N scenarios for a straddle centred at each strike.  Reuses base bars.

    The market-data fields are identical; only the scenario_id and the
    implied premium differ.  In a real broker, you'd resolve each strike
    to a Contract and look up its quote; here we just vary the
    `straddle_premium` linearly with the strike (placeholder).
    """
    out: list[SensorFrame] = []
    base_prem = base.straddle_premium or 100.0
    for k, s in enumerate(strikes):
        out.append(
            SensorFrame(
                timestamp=base.timestamp,
                index_bars=base.index_bars,
                vix=base.vix,
                vix_bars=base.vix_bars,
                straddle_premium=base_prem * (0.6 + 0.2 * k),
                entry_credit=base.entry_credit,
                days_to_expiry=base.days_to_expiry,
                minutes_since_open=base.minutes_since_open,
                position_lots=base.position_lots,
            )
        )
    return out


class OptionDecider:
    """Ensemble-backed parallel evaluation of N market scenarios.

    Parameters
    ----------
    brain_factory : callable
        Module-level function returning a ``BrainProtocol``.  Required
        for the keep-alive ensemble.
    num_workers : int | None
        Defaults to ``min(8, cpu_count - 1)`` to match the benchmark's
        sweet spot.
    neural_ms : float
        Per-scenario simulated time in ms.  Default 200 ms (20 steps of
        10 ms, plenty for short-horizon decisions).
    encoder : str
        Encoder name registered in flybrainer.encoders.ENCODER_NAMES.
    warmup : bool
        If True, run a dummy stimulus through the pool on construction
        so the first real decision doesn't pay JIT + graph load cost.
    """

    def __init__(
        self,
        brain_factory: Callable[[], BrainProtocol],
        num_workers: int | None = None,
        neural_ms: float = 200.0,
        encoder: str = "bars",
        warmup: bool = True,
    ):
        if num_workers is None:
            num_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
        self.brain_factory = brain_factory
        self.num_workers = int(num_workers)
        self.neural_ms = float(neural_ms)
        self.encoder_name = encoder
        # Template brain for encoding (stays on main process; cheap)
        self._template_brain = brain_factory()
        self._eye_map = self._template_brain.eye_map()
        self.encoder = make_encoder(encoder, eye_map=self._eye_map)
        # Ensemble pool
        self.ensemble = BatchBrain(
            num_workers=self.num_workers, brain_factory=brain_factory
        )
        self.ensemble.start()
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        # Build a zero stimulus matching the brain's input shape.
        n_r16 = len(self._template_brain.populations["R1-R6"])
        n_r8 = len(self._template_brain.populations["R8"])
        stim = Stimulus(
            r16=np.zeros(n_r16, dtype=np.float64),
            r8=np.zeros(n_r8, dtype=np.float64),
        )
        # Issue num_workers no-op tasks so every worker JIT-compiles
        # and constructs its brain once.
        n = self.num_workers
        self.ensemble.observe_batch([stim] * n, neural_ms=10.0)
        # The above reset the workers' brains to clock=0; reset the
        # template brain too so the user sees a clean state.
        self._template_brain.reset()

    def evaluate(
        self,
        scenarios: list[SensorFrame],
        scenario_ids: list[str] | None = None,
        neural_ms: float | None = None,
    ) -> list[CandidateResponse]:
        """Run N scenarios in parallel, return per-scenario responses.

        Results are returned in the same order as `scenarios`.  The
        ensemble resets each worker's brain before the batch starts,
        so responses are independent.
        """
        n = len(scenarios)
        if n == 0:
            return []
        if scenario_ids is None:
            scenario_ids = [f"cand_{i:04d}" for i in range(n)]
        if len(scenario_ids) != n:
            raise ValueError(
                f"scenario_ids length {len(scenario_ids)} != scenarios length {n}"
            )
        neural_ms = float(neural_ms) if neural_ms is not None else self.neural_ms

        t_encode_0 = time.perf_counter()
        stimuli = [self.encoder.encode(s, self._template_brain) for s in scenarios]
        encode_ms = (time.perf_counter() - t_encode_0) * 1000

        t_run_0 = time.perf_counter()
        results = self.ensemble.observe_batch(stimuli, neural_ms=neural_ms)
        run_ms = (time.perf_counter() - t_run_0) * 1000
        wall_times = self.ensemble.last_wall_times

        responses: list[CandidateResponse] = []
        for sid, r, wt in zip(scenario_ids, results, wall_times):
            responses.append(
                CandidateResponse(
                    scenario_id=sid,
                    counts=r.counts,
                    sim_ms=r.sim_ms,
                    compute_seconds=r.compute_seconds,
                    wall_seconds=float(wt),
                    population_rates=self._template_brain.population_rates(
                        r.counts, r.neural_ms
                    ),
                )
            )
        responses[0]._meta = {  # type: ignore[attr-defined]
            "encode_ms": encode_ms,
            "run_ms": run_ms,
            "n_scenarios": n,
            "num_workers": self.num_workers,
            "neural_ms": neural_ms,
        }
        return responses

    def shutdown(self) -> None:
        self.ensemble.shutdown()

    def __enter__(self) -> OptionDecider:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


__all__ = [
    "OptionDecider",
    "CandidateResponse",
    "make_iv_scenarios",
    "make_strike_scenarios",
]
