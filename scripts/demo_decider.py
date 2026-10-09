"""Demo: OptionDecider evaluating 8 IV scenarios on MaleCNS in parallel.

Compares the OptionDecider flow against the raw BatchBrain API and shows
the per-scenario brain response.  Useful as a smoke test and as a
worked example for the design doc.
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from flybrainer import OptionDecider, make_iv_scenarios
from flybrainer.decision import CandidateResponse
from flybrainer.interfaces import Bar, SensorFrame
from scripts._bench_factories import make_malecns_brain, make_toy_brain


def make_frame():
    IST = timezone(timedelta(hours=5, minutes=30))
    bars = tuple(
        Bar(
            timestamp=datetime(2024, 1, 15, 9, 25 + i, tzinfo=IST),
            open=21600.0 + i * 5,
            high=21650.0 + i * 5,
            low=21580.0 + i * 5,
            close=21630.0 + i * 5,
            volume=1000 + i * 50,
        )
        for i in range(3)
    )
    return SensorFrame(
        timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=IST),
        index_bars=bars,
        vix=14.2,
        vix_bars=bars,
        straddle_premium=120.5,
        entry_credit=None,
        days_to_expiry=4.0,
        minutes_since_open=10,
        position_lots=0,
    )


def demo(label: str, brain_factory, n_scenarios: int = 8) -> None:
    print(f"\n========== {label} ==========")
    t0 = time.perf_counter()
    with OptionDecider(brain_factory, num_workers=8, neural_ms=200.0, warmup=True) as decider:
        setup_ms = (time.perf_counter() - t0) * 1000
        print(f"setup: {setup_ms:7.0f} ms  (template brain + pool warm-up)")

        # Generate N candidate scenarios (IV offsets 0.5x .. 2.0x)
        base = make_frame()
        scenarios = make_iv_scenarios(
            base, iv_offsets=np.linspace(0.5, 2.0, n_scenarios).tolist()
        )
        ids = [f"iv={o:.2f}x" for o in np.linspace(0.5, 2.0, n_scenarios)]

        # Evaluate all N in parallel
        t0 = time.perf_counter()
        responses = decider.evaluate(scenarios, scenario_ids=ids)
        wall_ms = (time.perf_counter() - t0) * 1000

        # Summary
        brain = decider._template_brain
        meta = getattr(responses[0], "_meta", {})
        print(
            f"evaluate: {wall_ms:7.0f} ms  "
            f"(encode={meta.get('encode_ms', 0):.1f} ms, run={meta.get('run_ms', 0):.0f} ms, "
            f"per-call wall={wall_ms/n_scenarios:.0f} ms)"
        )
        print(f"\n{'scenario':<14} {'KC Hz':>8} {'MBON Hz':>9} {'DNp20L Hz':>10} {'wall ms':>9}")
        print("-" * 56)
        for r in responses:
            pr = r.population_rates
            print(
                f"{r.scenario_id:<14} "
                f"{pr.get('KC', 0):>8.2f} "
                f"{pr.get('MBON', 0):>9.2f} "
                f"{pr.get('DNp20_L', 0):>10.2f} "
                f"{r.wall_seconds*1000:>9.0f}"
            )
        # Ranking
        kc_scores = [(r.scenario_id, r.score("KC", brain)) for r in responses]
        kc_scores.sort(key=lambda t: -t[1])
        print("\nranked by KC activity (descending):")
        for sid, sc in kc_scores:
            print(f"  {sid:<14}  KC counts = {sc:.1f}")


def main():
    print(f"CPU count: {os.cpu_count()}")
    demo("100-neuron toy graph", make_toy_brain, n_scenarios=8)
    demo("MaleCNS (166,700 neurons)", make_malecns_brain, n_scenarios=8)


if __name__ == "__main__":
    main()
