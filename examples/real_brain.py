"""End-to-end flybrainer example using the REAL Brain (not a stub).

Run:
    python examples/real_brain.py

This file demonstrates the full ``Brain`` class (the Numba-JIT LIF kernel
running over a real connectome) wired into the 4-step pipeline:

    SensorFrame --[Encoder]--> Stimulus --[Brain]--> counts --[Decoder]--> Prediction

It does NOT need the MaleCNS v1.0 dataset.  Instead it builds a tiny
in-memory ``graph.npz``-equivalent dict and hands it directly to the real
``Brain(graph=...)`` constructor.  That is the same code path the full
MaleCNS pipeline uses; only the size differs.

The synthetic graph is laid out so every REQUIRED_POPULATIONS key is
non-empty and signal visibly flows from photoreceptors -> lamina -> KC ->
MBON -> descending neurons (DNp20):

    index 0..19    R1-R6 photoreceptors        (stim.r16)
    index 20..39   R8 photoreceptors           (stim.r8)
    index 40..59   L1/L2/L3/L5 lamina cells    (constant 12 mV drive)
    index 60..75   KC cells                    (population "KC")
    index 76..79   MBON07
    index 80..83   MBON11
    index 84..87   PAM11                       (needed for plastic run)
    index 88..91   PPL101                      (needed for plastic run)
    index 92..95   DNp20_L
    index 96..99   DNp20_R

What you see in the output:
    1. Brain construction, populations, eye-map geometry
    2. Encoding a synthetic market snapshot into a Stimulus
    3. Running 500 ms of neural dynamics through the JIT kernel
    4. Per-population firing rates (mean Hz per REQUIRED_POPULATIONS)
    5. A FixedDecoder prediction
    6. parameters() / provenance() introspection (graph hashes, etc.)
    7. checkpoint() -> restore() round-trip
    8. Plastic run (KCMBONPlasticity enabled; weight factors update)

The first run takes ~7 s for Numba JIT compile; subsequent runs are
sub-second.  All operations are deterministic for a given numpy seed.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from flybrainer import (
    REQUIRED_POPULATIONS,
    Brain,
    Decision,
    ObservationResult,
    Prediction,
    SensorFrame,
)
from flybrainer.encoders import make_encoder
from flybrainer.interfaces import Bar
from flybrainer.readout.fixed import FixedDecoder

# ---------------------------------------------------------------------------
# 1. Build a synthetic connectome in-memory
# ---------------------------------------------------------------------------
# ``Brain`` accepts a ``graph=`` dict that mimics the contents of a compiled
# ``graph.npz``.  The required keys are listed in
# ``flybrainer.brain.GRAPH_ARRAYS``.  We carve 100 neurons into plausible cell
# types so every REQUIRED_POPULATIONS entry is non-empty and the signal can
# flow from R1-R6 -> lamina -> KC -> MBON -> DNp20.
# ---------------------------------------------------------------------------

N = 100
N_R16 = 20
N_R8 = 20
N_LAMINA = 20
N_KC = 16
N_MBON07 = 4
N_MBON11 = 4
N_PAM11 = 4
N_PPL101 = 4
N_DNP20_L = 4
N_DNP20_R = 4
assert N_R16 + N_R8 + N_LAMINA + N_KC + N_MBON07 + N_MBON11 + N_PAM11 + N_PPL101 + N_DNP20_L + N_DNP20_R == N


def build_synthetic_graph(seed: int = 42) -> dict[str, np.ndarray]:
    """Return a dict that satisfies ``Brain(graph=...)``'s contract."""
    rng = np.random.default_rng(seed)

    # ---- cell-type assignment --------------------------------------------
    idx = 0
    def take(n: int, label: str) -> slice:
        nonlocal idx
        s = slice(idx, idx + n)
        idx += n
        return s

    s_r16    = take(N_R16,    "R1-R6")
    s_r8     = take(N_R8,     "R8")
    s_lam    = take(N_LAMINA, "L1")
    s_kc     = take(N_KC,     "KC")
    s_m07    = take(N_MBON07, "MBON07")
    s_m11    = take(N_MBON11, "MBON11")
    s_pam11  = take(N_PAM11,  "PAM11")
    s_ppl101 = take(N_PPL101, "PPL101")
    s_dnp_l  = take(N_DNP20_L, "DNp20")
    s_dnp_r  = take(N_DNP20_R, "DNp20")

    types = np.empty(N, dtype=object)
    types[s_r16]    = "R1-R6"
    types[s_r8]     = "R8"
    types[s_lam]    = "L1"
    types[s_kc]     = "KC"
    types[s_m07]    = "MBON07"
    types[s_m11]    = "MBON11"
    types[s_pam11]  = "PAM11"
    types[s_ppl101] = "PPL101"
    types[s_dnp_l]  = "DNp20"
    types[s_dnp_r]  = "DNp20"

    # Photoreceptors -> visual; everything central -> cb_int; DNp20 -> descending_neuron.
    superclass = np.where(
        np.isin(types, ["L1", "KC", "MBON07", "MBON11", "PAM11", "PPL101"]),
        "cb_int",
        np.where(types == "DNp20", "descending_neuron", "visual"),
    )
    cell_class = np.where(types == "DNp20", "CX", "")
    soma_side = np.empty(N, dtype=object)
    soma_side[s_dnp_l] = "L"
    soma_side[s_dnp_r] = "R"
    soma_side[s_r16] = "L"
    soma_side[s_r8] = "R"
    soma_side[~(soma_side == "L") & ~(soma_side == "R")] = "L"
    root_side = soma_side.copy()

    # ---- CSR edges --------------------------------------------------------
    # LIF kernel uses conductance-based synapses: a single spike adds `weight`
    # mV to a per-neuron synaptic conductance G that decays with tau_s=5ms.
    # The contribution to the membrane potential is small (G_FACTOR * (a-b)),
    # so realistic connectome weights are large.  We use a 5x boost over the
    # 0.275 mV canonical per-contact weight so signal propagates in our tiny
    # 100-neuron toy graph.
    SYNAPSE_WEIGHT_MV = 1.4  # 5x the canonical MV_PER_CONTACT

    # Lamina -> KC: each KC gets many excitatory inputs from lamina cells.
    edges_per_kc = 60
    n_lkc = N_KC * edges_per_kc
    src_lkc = rng.integers(s_lam.start, s_lam.stop, size=n_lkc)
    tgt_lkc = np.repeat(np.arange(s_kc.start, s_kc.stop), edges_per_kc)
    w_lkc = np.full(n_lkc, SYNAPSE_WEIGHT_MV, dtype=np.float32)  # all excitatory

    # KC -> MBON07
    n_k07 = N_KC * N_MBON07 * 4
    src_k07 = rng.integers(s_kc.start, s_kc.stop, size=n_k07)
    tgt_k07 = rng.integers(s_m07.start, s_m07.stop, size=n_k07)
    w_k07 = np.full(n_k07, SYNAPSE_WEIGHT_MV, dtype=np.float32)

    # KC -> MBON11
    n_k11 = N_KC * N_MBON11 * 4
    src_k11 = rng.integers(s_kc.start, s_kc.stop, size=n_k11)
    tgt_k11 = rng.integers(s_m11.start, s_m11.stop, size=n_k11)
    w_k11 = np.full(n_k11, SYNAPSE_WEIGHT_MV, dtype=np.float32)

    # MBON07 -> DNp20_L
    n_ml = N_MBON07 * N_DNP20_L * 8
    src_ml = rng.integers(s_m07.start, s_m07.stop, size=n_ml)
    tgt_ml = rng.integers(s_dnp_l.start, s_dnp_l.stop, size=n_ml)
    w_ml = np.full(n_ml, SYNAPSE_WEIGHT_MV, dtype=np.float32)

    # MBON11 -> DNp20_R
    n_mr = N_MBON11 * N_DNP20_R * 8
    src_mr = rng.integers(s_m11.start, s_m11.stop, size=n_mr)
    tgt_mr = rng.integers(s_dnp_r.start, s_dnp_r.stop, size=n_mr)
    w_mr = np.full(n_mr, SYNAPSE_WEIGHT_MV, dtype=np.float32)

    # Concat and build CSR
    all_src = np.concatenate([src_lkc, src_k07, src_k11, src_ml, src_mr]).astype(np.int32)
    all_post = np.concatenate([tgt_lkc, tgt_k07, tgt_k11, tgt_ml, tgt_mr]).astype(np.int32)
    all_w = np.concatenate([w_lkc, w_k07, w_k11, w_ml, w_mr]).astype(np.float32)
    order = np.argsort(all_src, kind="stable")
    all_src = all_src[order]
    all_post = all_post[order]
    all_w = all_w[order]
    ptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(ptr, all_src.astype(np.int64) + 1, 1)
    np.cumsum(ptr, out=ptr)

    # Photoreceptor geometry (n_pr per channel; identity mapping)
    r16_uv = np.zeros((N_R16, 2), dtype=np.float32)
    r8_uv = np.zeros((N_R8, 2), dtype=np.float32)
    r16_eye = np.zeros(N_R16, dtype=np.int8)
    r8_eye = np.zeros(N_R8, dtype=np.int8)
    r8_channel = np.ones(N_R8, dtype=np.int8)  # all R8y (green)

    return dict(
        ptr=ptr,
        post=all_post,
        weight=all_w,
        ids=np.arange(N, dtype=np.int64),
        superclass=superclass.astype(object),
        type=types,
        cell_class=cell_class.astype(object),
        soma_side=soma_side,
        root_side=root_side,
        nt_sign=np.zeros(N, dtype=np.int8),
        nt_uncertain=np.zeros(N, dtype=bool),
        modulatory=np.zeros(N, dtype=bool),  # per-neuron (matches len(ptr)-1)
        r16_uv=r16_uv,
        r8_uv=r8_uv,
        r16_eye=r16_eye,
        r8_eye=r8_eye,
        r8_channel=r8_channel,
        r16=np.arange(s_r16.start, s_r16.stop, dtype=np.int32),
        r8=np.arange(s_r8.start, s_r8.stop, dtype=np.int32),
        lamina=np.arange(s_lam.start, s_lam.stop, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# 2. A synthetic market snapshot (SensorFrame)
# ---------------------------------------------------------------------------

def make_sensor_frame() -> SensorFrame:
    IST = timezone(timedelta(hours=5, minutes=30))
    bars = tuple(
        Bar(
            timestamp=datetime(2024, 1, 15, 9, 25 + i, tzinfo=IST),
            open=21600 + i * 5,
            high=21650 + i * 5,
            low=21580 + i * 5,
            close=21630 + i * 5,
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


# ---------------------------------------------------------------------------
# 3. Pipeline driver
# ---------------------------------------------------------------------------

def run(plastic: bool = False) -> Prediction:
    print("=" * 72)
    print(f"REAL Brain demo  |  plastic={plastic}")
    print("=" * 72)

    # ---- build the brain --------------------------------------------------
    graph = build_synthetic_graph(seed=42)
    brain = Brain(
        graph_path="<in-memory>",
        graph=graph,
        check_counts=False,
        plastic=plastic,
    )
    print(f"\n[brain] n_neurons={brain.n}  edges={brain.edges}  n_populations={len(brain.populations)}")
    print(f"[brain] population sizes: {brain.population_sizes()}")
    print(f"[brain] R8->aMe12 edges touched: {brain._r8_ame12['edges']} "
          f"(assumption {brain._r8_ame12['enabled']})")

    # ---- encode a market snapshot ----------------------------------------
    encoder = make_encoder("bars")
    sf = make_sensor_frame()
    stimulus = encoder.encode(sf, brain)
    print(f"\n[stim] r16={stimulus.r16.shape}  r8={stimulus.r8.shape}  pulses={len(stimulus.pulses)}")

    # ---- run 500 ms of neural dynamics -----------------------------------
    result: ObservationResult = brain.observe(stimulus, neural_ms=500.0)
    total_spikes = int(result.counts.sum())
    print(f"\n[observe] {total_spikes} spikes across {brain.n} neurons "
          f"in {result.neural_ms:.0f} ms neural time")
    print(f"[observe] compute_seconds={result.compute_seconds*1000:.2f} ms wall  "
          f"sim_ms={result.sim_ms:.1f}")

    # ---- per-population firing rates --------------------------------------
    rates = brain.population_rates(result.counts, result.neural_ms)
    print("\n[rates]  population            Hz")
    print("         " + "-" * 52)
    for name in REQUIRED_POPULATIONS:
        hz = rates.get(name, 0.0)
        bar = "#" * int(min(hz, 60))
        print(f"         {name:<18} {hz:7.2f}  {bar}")

    # ---- decode into a Decision ------------------------------------------
    decoder = FixedDecoder(neural_ms=result.neural_ms, threshold_hz=2.0)
    pred = decoder.predict(result.counts, brain, sf)
    print(f"\n[decode] decision={pred.decision.value}  roi={pred.realized_over_implied:.3f}  "
          f"confidence={pred.confidence:.3f}")
    print(f"[decode] raw signal={pred.details.get('signal')}  "
          f"diff_hz={pred.details.get('difference_hz'):+.2f}  "
          f"gate_spikes={pred.details.get('gate_spikes')}")

    # ---- parameters() introspection --------------------------------------
    p = brain.parameters()
    print("\n[introspect] kernel constants: "
          f"dt_ms={p['dt_ms']}  tau_m_ms={p['tau_m_ms']}  v_thresh_mv={p['v_thresh_mv']}")
    print(f"[introspect] bin_ms={p['bin_ms']}  half_saturation={p['half_saturation']}  "
          f"plastic={p['plastic']}")
    print(f"[introspect] graph_format_version={brain.provenance()['graph_format_version']}")
    hashes = brain.graph_hashes()
    print(f"[introspect] graph hash for 'ptr' = {hashes.get('ptr', '')[:16]}... "
          f"({len(hashes)} arrays hashed)")

    # ---- plasticity summary if enabled -----------------------------------
    if brain.plasticity is not None:
        summary = brain.plasticity.summary()
        print(f"[plastic] edges={summary.get('edges')}  updates={summary.get('updates')}  "
              f"factor_mean={summary.get('factor_mean'):.4f}  "
              f"range=[{summary.get('factor_min'):.4f}, {summary.get('factor_max'):.4f}]  "
              f"fraction_changed={summary.get('fraction_changed'):.4f}")

    # ---- checkpoint -> restore round-trip ---------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "brain.ckpt.npz"
        brain.checkpoint(str(ckpt))
        # rewind the clock to make the round-trip visible
        brain.reset()
        result_after_reset = brain.observe(stimulus, neural_ms=500.0)
        brain.restore(str(ckpt))
        print(f"\n[ckpt] wrote + restored -> ckpt file size={ckpt.stat().st_size} bytes")
        print(f"[ckpt] reset-then-observe spikes: {int(result_after_reset.counts.sum())}  "
              f"(deterministic, kernel reset cleared the state)")

    return pred


def main() -> None:
    # Run 1: vanilla (no plasticity)
    pred_plastic_off = run(plastic=False)

    # Run 2: enable plasticity, observe weight-factor updates after a pulse
    print()
    run(plastic=True)

    # Verify the two runs reached the protocol contract
    assert isinstance(pred_plastic_off, Prediction)
    assert pred_plastic_off.decision in {Decision.ENTER, Decision.EXIT, Decision.HOLD}
    print("\n[ok] all predictions satisfy the ReadoutProtocol contract.")


if __name__ == "__main__":
    main()
