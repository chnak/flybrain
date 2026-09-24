# flybrainer

> A reusable, Numba-JIT accelerated **fruit-fly connectome** simulator
> extracted from [OpenFly](https://github.com/marketcalls/openfly).

`flybrainer` packages the neuroscience core of OpenFly — the Numba-JIT
leaky-integrate-and-fire kernel, the full MaleCNS v1.0 connectome toolchain,
three sensory encoders, two readouts, and dopamine-gated KC→MBON plasticity —
as a self-contained Python library with no broker / market / frontend
dependencies.

| | |
|---|---|
| **Package** | `flybrainer`  (`pip install flybrainer`) |
| **Python**  | `>= 3.12, < 3.14` |
| **Core deps** | `numpy >= 2.2`, `numba >= 0.61`, `pillow >= 10` |
| **Kernel**  | `openfly-lif-1.1` — event-driven LIF, 18M+ spikes/sec, single-threaded exact |
| **Connectome** | MaleCNS v1.0 (166,700 neurons, ~3.5 M synapses) |
| **License** | MIT (the MaleCNS dataset itself is CC-BY 4.0) |

---

## Why flybrainer?

OpenFly is a trading system. The brain model inside it is excellent but
inseparable from broker / market / front-end code. `flybrainer` isolates the
brain so it can be reused for **any** sensor → readout task: trading,
robotics, audio classification, RL environments, neuroscience research, etc.

The boundary is a single file: [`src/flybrainer/interfaces.py`](src/flybrainer/interfaces.py)
defines four `Protocol`s (`BrainProtocol`, `EncoderProtocol`, `ReadoutProtocol`,
`Decision`) and four frozen dataclasses (`Stimulus`, `ObservationResult`,
`SensorFrame`, `Prediction`). The rest of the system codes against those
contracts; you can swap implementations freely.

---

## Install

```bash
# Minimum (no MaleCNS, no sklearn):
pip install flybrainer

# Local dev install (recommended):
pip install -e ".[dev]"

# Everything:
pip install -e ".[feather,readout,download,dev]"
```

| Extra | Adds | Needed for |
|---|---|---|
| `feather` | `pandas`, `pyarrow` | `compile_graph(...)` from raw `.feather` MaleCNS files |
| `readout` | `scikit-learn`, `scipy`, `joblib` | `ReservoirReadout` (ridge / logistic classifiers) |
| `download` | `requests` | Optional HTTP session reuse for downloads |
| `dev` | `pytest`, `pytest-cov`, `ruff` | Tests + lint |

Without `[feather]`, `compile_graph` still works **iff** the intermediate
`.pre` arrays are already on disk (you just can't re-derive them from raw
feather files).

---

## 30-second example

The shortest path that runs is the **stub pipeline** — it touches every
contract class and confirms your install is wired correctly, without
requiring MaleCNS or a compiled graph:

```python
import numpy as np
from flybrainer import (
    REQUIRED_POPULATIONS, BrainProtocol, ObservationResult,
    Stimulus, SensorFrame, Prediction, Decision,
)
from flybrainer.encoders import make_encoder
from flybrainer.readout.fixed import FixedDecoder

class StubBrain(BrainProtocol):
    n = 256
    def __post_init__(self): self.populations = {}
    # ... (see examples/quickstart.py for the full 30-line stub)

brain = StubBrain()
sf = SensorFrame(...)          # build your market snapshot
stimulus = make_encoder("bars").encode(sf, brain)
result = brain.observe(stimulus, neural_ms=200.0)
prediction = FixedDecoder(neural_ms=200.0).predict(result.counts, brain, sf)
print(prediction.decision)     # ENTER / EXIT / HOLD
```

The **real pipeline** (Numba-JIT LIF over a compiled connectome) is one extra
line — just replace the stub with the real `Brain`:

```python
from flybrainer import Brain, load_graph

brain = Brain(graph_path=load_graph("~/.flybrain/graph.npz"))
# then run the same encode → observe → predict pipeline above.
```

---

## Real example: 100-neuron toy graph (no MaleCNS needed)

`examples/real_brain.py` demonstrates the full pipeline against a
hand-built 100-neuron toy graph: signal flows from photoreceptors → lamina
→ KC → MBON07/MBON11 → DNp20_L/R descending neurons, with checkpoint/restore
and plasticity weight updates. Same code paths as the full MaleCNS pipeline,
smaller scale.

```bash
python examples/real_brain.py
```

Expected output highlights:

```
[brain] n_neurons=100  edges=...  n_populations=...
[observe] N spikes across 100 neurons in 500 ms neural time
[rates]   population           Hz
          R1-R6               70.00  ######
          KC                   0.00
          MBON07               ...
[plastic] edges=...  updates=10  factor_mean=1.0000 ...
```

---

## Real example: full MaleCNS v1.0 (166,700 neurons)

`examples/real_malecns_demo.py` runs the same pipeline against the real
MaleCNS connectome. First time it auto-downloads (~1 GB), verifies SHA-256,
and compiles the `.feather` files into a single `graph.npz`:

```bash
pip install -e ".[feather,download]"
python examples/real_malecns_demo.py
```

Compiled `graph.npz` and dataset files live under
`~/.flybrain/` (override with the `FLYBRAINER_DATA_DIR` environment variable).

---

## Public API at a glance

```python
# contracts (stable, dependency-free)
from flybrainer import (
    Stimulus, ObservationResult, Prediction, Decision,
    SensorFrame, BrainProtocol, EncoderProtocol, ReadoutProtocol,
    REQUIRED_POPULATIONS,
)

# photoreceptor geometry
from flybrainer import EyeMap, default_eye_map, resolve_eye_map

# encoders (SensorFrame → Stimulus)
from flybrainer.encoders import (
    make_encoder, ENCODER_NAMES, FEATURE_NAMES,
    ChartEncoder, BarsEncoder, FeatureEncoder,
)

# brain (compile connectome → run dynamics)
from flybrainer import (
    Brain, load_graph, build_populations, r8_ame12_edges,
)

# kernel introspection
from flybrainer import (
    Kernel, KernelState, KERNEL_PARAMETERS, KERNEL_VERSION,
)

# dopamine-gated plasticity
from flybrainer import (
    KCMBONPlasticity, PlasticityConfig, MV_PER_CONTACT,
)

# readouts (counts → Decision)
from flybrainer.readout.fixed      import FixedDecoder
from flybrainer.readout.reservoir  import ReservoirReadout   # needs [readout]
```

`Brain` surface (full list — see
[`src/flybrainer/brain.py`](src/flybrainer/brain.py)):

| Method | Returns | Purpose |
|---|---|---|
| `observe(stim, neural_ms)` | `ObservationResult` | Run dynamics, return spike counts |
| `population_rates(counts, ms)` | `dict[str, float]` | Hz per required population |
| `parameters()` | `dict` | Kernel constants, bin size, plastic flag |
| `provenance()` | `dict` | Kernel/graph versions, R8→aMe12 assumption |
| `graph_hashes()` | `dict[str, str]` | SHA-256 per graph array |
| `checkpoint(path)` / `restore(path)` | `None` | Persist & rewind kernel state |
| `reset()` | `None` | Reset membrane voltages, plasticity traces |
| `eye_map()` | `EyeMap` | Photoreceptor (u, v) geometry |
| `population_sizes()` | `dict[str, int]` | Population → neuron count |

Constructor: `Brain(graph_path=..., graph=..., half_saturation=0.5,
plastic=False, plasticity_config=None, check_counts=True,
r8_ame12_excitatory=True)` — pass **either** `graph_path` to load a compiled
`.npz` **or** `graph=` to inject an in-memory dict.

---

## Architecture

```
src/flybrainer/
├── interfaces.py     # zero-dep contracts (Stimulus, BrainProtocol, SensorFrame …)
├── eyemap.py         # EyeMap dataclass + resolver (left/right eye, UV coords)
├── encoders.py       # ChartEncoder, BarsEncoder, FeatureEncoder, make_encoder()
├── kernel.py         # Numba @njit LIF kernel + KernelState (openfly-lif-1.1)
├── plasticity.py     # KC→MBON dopamine-gated plasticity (Ormond-style Oja)
├── brain.py          # Brain class — orchestrator over Kernel + plasticity
├── paths.py          # data-dir resolver (~/.flybrain or $FLYBRAINER_DATA_DIR)
├── readout/
│   ├── fixed.py      # FixedDecoder — no training, compare L vs R spike counts
│   └── reservoir.py  # ReservoirReadout — ridge / logistic (needs [readout])
└── connectome/
    ├── normalize.py  # MaleCNS feather → arrays (needs [feather])
    ├── compile.py    # arrays → graph.npz (cross-platform)
    ├── sources.py    # MaleCNS v1.0 metadata + SHA-256
    ├── download.py   # resumable download (urllib only)
    └── verify.py     # SHA-256 helpers for files and arrays
```

### Kernel

`openfly-lif-1.1` is an event-driven, single-threaded, exact leaky
integrate-and-fire:

- **dt = 0.1 ms** (10 µs), closed-form integration between events
- Membrane: `τ_m dv/dt = -(v - rest) + I + g - A`  (τ_m = 20 ms)
- Synaptic conductance: `τ_s dg/dt = -g`  (τ_s = 5 ms, conductance-based)
- Adaptation: `τ_a dA/dt = -A`  (τ_a = 200 ms; KC gets +8 mV per spike)
- Axonal delay 1.8 ms (19-slot ring buffer), refractory 2.2 ms
- Kenyon cells rest at −60 mV (others −52 mV), threshold −45 mV
- 10 ms observation bins, sleep/wake list — no approximations on silent neurons

### The R8 → aMe12 assumption

The MaleCNS transmitter sign rule makes every R8 photoreceptor output
inhibitory (histamine), which would silence the aMe12 accessory medulla
cells and, through their 191 synapses onto Kenyon cells, the whole
downstream brain. In vivo (Nature 2023, DOI
[10.1038/s41586-023-06681-6](https://doi.org/10.1038/s41586-023-06681-6))
R8 **excites** aMe12. `Brain(r8_ame12_excitatory=True)` (default) flips
the sign of those 390 edges on its own weight copy; the compiled graph
and its hashes stay pristine. The decision (edge count, weight SHA, both
before/after) is recorded in `provenance()`.

### Plasticity

`Brain(plastic=True)` enables dopamine-gated KC→MBON plasticity on the two
canonical compartments:

| Compartment | DAN → MBON | Edges (MaleCNS v1.0) |
|---|---|---|
| Anteriors + | PAM11 → MBON07 | ~7,835 |
| Posterior  | PPL101 → MBON11 | ~7,835 |

Rule (baseline-centered, anti-Hebbian, dopamine-gated):

```
gain_m   = contacts(DAN→m) / contacts(DAN→all MBONs in compartment)
x_k      = KC rate trace (Hz, τ = 1 s)
y_m      = MBON rate trace (Hz, τ = 1 s)
d_m      = gain_m · mean DAN rate of compartment (τ = 1 s)
ȳ_m      = slow baseline of y_m (τ = 1800 s)
u_e      = -η · d_m · x_k · (y_m - ȳ_m)             # for edge e = k → m
u_filt_e ← low-pass of u_e (τ = 50 ms)
f_e      ← clip(f_e + u_filt_e · bin_s + (1-f_e) · bin_s / 1800s, 0.1, 2.0)
weight_e = baseline_e · f_e
```

Per-edge `factor_e ∈ [0.1, 2.0]`; updates applied every 10 ms from real
spike counts. Fully deterministic, no RNG.

---

## Tests

```bash
pip install -e ".[dev,feather,readout]"
pytest tests/ -v
```

The suite covers import contracts, the encoder/readout protocol, the toy
graph pipeline, and the connectivity compile round-trip. The `[feather]`
extra is only needed for the round-trip tests.

---

## Performance notes

- First `Brain(...)` call costs ~7 s on a cold Numba JIT cache; subsequent
  loads are sub-second (Numba caches by source hash in
  `__pycache__/_numba_cache_*.nbc`).
- Throughput scales linearly with active neurons; on a 2024 laptop the
  full MaleCNS v1.0 graph (~166 k neurons, ~3.5 M synapses) runs at
  ~50 ms wall-clock per 500 ms simulated when driven by photoreceptor
  activity that wakes ~10–15 % of cells per bin.
- `Bin ms = 1.0` (10 × 0.1 ms steps) is the smallest unit; pulses shorter
  than the bin are silently dropped from `ObservationResult.sim_ms` but
  still affect the spike counts of the bin they fall into.

---

## Citation

- **Brain model**: Aso & Gorinov — *bioRxiv 2024*
- **Connectome**: MaleCNS v1.0 — [malecns.org](https://www.malecns.org)
- **This extraction**: OpenFly — [github.com/marketcalls/openfly](https://github.com/marketcalls/openfly)
- **R8 → aMe12 assumption**: [Nature 2023, DOI 10.1038/s41586-023-06681-6](https://doi.org/10.1038/s41586-023-06681-6)

---

## Documentation

- 中文用户手册 (10 章 / 7 个实战场景 / FAQ):
  [`docs/USAGE.zh.md`](docs/USAGE.zh.md)
- Source-of-truth API: [`src/flybrainer/interfaces.py`](src/flybrainer/interfaces.py)
- Run-anytime demos: [`examples/`](examples/)

---

## License

MIT — see [`LICENSE`](LICENSE). The MaleCNS dataset is CC-BY 4.0 (separate
license, see [malecns.org](https://www.malecns.org)).