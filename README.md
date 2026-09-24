# flybrain

A reusable, Numba-JIT accelerated **fruit-fly connectome** simulator —
extracted from [OpenFly](https://github.com/marketcalls/openfly).

* **Zero heavy deps** by default: `numpy` + `numba` + `pillow` only.
* **MaleCNS v1.0 connectome** loadable via `[feather,download]` extras.
* **3 encoders** + **2 readouts** + **optional plasticity**.
* **Closable LIF** kernel (Numba `@njit`), 18M+ spikes/sec.

## Why?

OpenFly is a trading system. The brain model inside it is excellent but
inseparable from the broker / market / front-end code. This package
isolates the brain so it can be reused for **any** sensor → readout task:
trading, robotics, audio classification, RL environments, etc.

## Install

```bash
# Minimum (no MaleCNS, no sklearn):
pip install flybrain

# With everything:
pip install flybrain[feather,readout,download]
```

## 30-second example

```python
import numpy as np
from flybrain import Brain, Stimulus

# Build a tiny synthetic graph (no MaleCNS required)
n = 100
g = dict(
    ptr=np.array([0, n], dtype=np.int64),
    post=np.random.randint(0, n, size=n).astype(np.int32),
    weight=np.where(np.random.rand(n) > 0.5, 0.275, -0.275),
    modulatory=np.zeros(n, dtype=bool),
    type=np.array(["L1"]*90 + ["KC"]*4 + ["MBON07"]*3 + ["MBON11"]*3),
    superclass=np.array(["cb_int"]*n),
    r16_uv=np.zeros((20, 2), dtype=np.float32),
    r8_uv=np.zeros((20, 2), dtype=np.float32),
    r16_eye=np.zeros(20, dtype=np.int8),
    r8_eye=np.zeros(20, dtype=np.int8),
    r8_channel=np.ones(20, dtype=np.int8),
    r16=np.arange(20, dtype=np.int32),
    r8=np.arange(20, dtype=np.int32),
)
brain = Brain(graph_path="<test>", graph=g)
stim = Stimulus(r16=np.random.rand(20), r8=np.random.rand(20))
result = brain.observe(stim, neural_ms=500.0)
print("total spikes:", result.counts.sum())
```

See `examples/` for more.

## Architecture

```
src/flybrain/
├── interfaces.py    # zero-dep contracts (Stimulus, BrainProtocol, SensorFrame)
├── eyemap.py        # EyeMap dataclass + resolver
├── encoders.py      # ChartEncoder, BarsEncoder, FeatureEncoder + make_encoder()
├── kernel.py        # Numba @njit LIF kernel + KernelState
├── plasticity.py    # KC→MBON plasticity (Ormond-style Oja)
├── brain.py          # Brain class (the orchestrator)
├── readout/
│   ├── fixed.py     # FixedDecoder (no training)
│   └── reservoir.py # ReservoirReadout (ridge / logistic; needs [readout])
└── connectome/
    ├── normalize.py # NT sign rules (feather source → arrays; needs [feather])
    ├── compile.py   # arrays → graph.npz (multi-platform)
    ├── sources.py   # MaleCNS v1.0 metadata + SHA256
    ├── download.py  # resumable download (urllib only)
    └── verify.py    # sha256 file/array helpers
```

## Optional dependencies

| extra | adds | when needed |
|---|---|---|
| `feather` | `pandas`, `pyarrow` | `from flybrain.connectome import normalize, compile_graph` |
| `readout` | `scikit-learn`, `scipy`, `joblib` | `from flybrain.readout import ReservoirReadout` |
| `download` | `requests` | `download_source(...)` |
| `dev` | `pytest`, `pytest-cov`, `ruff` | testing |

Without `[feather]`, `compile_graph` still works if `.pre` arrays are
already on disk (you just can't re-derive from raw `.feather` files).

## Tests

```bash
pip install flybrain[feather,readout,dev]
pytest tests/ -v
```

## Citation

* Brain model: Aso & Gorinov — *bioRxiv 2024*
* Connectome: MaleCNS v1.0 — *https://www.malecns.org*
* This extraction: OpenFly — *github.com/marketcalls/openfly*

## Documentation

Detailed Chinese user manual (API contracts, 7 worked scenarios, FAQ):

* [`docs/USAGE.zh.md`](docs/USAGE.zh.md) — 991 lines / 10 chapters

Quick look-up order: 第 3 章 (core concepts) → 第 5 章 (full API) → 第 6 章 (worked scenarios) → 第 10 章 (FAQ).

## License

MIT. The MaleCNS dataset is CC-BY 4.0 (separate license).
