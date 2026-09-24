"""Brain: the BrainProtocol implementation over the compiled graph.

Populations (name -> int32 index array in graph order):

    R1-R6            R1-R6 photoreceptors mapped to an ommatidial column (stimulus order)
    R8               all mapped R8p and R8y cells (stimulus order for `Stimulus.r8`)
    R8p, R8y         subsets of R8 by channel (2 = R8p blue, 1 = R8y green)
    lamina           L1, L2, L3 and L5 cells
    KC               type starts with KC
    PAM11, PPL101    dopamine neurons of the two compartments used by the plastic arm
    MBON07, MBON11   the corresponding output neurons; MBON = every type starting with MBON
    DNp20_L, DNp20_R type DNp20 by somaSide; DNpe017 both sides
    DN               superclass == descending_neuron (the tbc, sensory_descending and
                     efferent_descending variants are excluded)
    central_complex  class == CX (the MaleCNS `class` column)
    random2000       2,000 neurons sampled without replacement (numpy default_rng, seed
                     20260912) from superclass cb_* excluding photoreceptors and lamina

Stimulus: `r16` and `r8` are luminance in [0, 1] per mapped R1-R6 and R8
cell, low-passed with tau 10 ms (updated per 10 ms bin). Drive: lamina 12 mV
constant, photoreceptors 30 x L / (h + L) mV with half-saturation h
(constructor parameter, settings key neural.half_saturation). Pulses add a
constant to a population's drive for the first `duration_ms` of the
observation. The simulation runs in 10 ms bins so pulses can end mid
observation; the clock continues across observations.

Declared assumption (Brain(r8_ame12_excitatory=True), settings key
neural.r8_ame12_excitatory, default on): the 390 edges from R8
photoreceptors onto the six aMe12 cells are made excitatory on the brain's
own copy of the weights, because R8 drives aMe12 in vivo
(https://doi.org/10.1038/s41586-023-06681-6) while the histamine sign rule
would silence it. Without it no visual signal reaches the Kenyon cells.
Recorded in parameters() and provenance() with the edge count and hashes.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flybrain import kernel as K
from flybrain.connectome.compile import GRAPH_FORMAT_VERSION
from flybrain.connectome.verify import sha256_array
from flybrain.interfaces import REQUIRED_POPULATIONS, ObservationResult, Stimulus
from flybrain.kernel import KERNEL_PARAMETERS, KERNEL_VERSION, Kernel, KernelState
from flybrain.paths import PATHS
from flybrain.plasticity import KCMBONPlasticity, PlasticityConfig

BIN_STEPS = 100
BIN_MS = BIN_STEPS * K.DT_MS
LUMINANCE_TAU_MS = 10.0
LAMINA_DRIVE_MV = 12.0
PHOTORECEPTOR_MAX_MV = 30.0
DEFAULT_HALF_SATURATION = 0.5
RANDOM_SAMPLE_SEED = 20260912
RANDOM_SAMPLE_SIZE = 2000
DESCENDING_SUPERCLASS = "descending_neuron"
CENTRAL_COMPLEX_CLASS = "CX"
CENTRAL_BRAIN_PREFIX = "cb_"
LAMINA_ALL_TYPES = ("L1", "L2", "L3", "L4", "L5")
PHOTORECEPTOR_PREFIXES = ("R1-R6", "R7", "R8")
CHECKPOINT_VERSION = 1

# Declared modeling assumption: R8 photoreceptors drive the aMe12 accessory medulla
# neurons (Nature 2023, https://doi.org/10.1038/s41586-023-06681-6). The transmitter
# sign rule makes every R8 output inhibitory (histamine), which silences aMe12 and,
# through its 191 synapses onto Kenyon cells, the whole downstream brain. With the
# flag on, the R8 to aMe12 edges are made excitatory on the brain's own copy of the
# weights; the compiled graph and its hashes stay pristine.
R8_AME12_SOURCE_PREFIX = "R8"
R8_AME12_TARGET_TYPE = "aMe12"
R8_AME12_CITATION = "https://doi.org/10.1038/s41586-023-06681-6"

EXPECTED_POPULATION_SIZES = {
    "PAM11": 15,
    "PPL101": 2,
    "MBON07": 4,
    "MBON11": 2,
    "R1-R6": 3335,
    "R8": 811,
}

GRAPH_ARRAYS = (
    "ptr",
    "post",
    "weight",
    "ids",
    "superclass",
    "type",
    "cell_class",
    "soma_side",
    "root_side",
    "nt_sign",
    "nt_uncertain",
    "modulatory",
    "r16",
    "r16_uv",
    "r16_eye",
    "r8",
    "r8_uv",
    "r8_eye",
    "r8_channel",
    "lamina",
)


@dataclass(frozen=True)
class EyeMap:
    """Photoreceptor geometry for the sensory encoders.

    `uv_r16[i]` is the (u, v) position in [0, 1] x [0, 1] of the cell
    `r16[i]`, which is also `populations["R1-R6"][i]` and receives
    `Stimulus.r16[i]`. Likewise for R8 with `Stimulus.r8`. `eye` is 0 for
    the left eye and 1 for the right; both eyes cover the full field.
    `r8_channel` is 1 for R8y (green) and 2 for R8p (blue).
    """

    uv_r16: np.ndarray
    eye_r16: np.ndarray
    uv_r8: np.ndarray
    eye_r8: np.ndarray
    r8_channel: np.ndarray
    r16: np.ndarray
    r8: np.ndarray


def load_graph(path: Path | str | None = None) -> dict[str, np.ndarray]:
    path = Path(path) if path else PATHS.graph
    if not path.exists():
        raise FileNotFoundError(f"compiled graph not found at {path} (run: openfly prepare)")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in GRAPH_ARRAYS if k not in z.files]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        return {k: z[k] for k in z.files}


def _startswith(values: np.ndarray, prefix: str) -> np.ndarray:
    return np.char.startswith(values.astype(str), prefix)


def build_populations(g: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    types = g["type"].astype(str)
    superclass = g["superclass"].astype(str)
    soma = g["soma_side"].astype(str)
    cell_class = g["cell_class"].astype(str)

    def idx(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask).astype(np.int32)

    r8 = np.asarray(g["r8"], dtype=np.int32)
    channel = np.asarray(g["r8_channel"], dtype=np.int8)
    pops: dict[str, np.ndarray] = {
        "R1-R6": np.asarray(g["r16"], dtype=np.int32),
        "R8": r8,
        "R8p": r8[channel == 2],
        "R8y": r8[channel == 1],
        "lamina": np.asarray(g["lamina"], dtype=np.int32),
        "KC": idx(_startswith(types, "KC")),
        "PAM11": idx(types == "PAM11"),
        "PPL101": idx(types == "PPL101"),
        "MBON07": idx(types == "MBON07"),
        "MBON11": idx(types == "MBON11"),
        "MBON": idx(_startswith(types, "MBON")),
        "DNp20_L": idx((types == "DNp20") & (soma == "L")),
        "DNp20_R": idx((types == "DNp20") & (soma == "R")),
        "DNpe017": idx(types == "DNpe017"),
        "DN": idx(superclass == DESCENDING_SUPERCLASS),
        "central_complex": idx(cell_class == CENTRAL_COMPLEX_CLASS),
    }
    photoreceptor = np.zeros(len(types), dtype=bool)
    for prefix in PHOTORECEPTOR_PREFIXES:
        photoreceptor |= _startswith(types, prefix)
    lamina_any = np.isin(types, LAMINA_ALL_TYPES)
    pool = idx(_startswith(superclass, CENTRAL_BRAIN_PREFIX) & ~photoreceptor & ~lamina_any)
    rng = np.random.default_rng(RANDOM_SAMPLE_SEED)
    size = min(RANDOM_SAMPLE_SIZE, len(pool))
    sample = rng.choice(pool, size=size, replace=False) if size else np.zeros(0, np.int32)
    pops["random2000"] = np.sort(sample).astype(np.int32)
    return pops


def r8_ame12_edges(g: dict[str, np.ndarray]) -> np.ndarray:
    """Edge indices from any R8 photoreceptor type onto aMe12 cells (sorted)."""
    types = g["type"].astype(str)
    ptr = g["ptr"]
    post = g["post"]
    sources = np.flatnonzero(np.char.startswith(types, R8_AME12_SOURCE_PREFIX))
    is_target = types == R8_AME12_TARGET_TYPE
    out = []
    for i in sources:
        e = np.arange(ptr[i], ptr[i + 1], dtype=np.int64)
        if len(e):
            sel = is_target[post[e]]
            if sel.any():
                out.append(e[sel])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def check_population_sizes(
    pops: dict[str, np.ndarray], expected: dict[str, int] | None = None
) -> None:
    expected = expected or EXPECTED_POPULATION_SIZES
    problems = [
        f"{name} has {len(pops[name])} cells, expected {size}"
        for name, size in expected.items()
        if len(pops.get(name, ())) != size
    ]
    missing = [name for name in REQUIRED_POPULATIONS if name not in pops]
    if missing:
        problems.append("missing populations: " + ", ".join(missing))
    if problems:
        raise ValueError("population check failed: " + "; ".join(problems))


class Brain:
    """BrainProtocol implementation. See the module docstring for conventions."""

    def __init__(
        self,
        graph_path: Path | str | None = None,
        half_saturation: float = DEFAULT_HALF_SATURATION,
        plastic: bool = False,
        plasticity_config: PlasticityConfig | None = None,
        check_counts: bool = True,
        graph: dict[str, np.ndarray] | None = None,
        r8_ame12_excitatory: bool = True,
    ):
        if half_saturation <= 0:
            raise ValueError("half_saturation must be positive")
        self.r8_ame12_excitatory = bool(r8_ame12_excitatory)
        self.graph_path = (
            str(Path(graph_path) if graph_path else PATHS.graph) if graph is None else "<in-memory>"
        )
        g = graph if graph is not None else load_graph(graph_path)
        self._graph = g
        self.n = int(len(g["ptr"]) - 1)
        self.edges = int(len(g["post"]))
        self.half_saturation = float(half_saturation)
        self.plastic = bool(plastic)
        self.types = g["type"].astype(str)
        self.populations = build_populations(g)
        if check_counts:
            check_population_sizes(self.populations)
        is_kc = np.zeros(self.n, dtype=np.uint8)
        is_kc[self.populations["KC"]] = 1
        # A plastic or sign-corrected brain rewrites weights, so it works on its own
        # copy and the compiled graph (and its hashes) stays pristine.
        weight = g["weight"].copy() if (self.plastic or self.r8_ame12_excitatory) else g["weight"]
        self._r8_ame12 = self._apply_r8_ame12(g, weight)
        self.kernel = Kernel(g["ptr"], g["post"], weight, g["modulatory"], is_kc)
        self.r16 = self.populations["R1-R6"]
        self.r8 = self.populations["R8"]
        self.lamina = self.populations["lamina"]
        self._drive = np.zeros(self.n, dtype=np.float64)
        self._bin_counts = np.zeros(self.n, dtype=np.int32)
        self.lp16 = np.zeros(len(self.r16), dtype=np.float64)
        self.lp8 = np.zeros(len(self.r8), dtype=np.float64)
        self.observations = 0
        self.plasticity: KCMBONPlasticity | None = None
        if self.plastic:
            self.plasticity = KCMBONPlasticity(
                self.kernel.ptr,
                self.kernel.post,
                self.kernel.weight,
                self.populations,
                plasticity_config,
            )
        self._hashes: dict[str, str] | None = None

    def _apply_r8_ame12(self, g: dict[str, np.ndarray], weight: np.ndarray) -> dict[str, Any]:
        """Make the R8 to aMe12 edges excitatory on `weight` if the flag is on."""
        edges = r8_ame12_edges(g)
        before = float(g["weight"][edges].sum()) if len(edges) else 0.0
        if self.r8_ame12_excitatory and len(edges):
            weight[edges] = np.abs(weight[edges])
        after = float(weight[edges].sum()) if len(edges) else 0.0
        return {
            "enabled": self.r8_ame12_excitatory,
            "source": f"type starts with {R8_AME12_SOURCE_PREFIX}",
            "target": f"type == {R8_AME12_TARGET_TYPE}",
            "edges": int(len(edges)),
            "edge_index_sha256": sha256_array(edges),
            "weight_sha256": sha256_array(np.ascontiguousarray(weight[edges])),
            "weight_sum_before_mv": before,
            "weight_sum_after_mv": after,
            "citation": R8_AME12_CITATION,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def clock(self) -> int:
        return self.kernel.clock

    @property
    def sim_ms(self) -> float:
        return self.kernel.sim_ms

    @property
    def active_count(self) -> int:
        return self.kernel.state.active_count

    def population_sizes(self) -> dict[str, int]:
        return {name: int(len(idx)) for name, idx in self.populations.items()}

    def eye_map(self) -> EyeMap:
        g = self._graph
        return EyeMap(
            uv_r16=np.asarray(g["r16_uv"], dtype=np.float32),
            eye_r16=np.asarray(g["r16_eye"], dtype=np.int8),
            uv_r8=np.asarray(g["r8_uv"], dtype=np.float32),
            eye_r8=np.asarray(g["r8_eye"], dtype=np.int8),
            r8_channel=np.asarray(g["r8_channel"], dtype=np.int8),
            r16=self.r16,
            r8=self.r8,
        )

    def graph_hashes(self) -> dict[str, str]:
        """SHA-256 per graph array (same digest as data/graph.lock.json)."""
        if self._hashes is None:
            self._hashes = {k: sha256_array(v) for k, v in sorted(self._graph.items())}
        return dict(self._hashes)

    def parameters(self) -> dict[str, Any]:
        return {
            **KERNEL_PARAMETERS,
            "bin_ms": BIN_MS,
            "luminance_tau_ms": LUMINANCE_TAU_MS,
            "lamina_drive_mv": LAMINA_DRIVE_MV,
            "photoreceptor_max_mv": PHOTORECEPTOR_MAX_MV,
            "half_saturation": self.half_saturation,
            "plastic": self.plastic,
            "plasticity": self.plasticity.config.as_dict() if self.plasticity else None,
            "random_sample_seed": RANDOM_SAMPLE_SEED,
            "r8_ame12_excitatory": self.r8_ame12_excitatory,
            "r8_ame12_edges": self._r8_ame12["edges"],
            "r8_ame12_edges_sha256": self._r8_ame12["edge_index_sha256"],
            "r8_ame12_weights_sha256": self._r8_ame12["weight_sha256"],
        }

    def provenance(self) -> dict:
        return {
            "kernel_version": KERNEL_VERSION,
            "graph_format_version": GRAPH_FORMAT_VERSION,
            "graph_path": self.graph_path,
            "graph_hashes": self.graph_hashes(),
            "n": self.n,
            "edges": self.edges,
            "parameters": self.parameters(),
            "population_sizes": self.population_sizes(),
            "population_definitions": {
                "DN": f"superclass == {DESCENDING_SUPERCLASS}",
                "central_complex": f"class == {CENTRAL_COMPLEX_CLASS}",
                "random2000": f"seed {RANDOM_SAMPLE_SEED}, superclass {CENTRAL_BRAIN_PREFIX}* minus photoreceptors and lamina",
                "modulatory": "no postsynaptic effect in the base model",
            },
            "assumptions": {"r8_ame12_excitatory": dict(self._r8_ame12)},
            "plasticity_state": self.plasticity.summary() if self.plasticity else None,
            "clock": self.clock,
            "sim_ms": self.sim_ms,
            "observations": self.observations,
        }

    def population_rates(self, counts: np.ndarray, neural_ms: float) -> dict[str, float]:
        """Mean firing rate in Hz per population for one observation's counts."""
        seconds = neural_ms / 1000.0
        out = {}
        for name, idx in self.populations.items():
            out[name] = float(counts[idx].mean() / seconds) if len(idx) and seconds > 0 else 0.0
        return out

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Fresh mutable state: membranes at rest, clock 0, plastic factors 1."""
        self.kernel.reset()
        self.lp16[:] = 0.0
        self.lp8[:] = 0.0
        self.observations = 0
        if self.plasticity is not None:
            self.plasticity.reset()

    def _luminance(self, values: Any, expected: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(arr) != expected:
            raise ValueError(f"stimulus.{name} has {len(arr)} values, expected {expected}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"stimulus.{name} contains non-finite values")
        return np.clip(arr, 0.0, 1.0)

    def _photoreceptor_drive(self, lp: np.ndarray) -> np.ndarray:
        return PHOTORECEPTOR_MAX_MV * lp / (self.half_saturation + lp)

    def observe(self, stimulus: Stimulus, neural_ms: float) -> ObservationResult:
        t0 = time.perf_counter()
        n_steps = int(round(float(neural_ms) / K.DT_MS))
        if n_steps <= 0:
            raise ValueError("neural_ms must be at least one step (0.1 ms)")
        r16 = self._luminance(stimulus.r16, len(self.r16), "r16")
        r8 = self._luminance(stimulus.r8, len(self.r8), "r8")
        pulses = []
        for pulse in stimulus.pulses or ():
            name, mv, duration = pulse
            if name not in self.populations:
                raise KeyError(f"pulse population {name!r} is not a known population")
            if not (math.isfinite(mv) and math.isfinite(duration)) or duration < 0:
                raise ValueError(
                    f"pulse {name!r}: current and duration must be finite and duration non-negative"
                )
            pulses.append((self.populations[name], float(mv), float(duration)))

        counts = np.zeros(self.n, dtype=np.int32)
        drive = self._drive
        bin_counts = self._bin_counts
        done = 0
        while done < n_steps:
            steps = min(BIN_STEPS, n_steps - done)
            bin_ms = steps * K.DT_MS
            alpha = 1.0 - math.exp(-bin_ms / LUMINANCE_TAU_MS)
            self.lp16 += (r16 - self.lp16) * alpha
            self.lp8 += (r8 - self.lp8) * alpha
            drive.fill(0.0)
            drive[self.lamina] = LAMINA_DRIVE_MV
            drive[self.r16] = self._photoreceptor_drive(self.lp16)
            drive[self.r8] = self._photoreceptor_drive(self.lp8)
            elapsed_ms = done * K.DT_MS
            for idx, mv, duration in pulses:
                if elapsed_ms < duration:
                    drive[idx] += mv
            bin_counts.fill(0)
            self.kernel.set_drive(drive)
            self.kernel.run(steps, bin_counts)
            counts += bin_counts
            if self.plasticity is not None:
                self.plasticity.update(bin_counts, bin_ms)
            done += steps
        self.observations += 1
        return ObservationResult(
            counts=counts,
            neural_ms=n_steps * K.DT_MS,
            sim_ms=self.sim_ms,
            compute_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "kernel_version": KERNEL_VERSION,
            "graph_format_version": GRAPH_FORMAT_VERSION,
            "graph_hashes": self.graph_hashes(),
            "n": self.n,
            "edges": self.edges,
            "parameters": self.parameters(),
            "population_sizes": self.population_sizes(),
            "observations": self.observations,
        }

    def checkpoint(self, path: str) -> None:
        """Save only mutable state plus metadata to a compressed npz."""
        arrays: dict[str, np.ndarray] = {}
        for k, v in self.kernel.state.to_arrays().items():
            arrays["k_" + k] = v
        arrays["lp16"] = self.lp16.copy()
        arrays["lp8"] = self.lp8.copy()
        if self.plasticity is not None:
            for k, v in self.plasticity.state().items():
                arrays["p_" + k] = v
        arrays["meta"] = np.array(json.dumps(self._metadata(), sort_keys=True))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(path)

    def restore(self, path: str) -> None:
        """Load a checkpoint written by `checkpoint`. Validates the metadata."""
        with np.load(Path(path), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays.pop("meta")))
        mine = self._metadata()
        problems = []
        for key in ("checkpoint_version", "kernel_version", "graph_format_version", "n", "edges"):
            if meta.get(key) != mine[key]:
                problems.append(f"{key}: checkpoint {meta.get(key)!r}, brain {mine[key]!r}")
        if meta.get("graph_hashes") != mine["graph_hashes"]:
            changed = sorted(
                k
                for k in set(meta.get("graph_hashes", {})) | set(mine["graph_hashes"])
                if meta.get("graph_hashes", {}).get(k) != mine["graph_hashes"].get(k)
            )
            problems.append("graph arrays differ: " + ", ".join(changed))
        for key in ("half_saturation", "plastic", "plasticity", "r8_ame12_excitatory"):
            if meta.get("parameters", {}).get(key) != mine["parameters"].get(key):
                problems.append(
                    f"parameter {key}: checkpoint {meta.get('parameters', {}).get(key)!r}, brain {mine['parameters'].get(key)!r}"
                )
        if meta.get("population_sizes") != mine["population_sizes"]:
            problems.append("population sizes differ")
        if problems:
            raise ValueError("checkpoint does not match this brain: " + "; ".join(problems))
        kernel_arrays = {k[2:]: v for k, v in arrays.items() if k.startswith("k_")}
        self.kernel.state = KernelState.from_arrays(kernel_arrays, self.kernel.rest)
        self.lp16 = np.array(arrays["lp16"], dtype=np.float64)
        self.lp8 = np.array(arrays["lp8"], dtype=np.float64)
        if self.plasticity is not None:
            self.plasticity.load_state({k[2:]: v for k, v in arrays.items() if k.startswith("p_")})
        self.observations = int(meta.get("observations", 0))
