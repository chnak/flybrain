"""Optional dopamine-gated plasticity on Kenyon cell to MBON synapses.

Off by default (Brain(plastic=True) enables it). The rule is a
baseline-centered, dopamine-gated, anti-Hebbian rate rule on every edge from
a Kenyon cell onto MBON07 or MBON11 (about 7,835 edges in MaleCNS v1.0):

    compartments: PAM11 -> MBON07, PPL101 -> MBON11
    gain_m      = contacts(DAN population -> m) / contacts(DAN population -> all MBONs of the compartment)
    x_k         = KC rate trace (Hz, tau 1 s)
    y_m         = MBON rate trace (Hz, tau 1 s)
    d_m         = gain_m x mean DAN rate of the compartment (Hz, trace tau 1 s)
    ybar_m      = slow baseline of y_m (tau 1800 s)
    u_e         = -eta x d_m x x_k x (y_m - ybar_m)        (e = k -> m)
    u_filt_e    <- low-pass of u_e with tau 50 ms
    f_e         <- clip(f_e + u_filt_e x bin_s + (1 - f_e) x bin_s / 1800 s, 0.1, 2.0)
    weight_e    = baseline_e x f_e

so a KC that fires while dopamine is present and its MBON fires above its
long-run baseline is depressed, one whose MBON is below baseline is
potentiated, the change is gated by the compartment's dopamine, and every
factor forgets back to 1 over 1800 s. Updates happen once per 10 ms bin
from the actual spike counts of that bin. Nothing here is random.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MV_PER_CONTACT = 0.275  # built-in


@dataclass(frozen=True)
class PlasticityConfig:
    eta: float = 0.001
    trace_tau_ms: float = 1000.0
    memory_tau_ms: float = 1_800_000.0
    filter_tau_ms: float = 50.0
    factor_min: float = 0.1
    factor_max: float = 2.0
    compartments: tuple[tuple[str, str], ...] = field(
        default=(("PAM11", "MBON07"), ("PPL101", "MBON11"))
    )

    def as_dict(self) -> dict:
        return {
            "eta": self.eta,
            "trace_tau_ms": self.trace_tau_ms,
            "memory_tau_ms": self.memory_tau_ms,
            "filter_tau_ms": self.filter_tau_ms,
            "factor_min": self.factor_min,
            "factor_max": self.factor_max,
            "compartments": [list(c) for c in self.compartments],
        }


class KCMBONPlasticity:
    """Holds the plastic edge set, its baseline weights, traces and factors.

    `weight` is the kernel's live weight array; plastic edges are rewritten
    in place after every update.
    """

    STATE_ARRAYS = ("factor", "x_kc", "y_mbon", "d_mbon", "ybar_mbon", "u_filt")

    def __init__(
        self,
        ptr: np.ndarray,
        post: np.ndarray,
        weight: np.ndarray,
        populations: dict[str, np.ndarray],
        config: PlasticityConfig | None = None,
    ):
        self.config = config or PlasticityConfig()
        self.weight = weight
        n = len(ptr) - 1
        kc = np.asarray(populations["KC"], dtype=np.int64)
        targets: list[int] = []
        target_comp: list[int] = []
        dan_pops: list[np.ndarray] = []
        for ci, (dan, mbon) in enumerate(self.config.compartments):
            for m in np.asarray(populations[mbon], dtype=np.int64):
                targets.append(int(m))
                target_comp.append(ci)
            dan_pops.append(np.asarray(populations[dan], dtype=np.int64))
        self.kc = kc
        self.targets = np.array(targets, dtype=np.int64)
        self.target_comp = np.array(target_comp, dtype=np.int64)
        self.dan_pops = dan_pops
        n_t = len(self.targets)
        target_local = np.full(n, -1, dtype=np.int64)
        target_local[self.targets] = np.arange(n_t)
        kc_local = np.full(n, -1, dtype=np.int64)
        kc_local[kc] = np.arange(len(kc))

        edges: list[np.ndarray] = []
        pre_local: list[np.ndarray] = []
        for i in kc:
            e = np.arange(ptr[i], ptr[i + 1], dtype=np.int64)
            if len(e) == 0:
                continue
            sel = target_local[post[e]] >= 0
            if sel.any():
                edges.append(e[sel])
                pre_local.append(np.full(int(sel.sum()), kc_local[i], dtype=np.int64))
        self.edges = np.concatenate(edges) if edges else np.zeros(0, np.int64)
        self.pre_local = np.concatenate(pre_local) if pre_local else np.zeros(0, np.int64)
        self.post_local = (
            target_local[post[self.edges]] if len(self.edges) else np.zeros(0, np.int64)
        )
        self.baseline = np.array(weight[self.edges], dtype=np.float32, copy=True)

        gain = np.zeros(n_t, dtype=np.float64)
        for ci, dan_idx in enumerate(dan_pops):
            c = np.zeros(n_t, dtype=np.float64)
            for d in dan_idx:
                e = np.arange(ptr[d], ptr[d + 1], dtype=np.int64)
                if len(e) == 0:
                    continue
                tl = target_local[post[e]]
                sel = tl >= 0
                if sel.any():
                    contacts = np.rint(np.abs(weight[e[sel]]) / MV_PER_CONTACT)
                    np.add.at(c, tl[sel], contacts)
            comp = self.target_comp == ci
            total = c[comp].sum()
            if total > 0:
                gain[comp] = c[comp] / total
        self.gain = gain
        self.reset()

    def reset(self) -> None:
        self.factor = np.ones(len(self.edges), dtype=np.float64)
        self.x_kc = np.zeros(len(self.kc), dtype=np.float64)
        self.y_mbon = np.zeros(len(self.targets), dtype=np.float64)
        self.d_mbon = np.zeros(len(self.targets), dtype=np.float64)
        self.ybar_mbon = np.zeros(len(self.targets), dtype=np.float64)
        self.u_filt = np.zeros(len(self.edges), dtype=np.float64)
        self.updates = 0
        self._apply()

    def _apply(self) -> None:
        if len(self.edges):
            self.weight[self.edges] = (self.baseline.astype(np.float64) * self.factor).astype(
                np.float32
            )

    def update(self, bin_counts: np.ndarray, bin_ms: float) -> None:
        """One plasticity step from the spike counts of a bin of `bin_ms` ms."""
        if len(self.edges) == 0 or bin_ms <= 0:
            return
        cfg = self.config
        s = bin_ms / 1000.0
        alpha_t = 1.0 - math.exp(-bin_ms / cfg.trace_tau_ms)
        alpha_m = 1.0 - math.exp(-bin_ms / cfg.memory_tau_ms)
        alpha_f = 1.0 - math.exp(-bin_ms / cfg.filter_tau_ms)
        kc_rate = bin_counts[self.kc].astype(np.float64) / s
        y_rate = bin_counts[self.targets].astype(np.float64) / s
        dan_rate = np.zeros(len(self.dan_pops), dtype=np.float64)
        for ci, dan_idx in enumerate(self.dan_pops):
            if len(dan_idx):
                dan_rate[ci] = bin_counts[dan_idx].astype(np.float64).mean() / s
        d_target = self.gain * dan_rate[self.target_comp]
        self.x_kc += (kc_rate - self.x_kc) * alpha_t
        self.y_mbon += (y_rate - self.y_mbon) * alpha_t
        self.d_mbon += (d_target - self.d_mbon) * alpha_t
        self.ybar_mbon += (self.y_mbon - self.ybar_mbon) * alpha_m
        u = (
            -cfg.eta
            * self.d_mbon[self.post_local]
            * self.x_kc[self.pre_local]
            * (self.y_mbon[self.post_local] - self.ybar_mbon[self.post_local])
        )
        self.u_filt += (u - self.u_filt) * alpha_f
        self.factor += self.u_filt * s
        self.factor += (1.0 - self.factor) * alpha_m
        np.clip(self.factor, cfg.factor_min, cfg.factor_max, out=self.factor)
        self.updates += 1
        self._apply()

    def state(self) -> dict[str, np.ndarray]:
        out = {name: getattr(self, name).copy() for name in self.STATE_ARRAYS}
        out["updates"] = np.array([self.updates], dtype=np.int64)
        return out

    def load_state(self, arrays: dict[str, np.ndarray]) -> None:
        for name in self.STATE_ARRAYS:
            value = np.asarray(arrays[name], dtype=np.float64)
            if value.shape != getattr(self, name).shape:
                raise ValueError(
                    f"plasticity state {name} has shape {value.shape}, expected {getattr(self, name).shape}"
                )
            setattr(self, name, value.copy())
        self.updates = int(np.asarray(arrays.get("updates", [0])).reshape(-1)[0])
        self._apply()

    def summary(self) -> dict:
        f = self.factor
        return {
            "edges": int(len(self.edges)),
            "targets": int(len(self.targets)),
            "updates": int(self.updates),
            "factor_mean": float(f.mean()) if len(f) else 1.0,
            "factor_min": float(f.min()) if len(f) else 1.0,
            "factor_max": float(f.max()) if len(f) else 1.0,
            "fraction_changed": float((np.abs(f - 1.0) > 1e-9).mean()) if len(f) else 0.0,
            "gain": [float(x) for x in self.gain],
        }
