"""Event-driven leaky integrate-and-fire kernel (numba, single-threaded, exact).

Model (per neuron i, all voltages in mV, time in ms):

    tau_m dv/dt = -(v - rest_i) + I_i + g - A
    tau_s dg/dt = -g
    tau_a dA/dt = -A

with tau_m = 20, tau_s = 5, tau_a = 200, threshold -45, rest -52 (-60 for
Kenyon cells), axonal delay 1.8 ms (18 steps of 0.1 ms, 19-slot ring buffer),
absolute refractory 2.2 ms (22 steps) during which synaptic input is dropped
and no spike is emitted (the constant drive keeps integrating). A Kenyon cell
spike adds 8 mV to its adaptation A.

Between events the state is advanced in closed form over d steps
(t = d x 0.1 ms, a = exp(-t/20), b = exp(-t/5), c = exp(-t/200)):

    v <- rest + (v - rest) a + I (1 - a) + g (a - b) / 3
    v <- v - A tau_a / (tau_a - 20) (c - a)
    g <- g b
    A <- A c

which is exact for piecewise-constant I. A neuron is kept on the active list
only while v > threshold or I > threshold - rest or I + g > threshold - rest;
outside that region it cannot reach threshold without new input (A only
hyperpolarizes), so sleeping is exact, not an approximation. A sleeping
neuron is woken by an incoming spike or by a drive change (checked at the
start of every call). Each neuron keeps its own `last` clock and is evolved
lazily; every neuron is materialized at the end of a call so the caller may
change the drive.

Spike at step s: count it, enqueue delivery into ring slot (s + 18) mod 19,
reset v = rest, g = 0, refractory until s + 22, and A += 8 for KC. At step
s + 18 every target j is evolved to that step and, if not refractory, gets
g[j] += weight[e]. Modulatory neurons (dopamine, serotonin, octopamine
consensus) deliver nothing in the base model; the plastic arm handles
dopamine separately.

Memory layout: the per-neuron state lives in one float64 (n, 8) array `S`
(one 64-byte record per neuron: v, g, A, last, refractory-until, rest, drive,
active position) so that a random access to a neuron touches one cache line.
Integer fields are stored as exact float64 integers.

Deterministic: no random numbers, no threads, fixed iteration order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

KERNEL_VERSION = "openfly-lif-1.1"

DT_MS = 0.1
TAU_M_MS = 20.0
TAU_S_MS = 5.0
TAU_A_MS = 200.0
V_THRESH_MV = -45.0
V_REST_MV = -52.0
V_REST_KC_MV = -60.0
DELAY_STEPS = 18
RING_SLOTS = 19
REFRACTORY_STEPS = 22
KC_ADAPT_MV = 8.0
TABLE_STEPS = 1024

G_FACTOR = TAU_S_MS / (TAU_M_MS - TAU_S_MS)  # 1/3
A_FACTOR = TAU_A_MS / (TAU_A_MS - TAU_M_MS)  # 10/9

# Columns of the state record.
V = 0
G = 1
A = 2
LAST = 3
REFRAC = 4
REST = 5
DRIVE = 6
POS = 7
RECORD = 8

KERNEL_PARAMETERS = {
    "dt_ms": DT_MS,
    "tau_m_ms": TAU_M_MS,
    "tau_s_ms": TAU_S_MS,
    "tau_a_ms": TAU_A_MS,
    "v_thresh_mv": V_THRESH_MV,
    "v_rest_mv": V_REST_MV,
    "v_rest_kc_mv": V_REST_KC_MV,
    "delay_steps": DELAY_STEPS,
    "ring_slots": RING_SLOTS,
    "refractory_steps": REFRACTORY_STEPS,
    "kc_adapt_mv": KC_ADAPT_MV,
}


def make_exp_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """exp(-d dt / tau) for d in [0, TABLE_STEPS) for the three time constants."""
    t = np.arange(TABLE_STEPS, dtype=np.float64) * DT_MS
    return np.exp(-t / TAU_M_MS), np.exp(-t / TAU_S_MS), np.exp(-t / TAU_A_MS)


def aligned_records(n: int, alignment: int = 64) -> np.ndarray:
    """A zeroed (n, RECORD) float64 array whose rows start on cache-line boundaries."""
    slack = alignment // 8
    buf = np.zeros(n * RECORD + slack, dtype=np.float64)
    offset = (-buf.ctypes.data % alignment) // 8
    return buf[offset : offset + n * RECORD].reshape(n, RECORD)


@njit(cache=True, inline="always")
def _evolve(i, now, S, a_tab, b_tab, c_tab):
    d = now - int(S[i, LAST])
    if d <= 0:
        return
    if d < TABLE_STEPS:
        a = a_tab[d]
        b = b_tab[d]
        c = c_tab[d]
    else:
        t = d * DT_MS
        a = math.exp(-t / TAU_M_MS)
        b = math.exp(-t / TAU_S_MS)
        c = math.exp(-t / TAU_A_MS)
    r = S[i, REST]
    gi = S[i, G]
    vi = r + (S[i, V] - r) * a + S[i, DRIVE] * (1.0 - a) + gi * (a - b) * G_FACTOR
    S[i, G] = gi * b
    ai = S[i, A]
    if ai != 0.0:
        vi -= ai * A_FACTOR * (c - a)
        S[i, A] = ai * c
    S[i, V] = vi
    S[i, LAST] = now


@njit(cache=True, inline="always")
def _eligible(i, S):
    gap = V_THRESH_MV - S[i, REST]
    di = S[i, DRIVE]
    return S[i, V] > V_THRESH_MV or di > gap or di + S[i, G] > gap


@njit(cache=True, inline="always")
def _activate(i, S, active_idx, n_active):
    k = n_active[0]
    active_idx[k] = i
    S[i, POS] = k
    n_active[0] = k + 1


@njit(cache=True, inline="always")
def _deactivate_at(k, S, active_idx, n_active):
    i = active_idx[k]
    last_k = n_active[0] - 1
    j = active_idx[last_k]
    active_idx[k] = j
    S[j, POS] = k
    S[i, POS] = -1.0
    n_active[0] = last_k


@njit(cache=True)
def run_steps(
    n_steps,
    clock,
    ptr,
    post,
    weight,
    modulatory,
    is_kc,
    S,
    ring,
    ring_count,
    active_idx,
    n_active,
    counts,
    a_tab,
    b_tab,
    c_tab,
):
    """Advance the network by n_steps from `clock`. Returns the new clock.

    `S[:, DRIVE]` is the per-neuron constant current (mV) for the whole call.
    `counts` accumulates spikes per neuron. Everything else is state.
    """
    n = S.shape[0]
    # The drive may have changed since the last call: wake eligible sleepers.
    for i in range(n):
        if S[i, POS] < 0.0 and _eligible(i, S):
            _activate(i, S, active_idx, n_active)
    now = clock
    for _ in range(n_steps):
        now += 1
        fnow = float(now)
        slot = now % RING_SLOTS
        # Deliver spikes emitted DELAY_STEPS ago.
        cnt = ring_count[slot]
        for k in range(cnt):
            i = ring[slot, k]
            if modulatory[i] != 0:
                continue
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                _evolve(j, now, S, a_tab, b_tab, c_tab)
                if S[j, REFRAC] <= fnow:
                    S[j, G] += weight[e]
                    if S[j, POS] < 0.0 and _eligible(j, S):
                        _activate(j, S, active_idx, n_active)
        ring_count[slot] = 0
        # Advance active neurons and detect spikes.
        out_slot = (now + DELAY_STEPS) % RING_SLOTS
        k = 0
        while k < n_active[0]:
            i = active_idx[k]
            _evolve(i, now, S, a_tab, b_tab, c_tab)
            if S[i, REFRAC] <= fnow and S[i, V] > V_THRESH_MV:
                counts[i] += 1
                c = ring_count[out_slot]
                ring[out_slot, c] = i
                ring_count[out_slot] = c + 1
                S[i, V] = S[i, REST]
                S[i, G] = 0.0
                S[i, REFRAC] = fnow + REFRACTORY_STEPS
                if is_kc[i] != 0:
                    S[i, A] += KC_ADAPT_MV
                if S[i, DRIVE] > V_THRESH_MV - S[i, REST]:
                    k += 1
                else:
                    _deactivate_at(k, S, active_idx, n_active)
            elif _eligible(i, S):
                k += 1
            else:
                _deactivate_at(k, S, active_idx, n_active)
    # Materialize every neuron so the caller may change the drive.
    for i in range(n):
        _evolve(i, now, S, a_tab, b_tab, c_tab)
    return now


@dataclass
class KernelState:
    """Mutable state. `S` holds the per-neuron records; see the module docstring."""

    S: np.ndarray  # float64 (n, RECORD), rows cache-line aligned
    ring: np.ndarray  # int32 (RING_SLOTS, n)
    ring_count: np.ndarray  # int64 (RING_SLOTS,)
    active_idx: np.ndarray  # int32 (n,)
    n_active: np.ndarray  # int64 (1,)
    clock: int

    @classmethod
    def fresh(cls, rest: np.ndarray) -> KernelState:
        n = len(rest)
        S = aligned_records(n)
        S[:, V] = rest
        S[:, REST] = rest
        S[:, POS] = -1.0
        return cls(
            S=S,
            ring=np.zeros((RING_SLOTS, n), np.int32),
            ring_count=np.zeros(RING_SLOTS, np.int64),
            active_idx=np.zeros(n, np.int32),
            n_active=np.zeros(1, np.int64),
            clock=0,
        )

    @property
    def n(self) -> int:
        return self.S.shape[0]

    @property
    def active_count(self) -> int:
        return int(self.n_active[0])

    @property
    def v(self) -> np.ndarray:
        return self.S[:, V]

    @property
    def g(self) -> np.ndarray:
        return self.S[:, G]

    @property
    def adapt(self) -> np.ndarray:
        return self.S[:, A]

    @property
    def refrac_until(self) -> np.ndarray:
        return self.S[:, REFRAC]

    @property
    def last(self) -> np.ndarray:
        return self.S[:, LAST]

    @property
    def drive(self) -> np.ndarray:
        return self.S[:, DRIVE]

    @property
    def active_pos(self) -> np.ndarray:
        return self.S[:, POS]

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Compact representation: only the used ring entries and active list."""
        used = [self.ring[s, : self.ring_count[s]] for s in range(RING_SLOTS)]
        return {
            "v": self.S[:, V].copy(),
            "g": self.S[:, G].copy(),
            "adapt": self.S[:, A].copy(),
            "refrac_until": self.S[:, REFRAC].astype(np.int64),
            "last": self.S[:, LAST].astype(np.int64),
            "ring_flat": np.concatenate(used).astype(np.int32) if used else np.zeros(0, np.int32),
            "ring_count": self.ring_count.copy(),
            "active_idx": self.active_idx[: self.n_active[0]].copy(),
            "clock": np.array([self.clock], dtype=np.int64),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], rest: np.ndarray) -> KernelState:
        n = len(rest)
        v = np.asarray(arrays["v"], dtype=np.float64)
        if len(v) != n:
            raise ValueError(f"checkpoint has {len(v)} neurons, graph has {n}")
        state = cls.fresh(rest)
        S = state.S
        S[:, V] = v
        S[:, G] = np.asarray(arrays["g"], dtype=np.float64)
        S[:, A] = np.asarray(arrays["adapt"], dtype=np.float64)
        S[:, REFRAC] = np.asarray(arrays["refrac_until"], dtype=np.float64)
        S[:, LAST] = np.asarray(arrays["last"], dtype=np.float64)
        ring_count = np.array(arrays["ring_count"], dtype=np.int64)
        flat = np.asarray(arrays["ring_flat"], dtype=np.int32)
        offset = 0
        for s in range(RING_SLOTS):
            c = int(ring_count[s])
            state.ring[s, :c] = flat[offset : offset + c]
            offset += c
        state.ring_count = ring_count
        active = np.asarray(arrays["active_idx"], dtype=np.int32)
        state.active_idx[: len(active)] = active
        S[active, POS] = np.arange(len(active), dtype=np.float64)
        state.n_active[0] = len(active)
        state.clock = int(np.asarray(arrays["clock"]).reshape(-1)[0])
        return state


class Kernel:
    """Graph plus state plus the jitted stepper."""

    def __init__(
        self,
        ptr: np.ndarray,
        post: np.ndarray,
        weight: np.ndarray,
        modulatory: np.ndarray,
        is_kc: np.ndarray,
        rest: np.ndarray | None = None,
    ):
        self.ptr = np.ascontiguousarray(ptr, dtype=np.int64)
        self.post = np.ascontiguousarray(post, dtype=np.int32)
        self.weight = np.ascontiguousarray(weight, dtype=np.float32)
        self.modulatory = np.ascontiguousarray(modulatory, dtype=np.uint8)
        self.is_kc = np.ascontiguousarray(is_kc, dtype=np.uint8)
        n = len(self.ptr) - 1
        if rest is None:
            rest = np.where(self.is_kc != 0, V_REST_KC_MV, V_REST_MV)
        self.rest = np.ascontiguousarray(rest, dtype=np.float64)
        if len(self.rest) != n or len(self.modulatory) != n or len(self.is_kc) != n:
            raise ValueError("per-neuron arrays must have length n = len(ptr) - 1")
        if len(self.post) != int(self.ptr[-1]) or len(self.weight) != len(self.post):
            raise ValueError("post and weight must have length ptr[-1]")
        self.n = n
        self.a_tab, self.b_tab, self.c_tab = make_exp_tables()
        self.state = KernelState.fresh(self.rest)

    def reset(self) -> None:
        self.state = KernelState.fresh(self.rest)

    def set_drive(self, drive: np.ndarray) -> None:
        """Set the per-neuron constant current (mV) used by the next `run`."""
        if len(drive) != self.n:
            raise ValueError("drive must have length n")
        self.state.S[:, DRIVE] = drive

    def run(self, n_steps: int, counts: np.ndarray) -> int:
        """Advance by n_steps under the current drive. Returns the clock."""
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if len(counts) != self.n or counts.dtype != np.int32:
            raise ValueError("counts must be int32 of length n")
        s = self.state
        s.clock = int(
            run_steps(
                int(n_steps),
                s.clock,
                self.ptr,
                self.post,
                self.weight,
                self.modulatory,
                self.is_kc,
                s.S,
                s.ring,
                s.ring_count,
                s.active_idx,
                s.n_active,
                counts,
                self.a_tab,
                self.b_tab,
                self.c_tab,
            )
        )
        return s.clock

    @property
    def clock(self) -> int:
        return self.state.clock

    @property
    def sim_ms(self) -> float:
        return self.state.clock * DT_MS
