"""Eye map: retinal coordinates of the mapped photoreceptors.

The neural package exposes `brain.eye_map()` returning an object with:

    uv_r16      float32 (n_r16, 2) in [0, 1], one row per R1-R6 cell in population order
    eye_r16     int8 (n_r16,), 0 left eye, 1 right eye
    uv_r8       float32 (n_r8, 2)
    eye_r8      int8 (n_r8,)
    r8_channel  int8 (n_r8,), 1 for R8y (green), 2 for R8p (blue)

`u` runs from the periphery of each eye (0) to the midline (1) for the left
eye and the other way round for the right eye, `v` from ventral (0) to dorsal
(1). This module gives a plain dataclass with those fields, a converter that
accepts any object carrying them, and a deterministic fallback map built from
population sizes for brains that do not provide one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FIELDS = ("uv_r16", "eye_r16", "uv_r8", "eye_r8", "r8_channel")


@dataclass(frozen=True)
class EyeMap:
    uv_r16: np.ndarray
    eye_r16: np.ndarray
    uv_r8: np.ndarray
    eye_r8: np.ndarray
    r8_channel: np.ndarray

    @property
    def n_r16(self) -> int:
        return int(self.uv_r16.shape[0])

    @property
    def n_r8(self) -> int:
        return int(self.uv_r8.shape[0])

    def validate(self) -> None:
        if self.uv_r16.ndim != 2 or self.uv_r16.shape[1] != 2:
            raise ValueError("uv_r16 must be (n, 2)")
        if self.uv_r8.ndim != 2 or self.uv_r8.shape[1] != 2:
            raise ValueError("uv_r8 must be (n, 2)")
        if self.eye_r16.shape[0] != self.n_r16 or self.eye_r8.shape[0] != self.n_r8:
            raise ValueError("eye arrays must match uv arrays")
        if self.r8_channel.shape[0] != self.n_r8:
            raise ValueError("r8_channel must match uv_r8")
        for arr in (self.uv_r16, self.uv_r8):
            if arr.size and (np.nanmin(arr) < 0.0 or np.nanmax(arr) > 1.0):
                raise ValueError("uv coordinates must lie in [0, 1]")


def as_eye_map(obj) -> EyeMap:
    """Accept an EyeMap, any object with the five fields, or a dict of arrays."""
    if isinstance(obj, EyeMap):
        obj.validate()
        return obj
    if isinstance(obj, dict):
        values = {k: obj[k] for k in _FIELDS}
    else:
        values = {k: getattr(obj, k) for k in _FIELDS}
    em = EyeMap(
        uv_r16=np.ascontiguousarray(values["uv_r16"], dtype=np.float32),
        eye_r16=np.ascontiguousarray(values["eye_r16"], dtype=np.int8),
        uv_r8=np.ascontiguousarray(values["uv_r8"], dtype=np.float32),
        eye_r8=np.ascontiguousarray(values["eye_r8"], dtype=np.int8),
        r8_channel=np.ascontiguousarray(values["r8_channel"], dtype=np.int8),
    )
    em.validate()
    return em


def _grid_uv(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic hexagonal-looking layout for n cells split over two eyes."""
    n_left = (n + 1) // 2
    n_right = n - n_left
    rng = np.random.default_rng(seed)
    out_uv = np.zeros((n, 2), dtype=np.float32)
    out_eye = np.zeros(n, dtype=np.int8)
    pos = 0
    for eye, count in ((0, n_left), (1, n_right)):
        if count == 0:
            continue
        cols = max(1, int(np.ceil(np.sqrt(count * 16 / 9))))
        rows = max(1, int(np.ceil(count / cols)))
        idx = np.arange(count)
        c = idx % cols
        r = idx // cols
        u = (c + 0.5 + 0.25 * (r % 2)) / cols
        v = (r + 0.5) / rows
        u = u + rng.uniform(-0.15, 0.15, size=count) / cols
        v = v + rng.uniform(-0.15, 0.15, size=count) / rows
        out_uv[pos : pos + count, 0] = np.clip(u, 0.0, 1.0)
        out_uv[pos : pos + count, 1] = np.clip(v, 0.0, 1.0)
        out_eye[pos : pos + count] = eye
        pos += count
    return out_uv, out_eye


def default_eye_map(brain, seed: int = 7) -> EyeMap:
    """Fallback map from population sizes when the brain has no eye_map().

    R8 cells are ordered as the brain lists them: the R8 array covers the
    union of the R8p and R8y populations in ascending neuron index, with
    channel 2 for members of R8p and 1 for members of R8y.
    """
    pops = brain.populations
    n_r16 = int(len(pops["R1-R6"]))
    r8p = np.asarray(pops.get("R8p", np.zeros(0, dtype=np.int32)))
    r8y = np.asarray(pops.get("R8y", np.zeros(0, dtype=np.int32)))
    r8_ids = np.union1d(r8p, r8y)
    n_r8 = int(len(r8_ids))
    uv16, eye16 = _grid_uv(n_r16, seed)
    uv8, eye8 = _grid_uv(n_r8, seed + 1)
    channel = np.where(np.isin(r8_ids, r8p), 2, 1).astype(np.int8)
    return EyeMap(uv_r16=uv16, eye_r16=eye16, uv_r8=uv8, eye_r8=eye8, r8_channel=channel)


def resolve_eye_map(brain, eye_map=None) -> EyeMap:
    """Use the given map, else brain.eye_map(), else the population-size fallback."""
    if eye_map is not None:
        return as_eye_map(eye_map)
    getter = getattr(brain, "eye_map", None)
    if callable(getter):
        try:
            return as_eye_map(getter())
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return default_eye_map(brain)
