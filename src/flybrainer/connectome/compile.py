"""Compile the normalized MaleCNS graph into data/graph.npz.

Arrays written (all in graph order, index = position of the neuron in the
sorted retained bodyId list):

    ptr          int64   (n + 1)   CSR row pointer by presynaptic neuron
    post         int32   (e,)      postsynaptic index per edge, sorted within row
    weight       float32 (e,)      contacts x sign x 0.275 mV
    ids          int64   (n,)      bodyId
    superclass, type, instance, cell_class, soma_side, root_side   unicode (n,)
    nt_sign      int8    (n,)      +1 or -1 per presynaptic neuron
    nt_uncertain bool    (n,)      sign was a proxy (ambiguous, unclear, missing)
    modulatory   uint8   (n,)      1 if consensus is only dopamine/serotonin/octopamine
    r16          int32   (n_r16,)  R1-R6 cells that map to an ommatidial column
    r16_uv       float32 (n_r16, 2) column position in [0, 1] x [0, 1], per eye
    r16_eye      int8    (n_r16,)  0 left, 1 right (rootSide)
    r16_column   int32   (n_r16,)  hex1 * 64 + hex2 of the assigned column
    r8, r8_uv, r8_eye, r8_column   the same for R8p and R8y cells
    r8_channel   int8    (n_r8,)   1 for R8y (green), 2 for R8p (blue)
    lamina       int32   (n_lam,)  L1, L2, L3 and L5 cells

Column assignment. An R1-R6 cell has no column annotation of its own, so
its column is the modal `assignedOlHex1/2` column over its outgoing edges
to lamina targets (L1 to L5) that carry a column, weighted by contact
count. R8 cells use the modal column over outgoing edges to any
column-annotated target. Hex coordinates become Cartesian with
x = h1 - 0.5 h2 and y = sqrt(3)/2 h2, and each eye is min-max normalized
to [0, 1] in both u and v independently, so both eyes project the full
stimulus field (deliberate: lateral asymmetry then means something in the
stimulus, not in the layout). R8 cells reuse the R1-R6 bounds of the same
eye and are clipped to [0, 1].
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.connectome import normalize as nz
from flybrain.connectome.sources import DATASET, LICENSE, SOURCES, source
from flybrain.connectome.verify import manifest_path, write_lock
from flybrain.paths import PATHS

GRAPH_FORMAT_VERSION = 1

LAMINA_TYPES = ("L1", "L2", "L3", "L5")
R16_COLUMN_TARGET_TYPES = ("L1", "L2", "L3", "L4", "L5")
R8_TYPES = ("R8p", "R8y")
R8_CHANNEL = {"R8y": 1, "R8p": 2}
COLUMN_BASE = 64  # column id = hex1 * COLUMN_BASE + hex2

Progress = Callable[[str], None]


def _say(progress: Progress | None, msg: str) -> None:
    if progress is not None:
        progress(msg)


def peak_memory_bytes() -> int | None:
    """Peak resident memory of this process, or None if unavailable."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if ok else None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss * 1024) if sys.platform.startswith("linux") else int(rss)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Column geometry
# ---------------------------------------------------------------------------


def hex_to_xy(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axial hex coordinates to Cartesian: x = h1 - 0.5 h2, y = sqrt(3)/2 h2."""
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    return h1 - 0.5 * h2, (math.sqrt(3.0) / 2.0) * h2


def column_ids(hex1: np.ndarray, hex2: np.ndarray) -> np.ndarray:
    """Encode per-neuron hex coordinates as one int (-1 where absent)."""
    hex1 = np.asarray(hex1, dtype=np.float64)
    hex2 = np.asarray(hex2, dtype=np.float64)
    has = np.isfinite(hex1) & np.isfinite(hex2)
    col = np.full(len(hex1), -1, dtype=np.int64)
    col[has] = hex1[has].astype(np.int64) * COLUMN_BASE + hex2[has].astype(np.int64)
    return col


def decode_column(col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col = np.asarray(col, dtype=np.int64)
    return col // COLUMN_BASE, col % COLUMN_BASE


def modal_column(
    pre: np.ndarray,
    post: np.ndarray,
    contacts: np.ndarray,
    source_mask: np.ndarray,
    col: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Modal target column per source neuron, weighted by contact count.

    Returns an (n,) int64 array with the column id, -1 for sources with no
    column-annotated target. Ties go to the smallest column id.
    """
    n = len(col)
    out = np.full(n, -1, dtype=np.int64)
    sel = source_mask[pre] & target_mask[post] & (col[post] >= 0)
    if not sel.any():
        return out
    p = pre[sel].astype(np.int64)
    c = col[post[sel]]
    w = np.asarray(contacts[sel], dtype=np.float64)
    span = COLUMN_BASE * COLUMN_BASE
    key = p * span + c
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=w)
    up = uk // span
    uc = uk % span
    order = np.lexsort((uc, -sums, up))
    up, uc = up[order], uc[order]
    first = np.concatenate(([True], up[1:] != up[:-1]))
    out[up[first]] = uc[first]
    return out


def eye_index(root_side: np.ndarray) -> np.ndarray:
    """0 for a left root, 1 for a right root, -1 otherwise."""
    side = np.asarray(root_side).astype(str)
    eye = np.full(len(side), -1, dtype=np.int8)
    eye[side == "L"] = 0
    eye[side == "R"] = 1
    return eye


def normalize_uv(
    x: np.ndarray, y: np.ndarray, eye: np.ndarray, bounds: dict[int, dict[str, float]] | None = None
) -> tuple[np.ndarray, dict[int, dict[str, float]], int]:
    """Min-max normalize (x, y) to [0, 1] per eye.

    When `bounds` is given (from the R1-R6 cells of the same eye) it is reused
    and points outside are clipped; the number of clipped coordinates is
    returned. Otherwise the bounds are computed from the points themselves.
    """
    uv = np.zeros((len(x), 2), dtype=np.float32)
    used: dict[int, dict[str, float]] = {}
    clipped = 0
    for e in (0, 1):
        m = eye == e
        if not m.any():
            continue
        if bounds is not None and e in bounds:
            b = bounds[e]
        else:
            b = {
                "x_min": float(x[m].min()),
                "x_range": float(max(x[m].max() - x[m].min(), 1e-9)),
                "y_min": float(y[m].min()),
                "y_range": float(max(y[m].max() - y[m].min(), 1e-9)),
            }
        u = (x[m] - b["x_min"]) / b["x_range"]
        v = (y[m] - b["y_min"]) / b["y_range"]
        clipped += int(((u < 0) | (u > 1) | (v < 0) | (v > 1)).sum())
        uv[m, 0] = np.clip(u, 0.0, 1.0)
        uv[m, 1] = np.clip(v, 0.0, 1.0)
        used[e] = b
    return uv, used, clipped


def photoreceptor_geometry(
    pre: np.ndarray,
    post: np.ndarray,
    contacts: np.ndarray,
    types: np.ndarray,
    root_side: np.ndarray,
    hex1: np.ndarray,
    hex2: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map R1-R6 and R8 cells to eye coordinates. Returns (arrays, stats)."""
    types = np.asarray(types).astype(str)
    col = column_ids(hex1, hex2)
    has_col = col >= 0
    eye = eye_index(root_side)

    r16_mask = types == "R1-R6"
    r16_targets = np.isin(types, R16_COLUMN_TARGET_TYPES) & has_col
    r16_col = modal_column(pre, post, contacts, r16_mask, col, r16_targets)
    r16 = np.flatnonzero(r16_mask & (r16_col >= 0) & (eye >= 0)).astype(np.int32)
    h1, h2 = decode_column(r16_col[r16])
    x16, y16 = hex_to_xy(h1, h2)
    eye16 = eye[r16]
    uv16, bounds, _ = normalize_uv(x16, y16, eye16)

    r8_mask = np.isin(types, R8_TYPES)
    r8_col = modal_column(pre, post, contacts, r8_mask, col, has_col)
    r8 = np.flatnonzero(r8_mask & (r8_col >= 0) & (eye >= 0) & np.isin(eye, list(bounds))).astype(
        np.int32
    )
    h1, h2 = decode_column(r8_col[r8])
    x8, y8 = hex_to_xy(h1, h2)
    eye8 = eye[r8]
    uv8, _, clipped8 = normalize_uv(x8, y8, eye8, bounds)
    channel8 = np.array([R8_CHANNEL[t] for t in types[r8]], dtype=np.int8)

    arrays = {
        "r16": r16,
        "r16_uv": uv16.astype(np.float32),
        "r16_eye": eye16.astype(np.int8),
        "r16_column": r16_col[r16].astype(np.int32),
        "r8": r8,
        "r8_uv": uv8.astype(np.float32),
        "r8_eye": eye8.astype(np.int8),
        "r8_column": r8_col[r8].astype(np.int32),
        "r8_channel": channel8,
    }
    stats = {
        "r16_total": int(r16_mask.sum()),
        "r16_mapped": int(len(r16)),
        "r16_left": int((eye16 == 0).sum()),
        "r16_right": int((eye16 == 1).sum()),
        "r16_columns": int(len(np.unique(r16_col[r16]))),
        "r8p_total": int((types == "R8p").sum()),
        "r8y_total": int((types == "R8y").sum()),
        "r8_mapped": int(len(r8)),
        "r8_left": int((eye8 == 0).sum()),
        "r8_right": int((eye8 == 1).sum()),
        "r8p_mapped": int((channel8 == 2).sum()),
        "r8y_mapped": int((channel8 == 1).sum()),
        "r8_clipped_coordinates": int(clipped8),
        "column_annotated_neurons": int(has_col.sum()),
        "eye_bounds": {("left" if e == 0 else "right"): b for e, b in bounds.items()},
        "r16_column_targets": list(R16_COLUMN_TARGET_TYPES),
    }
    return arrays, stats


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def compile_graph(
    root: Path | None = None,
    graph_path: Path | None = None,
    progress: Progress | None = None,
    check_expected: bool = True,
) -> dict[str, Any]:
    """Normalize the sources and write graph.npz, the manifest and the lock.

    Returns the manifest dict. Raises AssertionError if the counts differ
    from the MaleCNS v1.0 expectations and `check_expected` is True.
    """
    root = root or PATHS.malecns
    graph_path = Path(graph_path) if graph_path else PATHS.graph
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    timing: dict[str, float] = {}

    _say(progress, "Reading annotations")
    t0 = time.perf_counter()
    ann = nz.load_annotations(source("annotations").path(root))
    nodes = nz.select_nodes(ann)
    arrays = nz.node_arrays(nodes)
    ids = arrays["ids"]
    n = int(len(ids))
    timing["annotations_s"] = round(time.perf_counter() - t0, 2)
    _say(progress, f"  retained neurons: {n:,} of {len(ann):,} bodies")

    _say(progress, "Reading neurotransmitters")
    t0 = time.perf_counter()
    nt = nz.load_neurotransmitters(ids, source("neurotransmitters").path(root))
    timing["neurotransmitters_s"] = round(time.perf_counter() - t0, 2)
    nt_values = nt["nt_values"]
    nt_counts = {str(k): int(v) for k, v in nt_values.fillna("<missing>").value_counts().items()}
    _say(
        progress,
        f"  sign +1: {int((nt['nt_sign'] == 1).sum()):,}, sign -1: {int((nt['nt_sign'] == -1).sum()):,}, uncertain: {int(nt['nt_uncertain'].sum()):,}, modulatory: {int(nt['modulatory'].sum()):,}",
    )

    _say(progress, "Reading weights (streaming record batches)")
    t0 = time.perf_counter()
    pre, post, contacts, edge_stats = nz.load_edges(ids, source("weights").path(root), progress)
    timing["edges_s"] = round(time.perf_counter() - t0, 2)
    _say(progress, f"  edges: {edge_stats['edges']:,}, contacts: {edge_stats['contacts']:,}")
    if check_expected:
        nz.assert_expected(n, edge_stats["edges"], edge_stats["contacts"])

    _say(progress, "Building CSR and weights")
    t0 = time.perf_counter()
    ptr = nz.build_csr(pre, post, n)
    sign = nt["nt_sign"]
    weight = (
        contacts.astype(np.float64) * sign[pre].astype(np.float64) * nz.MV_PER_CONTACT
    ).astype(np.float32)
    timing["csr_s"] = round(time.perf_counter() - t0, 2)

    _say(progress, "Mapping photoreceptors to eye coordinates")
    t0 = time.perf_counter()
    geometry, geo_stats = photoreceptor_geometry(
        pre, post, contacts, arrays["type"], arrays["root_side"], arrays["hex1"], arrays["hex2"]
    )
    lamina = np.flatnonzero(np.isin(arrays["type"], LAMINA_TYPES)).astype(np.int32)
    timing["geometry_s"] = round(time.perf_counter() - t0, 2)
    _say(
        progress,
        f"  R1-R6 mapped: {geo_stats['r16_mapped']} of {geo_stats['r16_total']}, R8 mapped: {geo_stats['r8_mapped']} of {geo_stats['r8p_total'] + geo_stats['r8y_total']}, lamina: {len(lamina)}",
    )

    out = {
        "ptr": ptr,
        "post": post.astype(np.int32),
        "weight": weight,
        "ids": ids,
        "superclass": arrays["superclass"],
        "type": arrays["type"],
        "instance": arrays["instance"],
        "cell_class": arrays["cell_class"],
        "soma_side": arrays["soma_side"],
        "root_side": arrays["root_side"],
        "nt_sign": sign.astype(np.int8),
        "nt_uncertain": nt["nt_uncertain"].astype(bool),
        "modulatory": nt["modulatory"].astype(np.uint8),
        "lamina": lamina,
        **geometry,
    }
    del pre, contacts

    _say(progress, f"Writing {graph_path}")
    t0 = time.perf_counter()
    _atomic_savez(graph_path, out)
    timing["write_s"] = round(time.perf_counter() - t0, 2)

    type_counts = {
        "R1-R6": geo_stats["r16_total"],
        "R8p": geo_stats["r8p_total"],
        "R8y": geo_stats["r8y_total"],
        "lamina_L1_L2_L3_L5": int(len(lamina)),
        "KC": int(np.char.startswith(arrays["type"], "KC").sum()),
        "PAM11": int((arrays["type"] == "PAM11").sum()),
        "PPL101": int((arrays["type"] == "PPL101").sum()),
        "MBON07": int((arrays["type"] == "MBON07").sum()),
        "MBON11": int((arrays["type"] == "MBON11").sum()),
        "MBON": int(np.char.startswith(arrays["type"], "MBON").sum()),
        "DNp20": int((arrays["type"] == "DNp20").sum()),
        "DNpe017": int((arrays["type"] == "DNpe017").sum()),
        "descending_neuron": int((arrays["superclass"] == "descending_neuron").sum()),
        "central_complex_class_CX": int((arrays["cell_class"] == "CX").sum()),
    }
    superclass_counts = {
        str(k): int(v)
        for k, v in zip(*np.unique(arrays["superclass"], return_counts=True), strict=True)
    }

    timing["total_s"] = round(time.perf_counter() - t_start, 2)
    manifest: dict[str, Any] = {
        "format_version": GRAPH_FORMAT_VERSION,
        "dataset": DATASET,
        "license": LICENSE,
        "sources": [
            {"role": s.role, "name": s.name, "bytes": s.bytes, "sha256": s.sha256} for s in SOURCES
        ],
        "policies": {
            "nodes": "superclass non-null and non-empty, status != Glia",
            "edges": "all edges between retained bodies, including self-connections and weight-1 edges",
            "weight": f"contacts x sign x {nz.MV_PER_CONTACT} mV (float32)",
            "sign": "+1 acetylcholine without gaba/glutamate/histamine; -1 gaba/glutamate/histamine without acetylcholine; else +1 flagged uncertain",
            "modulatory": "consensus tokens are only dopamine, serotonin or octopamine",
            "r16_column": "modal assignedOlHex column over outgoing edges to L1-L5 targets, weighted by contacts",
            "r8_column": "modal assignedOlHex column over outgoing edges to any column-annotated target, weighted by contacts",
            "uv": "x = h1 - 0.5 h2, y = sqrt(3)/2 h2, min-max to [0, 1] per eye in both axes; R8 reuses the R1-R6 bounds of its eye and is clipped",
        },
        "counts": {
            "bodies_in_annotations": int(len(ann)),
            "neurons": n,
            "edges": edge_stats["edges"],
            "contacts": edge_stats["contacts"],
            "self_edges": edge_stats["self_edges"],
            "weight_one_edges": edge_stats["weight_one_edges"],
            "duplicates_merged": edge_stats["duplicates_merged"],
            "weight_rows_read": edge_stats["rows_read"],
            "expected": dict(nz.EXPECTED),
        },
        "neurotransmitters": {
            "sign_plus": int((sign == 1).sum()),
            "sign_minus": int((sign == -1).sum()),
            "uncertain": int(nt["nt_uncertain"].sum()),
            "modulatory": int(nt["modulatory"].sum()),
            "missing_from_nt_file": int((~nt["nt_present"]).sum()),
            "consensus_values": nt_counts,
        },
        "photoreceptors": geo_stats,
        "type_counts": type_counts,
        "superclass_counts": superclass_counts,
        "timing": timing,
        "memory": {"peak_working_set_bytes": peak_memory_bytes()},
        "caveats": [
            "Signs for cells whose consensus neurotransmitter is unclear, missing or modulator-only are +1 and flagged nt_uncertain; this is a declared proxy.",
            "Modulatory neurons (dopamine, serotonin, octopamine consensus) have no postsynaptic effect in the base model; the plastic arm handles dopamine separately.",
            f"{geo_stats['r16_total'] - geo_stats['r16_mapped']} R1-R6 cells have no column-annotated lamina target and are not part of the R1-R6 stimulus population.",
            "Both eyes are min-max normalized to the full [0, 1] field independently; the two eyes see the same stimulus and there is no binocular overlap model.",
            f"R8 coordinates use the R1-R6 bounds of the same eye; {geo_stats['r8_clipped_coordinates']} R8 coordinate values were clipped to [0, 1].",
            "Column ties in the modal assignment go to the smallest column id.",
        ],
    }
    manifest_path(graph_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _say(progress, "Writing lock file")
    write_lock(graph_path, {"format_version": GRAPH_FORMAT_VERSION})
    _say(progress, f"Done in {timing['total_s']} s")
    return manifest


def read_manifest(graph_path: Path | None = None) -> dict[str, Any] | None:
    p = manifest_path(graph_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())
