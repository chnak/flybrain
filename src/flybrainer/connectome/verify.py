"""Integrity checks for the connectome sources and the compiled graph.

Two layers:

1. Sources: each feather file must have the pinned size and SHA-256.
2. Compiled graph: after compilation, the SHA-256 of every array in
   data/graph.npz is written to data/graph.lock.json together with its dtype
   and shape. `verify()` recomputes both layers and reports.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrainer.connectome.sources import SOURCES
from flybrainer.paths import PATHS

LOCK_NAME = "graph.lock.json"
MANIFEST_NAME = "graph-manifest.json"
CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """Digest of dtype, shape and raw bytes so a reinterpretation is detected."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype.str).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def lock_path(graph_path: Path | None = None) -> Path:
    graph_path = Path(graph_path) if graph_path else PATHS.graph
    return graph_path.with_name(LOCK_NAME)


def manifest_path(graph_path: Path | None = None) -> Path:
    graph_path = Path(graph_path) if graph_path else PATHS.graph
    return graph_path.with_name(MANIFEST_NAME)


def array_hashes(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": sha256_array(a), "dtype": str(a.dtype), "shape": list(a.shape)}
        for name, a in sorted(arrays.items())
    }


def graph_hashes(graph_path: Path | None = None) -> dict[str, dict[str, Any]]:
    graph_path = Path(graph_path) if graph_path else PATHS.graph
    with np.load(graph_path, allow_pickle=False) as z:
        return array_hashes({k: z[k] for k in z.files})


def write_lock(graph_path: Path | None = None, extra: dict | None = None) -> Path:
    """Compute and store the per-array hashes of the compiled graph."""
    graph_path = Path(graph_path) if graph_path else PATHS.graph
    lock = {
        "graph": graph_path.name,
        "file_sha256": sha256_file(graph_path),
        "file_bytes": graph_path.stat().st_size,
        "arrays": graph_hashes(graph_path),
    }
    if extra:
        lock.update(extra)
    out = lock_path(graph_path)
    out.write_text(json.dumps(lock, indent=2, sort_keys=True))
    return out


def read_lock(graph_path: Path | None = None) -> dict | None:
    p = lock_path(graph_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def verify_sources(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or PATHS.malecns
    out = []
    for src in SOURCES:
        path = src.path(root)
        row: dict[str, Any] = {
            "name": src.name,
            "present": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "expected_bytes": src.bytes,
            "size_ok": False,
            "sha256_ok": False,
            "sha256": None,
        }
        if row["present"]:
            row["size_ok"] = row["bytes"] == src.bytes
            if row["size_ok"]:
                row["sha256"] = sha256_file(path)
                row["sha256_ok"] = row["sha256"] == src.sha256
        out.append(row)
    return out


def verify_graph(graph_path: Path | None = None) -> dict[str, Any]:
    graph_path = Path(graph_path) if graph_path else PATHS.graph
    report: dict[str, Any] = {
        "graph": str(graph_path),
        "present": graph_path.exists(),
        "lock_present": lock_path(graph_path).exists(),
        "arrays_total": 0,
        "arrays_ok": 0,
        "arrays_bad": [],
        "arrays_missing": [],
        "arrays_unexpected": [],
        "ok": False,
    }
    if not report["present"] or not report["lock_present"]:
        return report
    lock = read_lock(graph_path) or {}
    expected = lock.get("arrays", {})
    actual = graph_hashes(graph_path)
    report["arrays_total"] = len(expected)
    for name, info in expected.items():
        if name not in actual:
            report["arrays_missing"].append(name)
        elif actual[name] != info:
            report["arrays_bad"].append(name)
        else:
            report["arrays_ok"] += 1
    report["arrays_unexpected"] = sorted(set(actual) - set(expected))
    report["ok"] = (
        report["arrays_ok"] == report["arrays_total"]
        and not report["arrays_bad"]
        and not report["arrays_missing"]
        and not report["arrays_unexpected"]
    )
    return report


def verify(root: Path | None = None, graph_path: Path | None = None) -> dict[str, Any]:
    """Verify sources and the compiled graph. Returns a report with an `ok` flag."""
    sources = verify_sources(root)
    graph = verify_graph(graph_path)
    return {
        "sources": sources,
        "sources_ok": all(s["sha256_ok"] for s in sources),
        "graph": graph,
        "ok": all(s["sha256_ok"] for s in sources) and graph["ok"],
    }


def format_report(report: dict[str, Any]) -> str:
    lines = ["Sources:"]
    for s in report["sources"]:
        if not s["present"]:
            state = "missing"
        elif not s["size_ok"]:
            state = f"size mismatch ({s['bytes']} of {s['expected_bytes']} bytes)"
        elif not s["sha256_ok"]:
            state = "sha256 mismatch"
        else:
            state = "ok"
        lines.append(f"  {s['name']}: {state}")
    g = report["graph"]
    lines.append("Compiled graph:")
    if not g["present"]:
        lines.append(f"  {g['graph']}: missing (run: openfly prepare)")
    elif not g["lock_present"]:
        lines.append(f"  {g['graph']}: present but no lock file (run: openfly prepare)")
    else:
        lines.append(f"  arrays verified: {g['arrays_ok']} of {g['arrays_total']}")
        if g["arrays_bad"]:
            lines.append(f"  arrays with changed hash: {', '.join(g['arrays_bad'])}")
        if g["arrays_missing"]:
            lines.append(f"  arrays missing: {', '.join(g['arrays_missing'])}")
        if g["arrays_unexpected"]:
            lines.append(f"  arrays not in lock: {', '.join(g['arrays_unexpected'])}")
    lines.append("Result: " + ("ok" if report["ok"] else "FAILED"))
    return "\n".join(lines)
