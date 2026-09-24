"""Node policy, edge policy and neurotransmitter sign rule.

Node policy: every body whose superclass is non-null and non-empty and whose
status is not Glia. Edge policy: every edge between retained bodies,
including self-connections and weight-1 edges.

Expected result on MaleCNS v1.0 (minconf 0.5): 166,700 neurons, 25,582,938
directed edges and 124,177,617 synaptic contacts. These are asserted.

Sign rule (per presynaptic neuron, from `consensus_nt`, comma-separated
tokens tolerated):

- +1 if the tokens contain acetylcholine and none of gaba, glutamate,
  histamine;
- -1 if the tokens contain any of gaba, glutamate, histamine and not
  acetylcholine;
- otherwise (ambiguous, unclear, missing, modulator-only) +1 and flagged
  uncertain. A cell whose tokens are only dopamine, serotonin or octopamine
  is additionally flagged modulatory.

Weight = contacts x sign x 0.275 mV (float32).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from flybrainer.connectome.sources import source

EXPECTED = {"neurons": 166_700, "edges": 25_582_938, "contacts": 124_177_617}

MV_PER_CONTACT = 0.275

EXCITATORY = frozenset({"acetylcholine"})
INHIBITORY = frozenset({"gaba", "glutamate", "histamine"})
MODULATORY = frozenset({"dopamine", "serotonin", "octopamine"})

ANNOTATION_COLUMNS = (
    "bodyId",
    "superclass",
    "status",
    "type",
    "instance",
    "class",
    "somaSide",
    "rootSide",
    "assignedOlHex1",
    "assignedOlHex2",
)

Progress = Callable[[str], None]


def _say(progress: Progress | None, msg: str) -> None:
    if progress is not None:
        progress(msg)


# ---------------------------------------------------------------------------
# Neurotransmitter sign rule
# ---------------------------------------------------------------------------


def tokenize_nt(value: Any) -> frozenset[str]:
    """Lower-cased, comma-separated tokens of a consensus_nt value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return frozenset()
    text = str(value).strip().lower()
    if not text:
        return frozenset()
    return frozenset(t.strip() for t in text.replace(";", ",").split(",") if t.strip())


def sign_from_nt(value: Any) -> tuple[int, bool, bool]:
    """Return (sign, uncertain, modulatory) for one consensus_nt value."""
    tokens = tokenize_nt(value)
    excitatory = bool(tokens & EXCITATORY)
    inhibitory = bool(tokens & INHIBITORY)
    if excitatory and not inhibitory:
        return 1, False, False
    if inhibitory and not excitatory:
        return -1, False, False
    modulatory = bool(tokens) and tokens <= MODULATORY
    return 1, True, modulatory


def signs_for_values(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised sign rule over a Series of consensus_nt strings (NaN allowed)."""
    cache: dict[Any, tuple[int, bool, bool]] = {}
    sign = np.ones(len(values), dtype=np.int8)
    uncertain = np.ones(len(values), dtype=bool)
    modulatory = np.zeros(len(values), dtype=np.uint8)
    for i, value in enumerate(values.tolist()):
        key = value if isinstance(value, str) else None
        res = cache.get(key)
        if res is None:
            res = sign_from_nt(value)
            cache[key] = res
        sign[i], uncertain[i], modulatory[i] = res[0], res[1], res[2]
    return sign, uncertain, modulatory


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def load_annotations(path: Path | None = None) -> pd.DataFrame:
    path = path or source("annotations").path()
    table = feather.read_table(str(path))
    available = [c for c in ANNOTATION_COLUMNS if c in table.column_names]
    missing = [c for c in ANNOTATION_COLUMNS if c not in table.column_names]
    df = table.select(available).to_pandas()
    for c in missing:
        df[c] = np.nan
    return df


def node_mask(ann: pd.DataFrame) -> np.ndarray:
    superclass = ann["superclass"]
    has_superclass = superclass.notna() & (superclass.astype(str).str.strip() != "")
    status = ann["status"].fillna("").astype(str)
    return (has_superclass & (status != "Glia")).to_numpy()


def select_nodes(ann: pd.DataFrame) -> pd.DataFrame:
    """Apply the node policy and return retained rows sorted by bodyId.

    The row position in the returned frame is the graph index of that neuron.
    """
    nodes = ann.loc[node_mask(ann)].copy()
    nodes = nodes.sort_values("bodyId", kind="stable").reset_index(drop=True)
    if nodes["bodyId"].duplicated().any():
        raise ValueError("annotations contain duplicate bodyId values")
    return nodes


def _string_column(series: pd.Series) -> np.ndarray:
    values = series.fillna("").astype(str).to_numpy()
    return np.asarray(values, dtype=np.str_)


def node_arrays(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-neuron annotation arrays in graph order."""
    return {
        "ids": nodes["bodyId"].to_numpy(dtype=np.int64),
        "superclass": _string_column(nodes["superclass"]),
        "type": _string_column(nodes["type"]),
        "instance": _string_column(nodes["instance"]),
        "cell_class": _string_column(nodes["class"]),
        "soma_side": _string_column(nodes["somaSide"]),
        "root_side": _string_column(nodes["rootSide"]),
        "hex1": nodes["assignedOlHex1"].to_numpy(dtype=np.float64),
        "hex2": nodes["assignedOlHex2"].to_numpy(dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Neurotransmitters
# ---------------------------------------------------------------------------


def load_neurotransmitters(ids: np.ndarray, path: Path | None = None) -> dict[str, np.ndarray]:
    """Sign, uncertainty and modulatory flags aligned to `ids` (graph order)."""
    path = path or source("neurotransmitters").path()
    table = feather.read_table(str(path), columns=["body", "consensus_nt"])
    nt = table.to_pandas()
    nt = nt.drop_duplicates("body", keep="first").set_index("body")
    values = nt["consensus_nt"].reindex(ids)
    sign, uncertain, modulatory = signs_for_values(values)
    present = values.notna().to_numpy()
    return {
        "nt_sign": sign,
        "nt_uncertain": uncertain,
        "modulatory": modulatory,
        "nt_present": present,
        "nt_values": values,
    }


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def _lookup(ids: np.ndarray, bodies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map body ids to graph indices through a sorted id array. Returns (index, found)."""
    n = len(ids)
    pos = np.searchsorted(ids, bodies)
    clipped = np.minimum(pos, n - 1)
    found = (pos < n) & (ids[clipped] == bodies)
    return clipped, found


def load_edges(
    ids: np.ndarray,
    path: Path | None = None,
    progress: Progress | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream the weights feather and keep edges between retained bodies.

    Returns (pre, post, contacts) sorted by (pre, post), plus a stats dict.
    Duplicate (pre, post) rows, if any, are summed and reported.
    """
    path = path or source("weights").path()
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    if not np.all(ids[1:] > ids[:-1]):
        raise ValueError("ids must be strictly increasing")
    pres: list[np.ndarray] = []
    posts: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    rows_total = 0
    t0 = time.perf_counter()
    with pa.memory_map(str(path), "r") as src:
        reader = ipc.open_file(src)
        nb = reader.num_record_batches
        for b in range(nb):
            batch = reader.get_batch(b)
            pre = batch.column("body_pre").to_numpy(zero_copy_only=False)
            post = batch.column("body_post").to_numpy(zero_copy_only=False)
            w = batch.column("weight").to_numpy(zero_copy_only=False)
            rows_total += len(pre)
            ip, okp = _lookup(ids, pre)
            jp, okq = _lookup(ids, post)
            keep = okp & okq
            if keep.any():
                pres.append(ip[keep].astype(np.int32))
                posts.append(jp[keep].astype(np.int32))
                ws.append(w[keep].astype(np.int64))
            if progress is not None and (b % 200 == 0 or b == nb - 1):
                _say(progress, f"  weights batch {b + 1} of {nb}, rows {rows_total:,}")
    if pres:
        pre = np.concatenate(pres)
        post = np.concatenate(posts)
        contacts = np.concatenate(ws)
    else:
        pre = np.zeros(0, np.int32)
        post = np.zeros(0, np.int32)
        contacts = np.zeros(0, np.int64)
    del pres, posts, ws
    order = np.lexsort((post, pre))
    pre = pre[order]
    post = post[order]
    contacts = contacts[order]
    del order
    duplicates = 0
    if len(pre) > 1:
        same = (pre[1:] == pre[:-1]) & (post[1:] == post[:-1])
        duplicates = int(same.sum())
        if duplicates:
            first = np.concatenate(([True], ~same))
            group = np.cumsum(first) - 1
            contacts = np.bincount(group, weights=contacts).astype(np.int64)
            pre = pre[first]
            post = post[first]
    if contacts.size and contacts.min() < 1:
        raise ValueError("weights file contains non-positive contact counts")
    stats = {
        "rows_read": int(rows_total),
        "edges": int(len(pre)),
        "contacts": int(contacts.sum()),
        "duplicates_merged": duplicates,
        "self_edges": int((pre == post).sum()),
        "weight_one_edges": int((contacts == 1).sum()),
        "read_seconds": round(time.perf_counter() - t0, 2),
    }
    return pre, post, contacts, stats


def build_csr(pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    """Row pointer for edges already sorted by `pre`."""
    counts = np.bincount(pre, minlength=n).astype(np.int64)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr


def assert_expected(n: int, edges: int, contacts: int, expected: dict | None = None) -> None:
    expected = expected or EXPECTED
    problems = []
    if n != expected["neurons"]:
        problems.append(f"neurons {n:,} != expected {expected['neurons']:,}")
    if edges != expected["edges"]:
        problems.append(f"edges {edges:,} != expected {expected['edges']:,}")
    if contacts != expected["contacts"]:
        problems.append(f"contacts {contacts:,} != expected {expected['contacts']:,}")
    if problems:
        raise AssertionError(
            "normalized graph does not match MaleCNS v1.0 expectations: " + "; ".join(problems)
        )
