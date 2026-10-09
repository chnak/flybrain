"""Module-level brain factories for bench_decision.py.

These functions live in a module (not closures) so they are picklable and
the spawn-imported worker can call them by name.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is on path so we can import `examples.real_brain`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flybrainer import Brain  # noqa: E402
from flybrainer.interfaces import BrainProtocol  # noqa: E402


def make_toy_brain() -> BrainProtocol:
    """100-neuron synthetic graph. Fast, no I/O. Different seed per call."""
    # Lazy import: real_brain imports scipy/numba etc; keep top imports cheap
    from examples.real_brain import build_synthetic_graph

    g = build_synthetic_graph(seed=42)
    return Brain(graph_path="<x>", graph=g, check_counts=False, plastic=False)


def make_malecns_brain() -> BrainProtocol:
    """166,700-neuron MaleCNS brain. Loads compiled graph.npz on first call."""
    return Brain(plastic=False)


__all__ = ["make_toy_brain", "make_malecns_brain"]
