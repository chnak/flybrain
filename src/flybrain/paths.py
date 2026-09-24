"""Default paths for flybrain artifacts.

Replaces ``openfly.config.PATHS``.  Every location that needs a file on disk
(historical connectome downloads, compiled graph, etc.) goes through the
frozen ``PATHS`` instance defined here.

Override the root at runtime with the ``FLYBRAIN_DATA`` environment
variable; otherwise defaults to ``~/.flybrain``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Filesystem locations for flybrain artifacts."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root

    @property
    def malecns(self) -> Path:
        """Directory of raw MaleCNS feather files (download destination)."""
        return self.root / "malecns"

    @property
    def graph(self) -> Path:
        """Default compiled graph location (``graph.npz``)."""
        return self.root / "graph.npz"

    def ensure(self) -> None:
        """Create root + subdirs if missing."""
        for p in (self.root, self.malecns):
            p.mkdir(parents=True, exist_ok=True)


def _default_root() -> Path:
    env = os.environ.get("FLYBRAIN_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".flybrain").resolve()


PATHS: Paths = Paths(root=_default_root())


__all__ = ["Paths", "PATHS"]
