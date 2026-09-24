"""Readouts: turn spike counts into discrete / continuous predictions."""
from flybrainer.readout.fixed import ColumnBinding, FixedDecoder
from flybrainer.readout.reservoir import ReservoirReadout

__all__ = ["FixedDecoder", "ReservoirReadout", "ColumnBinding"]
