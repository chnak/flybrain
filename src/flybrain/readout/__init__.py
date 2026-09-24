"""Readouts: turn spike counts into discrete / continuous predictions."""
from flybrain.readout.fixed import ColumnBinding, FixedDecoder
from flybrain.readout.reservoir import ReservoirReadout

__all__ = ["FixedDecoder", "ReservoirReadout", "ColumnBinding"]
