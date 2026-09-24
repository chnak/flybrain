"""Shared types and protocols at module boundaries.

Every module that crosses a boundary (neural, sensory, readout, market,
straddle, execution, experiments, api) codes against these definitions.
Keep this file small, dependency-free (numpy only) and stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar. `timestamp` is tz-aware Asia/Kolkata, bar start."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Contract:
    """One option contract as resolved from the OpenAlgo symbol master."""

    symbol: str  # OpenAlgo symbol, e.g. NIFTY15SEP2623400CE
    exchange: str  # NFO
    underlying: str  # NIFTY
    expiry: date
    strike: float
    option_type: str  # CE or PE
    lot_size: int
    tick_size: float
    freeze_qty: int


@dataclass(frozen=True)
class Quote:
    """Top of book for one symbol. `timestamp` is the local receipt time (epoch seconds)."""

    symbol: str
    exchange: str
    ltp: float
    bid: float
    ask: float
    timestamp: float


@dataclass(frozen=True)
class StraddleQuote:
    """Both legs of a straddle quoted together."""

    call: Quote
    put: Quote
    strike: float
    expiry: date

    @property
    def combined_ltp(self) -> float:
        return self.call.ltp + self.put.ltp

    @property
    def synthetic_forward(self) -> float:
        return self.strike + self.call.ltp - self.put.ltp


@dataclass(frozen=True)
class SessionWindow:
    """Trading session for one date, all times tz-aware Asia/Kolkata."""

    trading_date: date
    market_open: datetime
    market_close: datetime
    trade_start: datetime  # 09:20 by rule
    last_entry: datetime  # default 14:30
    square_off: datetime  # 15:15 by rule
    is_expiry_day: bool


# ---------------------------------------------------------------------------
# Neural boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stimulus:
    """What the fly receives for one observation.

    `r16` and `r8` are luminance-like drives in [0, 1], one value per mapped
    R1-R6 and R8 photoreceptor respectively, in the brain's population order.
    `pulses` is a list of (population name, current in mV, duration in ms)
    for explicit stimulation such as dopamine reinforcement in the plastic arm.
    """

    r16: np.ndarray
    r8: np.ndarray
    pulses: tuple[tuple[str, float, float], ...] = ()


@dataclass(frozen=True)
class ObservationResult:
    """Spike counts for one observation, per neuron, plus bookkeeping."""

    counts: np.ndarray  # int32, length n
    neural_ms: float
    sim_ms: float  # cumulative simulated time after this observation
    compute_seconds: float


@runtime_checkable
class BrainProtocol(Protocol):
    """The only surface the rest of the system uses to talk to the connectome."""

    n: int
    populations: dict[str, np.ndarray]  # name -> int32 index array

    def observe(self, stimulus: Stimulus, neural_ms: float) -> ObservationResult: ...

    def checkpoint(self, path: str) -> None: ...

    def restore(self, path: str) -> None: ...

    def provenance(self) -> dict: ...


# Population names every BrainProtocol implementation must provide.
REQUIRED_POPULATIONS = (
    "R1-R6",
    "R8p",
    "R8y",
    "lamina",
    "KC",
    "PAM11",
    "PPL101",
    "MBON07",
    "MBON11",
    "MBON",
    "DNp20_L",
    "DNp20_R",
    "DNpe017",
    "DN",
    "central_complex",
    "random2000",
)


# ---------------------------------------------------------------------------
# Sensory and readout boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorFrame:
    """Everything an encoder may look at for one 5 minute bar. Past only."""

    timestamp: datetime  # bar close time, tz-aware Asia/Kolkata
    index_bars: tuple[Bar, ...]  # trailing window of NIFTY 5m bars, oldest first
    vix: float  # INDIAVIX last value
    vix_bars: tuple[Bar, ...]  # trailing INDIAVIX bars, may be empty
    straddle_premium: float | None  # combined ATM premium if known, else None
    entry_credit: float | None  # combined premium at entry if in a position
    days_to_expiry: float
    minutes_since_open: int
    position_lots: int  # negative for short straddle, 0 when flat


class Decision(str, Enum):  # noqa: UP042 (str() of a member must stay the qualified name)
    ENTER = "ENTER"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Prediction:
    """Readout output. `realized_over_implied` above 1 means more movement than priced."""

    realized_over_implied: float
    confidence: float  # 0 to 1, readout-specific
    decision: Decision
    details: dict = field(default_factory=dict)


@runtime_checkable
class EncoderProtocol(Protocol):
    name: str

    def encode(self, observation: SensorFrame, brain: BrainProtocol) -> Stimulus: ...

    def config_hash(self) -> str: ...


@runtime_checkable
class ReadoutProtocol(Protocol):
    name: str

    def predict(self, counts: np.ndarray, brain: BrainProtocol, observation: SensorFrame) -> Prediction: ...

    def config_hash(self) -> str: ...


# ---------------------------------------------------------------------------
# Execution boundary
# ---------------------------------------------------------------------------


class Side(str, Enum):  # noqa: UP042
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Leg:
    contract: Contract
    side: Side
    quantity: int  # shares, a multiple of lot_size
    limit_price: float | None  # None means MARKET


@dataclass(frozen=True)
class Intent:
    """A straddle action the engine wants executed. Persisted before any network call."""

    intent_id: str
    kind: str  # ENTRY, EXIT, SQUARE_OFF, REPAIR
    legs: tuple[Leg, ...]
    reason: str
    created_at: float  # epoch seconds
    observation_hash: str | None = None


class IntentStatus(str, Enum):  # noqa: UP042
    PREPARED = "PREPARED"
    UNKNOWN = "UNKNOWN"
    ACCEPTED = "ACCEPTED"
    PARTIAL = "PARTIAL"
    SETTLED = "SETTLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Fill:
    intent_id: str
    symbol: str
    side: Side
    quantity: int
    average_price: float
    order_id: str
    timestamp: float


class Veto(Exception):
    """Raised by the guard. The guard rejects; it never substitutes."""


class UnresolvedOrder(Exception):
    """An exchange outcome could not be determined. The worker must halt for review."""
