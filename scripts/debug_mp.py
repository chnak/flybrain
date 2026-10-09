"""Debug why multi-process fails on MaleCNS."""
import sys
import time
import traceback
sys.path.insert(0, ".")

from flybrainer.ensemble import BatchBrain
from flybrainer import Brain
from flybrainer.encoders import make_encoder
from flybrainer.interfaces import Bar, SensorFrame
from datetime import datetime, timedelta, timezone


def make_frame():
    IST = timezone(timedelta(hours=5, minutes=30))
    bars = tuple(
        Bar(timestamp=datetime(2024, 1, 15, 9, 25 + i, tzinfo=IST),
            open=21600. + i * 5, high=21650. + i * 5,
            low=21580. + i * 5, close=21630. + i * 5, volume=1000 + i * 50)
        for i in range(3)
    )
    return SensorFrame(
        timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=IST),
        index_bars=bars, vix=14.2, vix_bars=bars,
        straddle_premium=120.5, entry_credit=None,
        days_to_expiry=4.0, minutes_since_open=10, position_lots=0,
    )


def factory(wid):
    print(f"  factory({wid}) called in pid={__import__('os').getpid()}", flush=True)
    b = Brain(plastic=False)
    print(f"  factory({wid}) done, n={b.n}", flush=True)
    return b


print("main pid:", __import__("os").getpid(), flush=True)
try:
    print("creating BatchBrain with 2 workers, keep_alive=True", flush=True)
    ens = BatchBrain(num_workers=2, keep_alive=True, brain_factory=factory)
    ens.start()
    print("start done", flush=True)

    brain = Brain(plastic=False)
    encoder = make_encoder("bars", eye_map=brain.eye_map())
    sf = make_frame()
    stim = encoder.encode(sf, brain)
    print("warm: 1 stim", flush=True)
    res = ens.observe_batch([stim], neural_ms=200.0)
    print(f"warm done, len={len(res)}", flush=True)

    print("real: 2 stim", flush=True)
    stims = [stim, stim]
    res = ens.observe_batch(stims, neural_ms=200.0)
    print(f"real done, len={len(res)}", flush=True)
    ens.shutdown()
    print("shutdown done", flush=True)
except Exception as e:
    print("EXCEPTION:", type(e).__name__, e)
    traceback.print_exc()
