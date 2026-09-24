"""Smoke tests: every public symbol imports and is callable / inspectable.

These tests do NOT require a compiled connectome graph -- they cover the
pure-Python surface (interfaces, eyemap, encoders, kernel constants,
plasticity config, readout base shapes).
"""
from __future__ import annotations

import numpy as np
import pytest

import flybrainer
from flybrainer import (
    ENCODER_NAMES,
    KERNEL_PARAMETERS,
    KERNEL_VERSION,
    MV_PER_CONTACT,
    REQUIRED_POPULATIONS,
    BarsEncoder,
    ChartEncoder,
    Decision,
    FeatureEncoder,
    Kernel,
    KernelState,
    PlasticityConfig,
    Prediction,
    SensorFrame,
    Stimulus,
    default_eye_map,
    make_encoder,
    resolve_eye_map,
)

# ---- top-level ----

def test_version_present():
    assert isinstance(flybrainer.__version__, str)
    assert flybrainer.__version__ == "0.1.0"


def test_all_has_expected_core_symbols():
    for name in (
        "Stimulus", "BrainProtocol", "SensorFrame", "Decision",
        "Prediction", "EncoderProtocol", "ReadoutProtocol",
        "Kernel", "KernelState", "KCMBONPlasticity",
        "ChartEncoder", "BarsEncoder", "FeatureEncoder",
        "EyeMap", "default_eye_map", "resolve_eye_map",
    ):
        assert name in flybrainer.__all__, name


# ---- interfaces ----

def test_required_populations_is_tuple():
    assert isinstance(REQUIRED_POPULATIONS, tuple)
    assert "R1-R6" in REQUIRED_POPULATIONS
    assert "KC" in REQUIRED_POPULATIONS
    assert "central_complex" in REQUIRED_POPULATIONS


def test_stimulus_construction():
    s = Stimulus(
        r16=np.zeros(6, dtype=np.float32),
        r8=np.zeros(2, dtype=np.float32),
    )
    assert s.r16.shape == (6,)
    assert s.r8.shape == (2,)
    assert s.pulses == ()


def test_decision_enum():
    assert Decision.ENTER == "ENTER"
    assert Decision.EXIT == "EXIT"
    assert Decision.HOLD == "HOLD"
    # str Enum: str() must be the qualified name, per noqa: UP042
    assert str(Decision.ENTER) == "Decision.ENTER"


def test_prediction_construction():
    p = Prediction(
        realized_over_implied=1.5,
        confidence=0.7,
        decision=Decision.ENTER,
    )
    assert p.realized_over_implied == 1.5
    assert p.decision is Decision.ENTER


def test_sensor_frame_construction():
    from datetime import datetime, timedelta, timezone
    IST = timezone(timedelta(hours=5, minutes=30))
    sf = SensorFrame(
        timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=IST),
        index_bars=(),
        vix=14.0,
        vix_bars=(),
        straddle_premium=None,
        entry_credit=None,
        days_to_expiry=4.0,
        minutes_since_open=10,
        position_lots=0,
    )
    assert sf.vix == 14.0
    assert sf.position_lots == 0


# ---- eyemap ----

def test_default_eye_map_requires_brain():
    # Real default_eye_map(brain) needs a compiled Brain; we just confirm the
    # signature is callable and reject a missing argument.
    import inspect
    sig = inspect.signature(default_eye_map)
    assert "brain" in sig.parameters
    with pytest.raises(TypeError):
        default_eye_map()  # missing required positional arg


def test_resolve_eye_map_requires_brain():
    import inspect
    sig = inspect.signature(resolve_eye_map)
    assert "brain" in sig.parameters
    with pytest.raises(TypeError):
        resolve_eye_map()  # missing required positional arg


# ---- encoders ----

def test_encoder_names_listed():
    # ENCODER_NAMES is a dict: alias -> canonical key (e.g. "A" -> "chart")
    assert isinstance(ENCODER_NAMES, dict)
    assert "A" in ENCODER_NAMES
    assert ENCODER_NAMES["A"] == "chart"
    assert ENCODER_NAMES["B"] == "bars"


def test_make_encoder_chart():
    e = make_encoder("chart")
    assert isinstance(e, ChartEncoder)


def test_make_encoder_bars():
    e = make_encoder("bars")
    assert isinstance(e, BarsEncoder)


def test_make_encoder_feature():
    # "C" / "features" (plural) maps to FeatureEncoder
    e = make_encoder("features")
    assert isinstance(e, FeatureEncoder)
    e2 = make_encoder("C")
    assert isinstance(e2, FeatureEncoder)


def test_make_encoder_unknown_raises():
    with pytest.raises((ValueError, KeyError)):
        make_encoder("does-not-exist")


# ---- kernel constants ----

def test_kernel_version_is_string():
    assert isinstance(KERNEL_VERSION, str)
    assert len(KERNEL_VERSION) > 0


def test_kernel_parameters_has_entries():
    assert isinstance(KERNEL_PARAMETERS, dict)
    assert len(KERNEL_PARAMETERS) > 0
    for k, v in KERNEL_PARAMETERS.items():
        assert isinstance(v, (int, float)), k


def test_kernel_and_state_are_classes():
    assert isinstance(Kernel, type)
    assert isinstance(KernelState, type)


# ---- plasticity ----

def test_mv_per_contact_is_positive():
    assert isinstance(MV_PER_CONTACT, float)
    assert MV_PER_CONTACT > 0


def test_plasticity_config_dataclass():
    cfg = PlasticityConfig()
    assert cfg is not None
