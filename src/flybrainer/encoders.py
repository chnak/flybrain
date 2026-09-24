"""The three sensory encoders: chart (A), retinotopic bars (B), feature patches (C).

Every encoder turns a SensorFrame into a Stimulus: one luminance in
[0, 1] per mapped R1-R6 cell and per mapped R8 cell, in the brain's
population order, sampled through the eye map. Encoders are deterministic,
look only at the past bars they are given, and hash their parameters into
`config_hash()` for provenance.

    encoder = make_encoder("B", settings, eye_map)
    stimulus = encoder.encode(observation, brain)
    png_bytes = encoder.render_png(stimulus, eye_map)
"""

from __future__ import annotations

import hashlib
import io
import json
import math

import numpy as np
from PIL import Image, ImageDraw

from flybrain.eyemap import EyeMap, as_eye_map, resolve_eye_map
from flybrain.interfaces import SensorFrame, Stimulus

ENCODER_NAMES = {
    "A": "chart",
    "B": "bars",
    "C": "features",
    "chart": "chart",
    "bars": "bars",
    "features": "features",
}

_SRGB_LUT = np.array(
    [
        (c / 12.92) if c <= 0.04045 else (((c + 0.055) / 1.055) ** 2.4)
        for c in (np.arange(256) / 255.0)
    ],
    dtype=np.float32,
)


def stimulus_hash(stimulus: Stimulus) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(stimulus.r16, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(stimulus.r8, dtype=np.float32).tobytes())
    return h.hexdigest()


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _closes(observation: SensorFrame) -> np.ndarray:
    return np.array([b.close for b in observation.index_bars], dtype=np.float64)


def _hlc(observation: SensorFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bars = observation.index_bars
    h = np.array([b.high for b in bars], dtype=np.float64)
    lo = np.array([b.low for b in bars], dtype=np.float64)
    c = np.array([b.close for b in bars], dtype=np.float64)
    return h, lo, c


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float:
    """Simple average true range over the last `window` bars. Falls back to 0.2 percent of price."""
    n = len(close)
    if n == 0:
        return 1.0
    if n < 2:
        return max(1e-6, 0.002 * float(close[-1]))
    prev = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)))
    tail = tr[-window:]
    value = float(np.mean(tail)) if tail.size else 0.0
    if not np.isfinite(value) or value <= 0.0:
        value = max(1e-6, 0.002 * float(close[-1]))
    return value


def log_returns(close: np.ndarray) -> np.ndarray:
    if len(close) < 2:
        return np.zeros(0, dtype=np.float64)
    safe = np.clip(close, 1e-9, None)
    return np.diff(np.log(safe))


def rolling_return_std(close: np.ndarray, window: int = 60, floor: float = 1e-6) -> float:
    r = log_returns(close)
    tail = r[-window:]
    if tail.size < 2:
        return max(floor, 1e-4)
    value = float(np.std(tail))
    return max(floor, value) if np.isfinite(value) else max(floor, 1e-4)


def vix_zscore(observation: SensorFrame, window: int = 20) -> float:
    closes = np.array([b.close for b in observation.vix_bars[-window:]], dtype=np.float64)
    if closes.size < 2:
        return 0.0
    std = float(np.std(closes))
    std = max(std, 0.05)
    return float((observation.vix - float(np.mean(closes))) / std)


def _config_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _BaseEncoder:
    name: str = "base"

    def __init__(self, eye_map=None):
        self._eye_map: EyeMap | None = as_eye_map(eye_map) if eye_map is not None else None
        self._last_hash: str | None = None

    def eye_map(self, brain=None) -> EyeMap:
        if self._eye_map is None:
            if brain is None:
                raise ValueError(f"encoder {self.name} needs an eye map or a brain")
            self._eye_map = resolve_eye_map(brain)
        return self._eye_map

    def params(self) -> dict:  # pragma: no cover, overridden
        return {"name": self.name}

    def config_hash(self) -> str:
        return _config_hash(self.params())

    def _finish(self, r16: np.ndarray, r8: np.ndarray) -> Stimulus:
        r16 = np.clip(np.nan_to_num(r16, nan=0.5), 0.0, 1.0).astype(np.float32)
        r8 = np.clip(np.nan_to_num(r8, nan=0.5), 0.0, 1.0).astype(np.float32)
        stim = Stimulus(r16=r16, r8=r8)
        self._last_hash = stimulus_hash(stim)
        return stim

    def render_png(self, stimulus: Stimulus, eye_map=None) -> bytes:
        em = as_eye_map(eye_map) if eye_map is not None else self.eye_map()
        return render_eye_png(stimulus, em)


# ---------------------------------------------------------------------------
# Encoder A: rendered chart
# ---------------------------------------------------------------------------


class ChartEncoder(_BaseEncoder):
    """Render the last 60 five-minute closes as a thick line and sample the image.

    Light field, 3 px line, vertical window of 3 x ATR14 centred on the current
    close, rising segments blue, falling segments red. R1-R6 receive linear
    luminance (Rec.709 weights on linearized sRGB), R8y the linear green
    channel and R8p the linear blue channel at their uv positions. Image
    columns run left (oldest) to right (newest); v = 1 is the top row.
    """

    name = "chart"

    def __init__(
        self,
        eye_map=None,
        *,
        width: int = 320,
        height: int = 180,
        bars: int = 60,
        line_width: int = 3,
        atr_window: int = 14,
        atr_multiple: float = 3.0,
        background: tuple[int, int, int] = (235, 235, 235),
        rising: tuple[int, int, int] = (0, 0, 255),
        falling: tuple[int, int, int] = (255, 0, 0),
        margin: int = 6,
    ):
        super().__init__(eye_map)
        self.width = int(width)
        self.height = int(height)
        self.bars = int(bars)
        self.line_width = int(line_width)
        self.atr_window = int(atr_window)
        self.atr_multiple = float(atr_multiple)
        self.background = tuple(int(c) for c in background)
        self.rising = tuple(int(c) for c in rising)
        self.falling = tuple(int(c) for c in falling)
        self.margin = int(margin)
        self.last_image: Image.Image | None = None

    def params(self) -> dict:
        return {
            "name": self.name,
            "version": 1,
            "width": self.width,
            "height": self.height,
            "bars": self.bars,
            "line_width": self.line_width,
            "atr_window": self.atr_window,
            "atr_multiple": self.atr_multiple,
            "background": list(self.background),
            "rising": list(self.rising),
            "falling": list(self.falling),
            "margin": self.margin,
        }

    def render_chart(self, observation: SensorFrame) -> Image.Image:
        high, low, close = _hlc(observation)
        img = Image.new("RGB", (self.width, self.height), self.background)
        if close.size == 0:
            return img
        draw = ImageDraw.Draw(img)
        window_closes = close[-self.bars :]
        current = float(close[-1])
        span = self.atr_multiple * atr(high, low, close, self.atr_window)
        usable_w = self.width - 2 * self.margin - 1
        usable_h = self.height - 2 * self.margin - 1
        n = len(window_closes)
        # Newest bar always at the right edge; fewer bars simply start further right.
        step = usable_w / max(1, self.bars - 1)
        x0 = self.margin + usable_w - step * (n - 1)
        xs = x0 + step * np.arange(n)
        ys = self.margin + usable_h / 2.0 - (window_closes - current) / span * usable_h
        for i in range(1, n):
            colour = self.rising if window_closes[i] >= window_closes[i - 1] else self.falling
            draw.line(
                [(float(xs[i - 1]), float(ys[i - 1])), (float(xs[i]), float(ys[i]))],
                fill=colour,
                width=self.line_width,
            )
        return img

    def sample(self, img: Image.Image, em: EyeMap) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(img, dtype=np.uint8)
        lin = _SRGB_LUT[arr]  # (H, W, 3) linear
        lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
        w, h = self.width, self.height

        def pix(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x = np.clip(np.rint(uv[:, 0] * (w - 1)).astype(np.int64), 0, w - 1)
            y = np.clip(np.rint((1.0 - uv[:, 1]) * (h - 1)).astype(np.int64), 0, h - 1)
            return y, x

        y16, x16 = pix(em.uv_r16)
        r16 = lum[y16, x16]
        y8, x8 = pix(em.uv_r8)
        green = lin[y8, x8, 1]
        blue = lin[y8, x8, 2]
        r8 = np.where(em.r8_channel == 1, green, blue)
        return r16, r8

    def encode(self, observation: SensorFrame, brain=None) -> Stimulus:
        em = self.eye_map(brain)
        img = self.render_chart(observation)
        self.last_image = img
        r16, r8 = self.sample(img, em)
        return self._finish(r16, r8)

    def render_png(self, stimulus: Stimulus, eye_map=None) -> bytes:
        if self.last_image is not None and self._last_hash == stimulus_hash(stimulus):
            buf = io.BytesIO()
            self.last_image.save(buf, format="PNG")
            return buf.getvalue()
        return super().render_png(stimulus, eye_map)


# ---------------------------------------------------------------------------
# Encoder B: retinotopic bars (default)
# ---------------------------------------------------------------------------


class BarsEncoder(_BaseEncoder):
    """Each column of ommatidia is one of the last N five-minute bars.

    Oldest bar at the periphery (u near 0 on the left eye, u near 1 on the
    right eye), newest at the midline. R1-R6 luminance is a sigmoid of the
    bar's log return over the rolling 60 bar return std. R8y (channel 1)
    carries the bar's range over ATR14, clipped. R8p (channel 2) is a slow
    channel: 0.5 + 0.25 x clip(VIX z-score over 20 days) + 0.25 x clip(premium
    change since entry over the entry credit) while in a position.
    """

    name = "bars"

    def __init__(
        self,
        eye_map=None,
        *,
        columns: int = 30,
        std_window: int = 60,
        atr_window: int = 14,
        range_atr_multiple: float = 2.0,
        vix_window: int = 20,
    ):
        super().__init__(eye_map)
        self.columns = int(columns)
        self.std_window = int(std_window)
        self.atr_window = int(atr_window)
        self.range_atr_multiple = float(range_atr_multiple)
        self.vix_window = int(vix_window)

    def params(self) -> dict:
        return {
            "name": self.name,
            "version": 1,
            "columns": self.columns,
            "std_window": self.std_window,
            "atr_window": self.atr_window,
            "range_atr_multiple": self.range_atr_multiple,
            "vix_window": self.vix_window,
        }

    def column_values(self, observation: SensorFrame) -> tuple[np.ndarray, np.ndarray, float]:
        """Per column (oldest first): luminance, range channel, and the slow channel value."""
        high, low, close = _hlc(observation)
        n = self.columns
        lum = np.full(n, 0.5, dtype=np.float64)
        rng_ch = np.zeros(n, dtype=np.float64)
        if close.size >= 2:
            rets = log_returns(close)
            std = rolling_return_std(close, self.std_window)
            atr_value = atr(high, low, close, self.atr_window)
            k = min(n, rets.size)
            lum[n - k :] = _sigmoid(rets[-k:] / std)
            ranges = (high - low)[-k:]
            rng_ch[n - k :] = np.clip(ranges / (self.range_atr_multiple * atr_value), 0.0, 1.0)
        z = float(np.clip(vix_zscore(observation, self.vix_window), -1.0, 1.0))
        prem = 0.0
        if (
            observation.position_lots != 0
            and observation.entry_credit
            and observation.straddle_premium is not None
            and observation.entry_credit > 0
        ):
            prem = float(
                np.clip(
                    (observation.straddle_premium - observation.entry_credit) / observation.entry_credit,
                    -1.0,
                    1.0,
                )
            )
        slow = 0.5 + 0.25 * z + 0.25 * prem
        return lum, rng_ch, slow

    def _column_index(self, uv: np.ndarray, eye: np.ndarray) -> np.ndarray:
        col = np.clip((uv[:, 0] * self.columns).astype(np.int64), 0, self.columns - 1)
        # Left eye: u near 0 is the periphery (oldest). Right eye: u near 1 is the periphery.
        return np.where(eye == 0, col, self.columns - 1 - col)

    def encode(self, observation: SensorFrame, brain=None) -> Stimulus:
        em = self.eye_map(brain)
        lum, rng_ch, slow = self.column_values(observation)
        c16 = self._column_index(em.uv_r16, em.eye_r16)
        r16 = lum[c16]
        c8 = self._column_index(em.uv_r8, em.eye_r8)
        r8 = np.where(em.r8_channel == 1, rng_ch[c8], slow)
        return self._finish(r16, r8)


# ---------------------------------------------------------------------------
# Encoder C: engineered features on disjoint retinal patches
# ---------------------------------------------------------------------------


FEATURE_NAMES = (
    "ret5",
    "ret15",
    "abs5",
    "abs15",
    "realized_over_implied",
    "vix_change",
    "time_of_day",
    "days_to_expiry",
    "position",
)


class FeatureEncoder(_BaseEncoder):
    """Nine engineered features, each painted on its own 3 x 3 patch of both eyes.

    Patch p = u_bin * 3 + v_bin. Feature order is FEATURE_NAMES. All features
    are squashed into [0, 1] with rolling scales computed from the observation
    window only.
    """

    name = "features"

    def __init__(
        self,
        eye_map=None,
        *,
        grid: int = 3,
        std_window: int = 60,
        realized_bars: int = 12,
        horizon_minutes: int = 60,
        session_minutes: int = 375,
        expiry_scale_days: float = 25.0,
    ):
        super().__init__(eye_map)
        self.grid = int(grid)
        self.std_window = int(std_window)
        self.realized_bars = int(realized_bars)
        self.horizon_minutes = int(horizon_minutes)
        self.session_minutes = int(session_minutes)
        self.expiry_scale_days = float(expiry_scale_days)

    def params(self) -> dict:
        return {
            "name": self.name,
            "version": 1,
            "grid": self.grid,
            "std_window": self.std_window,
            "realized_bars": self.realized_bars,
            "horizon_minutes": self.horizon_minutes,
            "session_minutes": self.session_minutes,
            "expiry_scale_days": self.expiry_scale_days,
            "features": list(FEATURE_NAMES),
        }

    def features(self, observation: SensorFrame) -> np.ndarray:
        close = _closes(observation)
        std = rolling_return_std(close, self.std_window)
        values = np.full(len(FEATURE_NAMES), 0.5, dtype=np.float64)

        def kret(k: int) -> float:
            if close.size > k:
                return float(math.log(max(close[-1], 1e-9) / max(close[-1 - k], 1e-9)))
            return 0.0

        r5, r15 = kret(5), kret(15)
        values[0] = _sigmoid(r5 / (std * math.sqrt(5.0)))
        values[1] = _sigmoid(r15 / (std * math.sqrt(15.0)))
        values[2] = np.clip(abs(r5) / (2.0 * std * math.sqrt(5.0)), 0.0, 1.0)
        values[3] = np.clip(abs(r15) / (2.0 * std * math.sqrt(15.0)), 0.0, 1.0)

        ratio = 0.5
        if observation.straddle_premium and close.size > self.realized_bars:
            realized = abs(float(close[-1] - close[-1 - self.realized_bars]))
            minutes_left = max(1.0, observation.days_to_expiry * self.session_minutes)
            implied = observation.straddle_premium * math.sqrt(
                min(1.0, self.horizon_minutes / minutes_left)
            )
            if implied > 0:
                ratio = float(np.clip(realized / implied / 2.0, 0.0, 1.0))
        values[4] = ratio

        vb = observation.vix_bars
        if len(vb) >= 2:
            closes = np.array([b.close for b in vb[-21:]], dtype=np.float64)
            changes = np.diff(closes)
            scale = max(float(np.std(changes)) if changes.size >= 2 else 0.0, 0.1)
            values[5] = _sigmoid((observation.vix - float(closes[-1])) / scale)
        elif len(vb) == 1:
            values[5] = _sigmoid((observation.vix - vb[-1].close) / 0.5)

        values[6] = np.clip(observation.minutes_since_open / self.session_minutes, 0.0, 1.0)
        values[7] = np.clip(observation.days_to_expiry / self.expiry_scale_days, 0.0, 1.0)
        values[8] = 1.0 if observation.position_lots != 0 else 0.0
        return values

    def _patch_index(self, uv: np.ndarray) -> np.ndarray:
        g = self.grid
        ub = np.clip((uv[:, 0] * g).astype(np.int64), 0, g - 1)
        vb = np.clip((uv[:, 1] * g).astype(np.int64), 0, g - 1)
        return ub * g + vb

    def encode(self, observation: SensorFrame, brain=None) -> Stimulus:
        em = self.eye_map(brain)
        values = self.features(observation)
        n_patches = self.grid * self.grid
        table = np.full(n_patches, 0.5, dtype=np.float64)
        table[: min(n_patches, values.size)] = values[:n_patches]
        r16 = table[self._patch_index(em.uv_r16)]
        r8 = table[self._patch_index(em.uv_r8)]
        return self._finish(r16, r8)


# ---------------------------------------------------------------------------
# Rendering of a stimulus as an eye map (used by B and C, and by A as fallback)
# ---------------------------------------------------------------------------


def render_eye_png(stimulus: Stimulus, em: EyeMap, width: int = 640, height: int = 360) -> bytes:
    """Two panels (left eye, right eye). R1-R6 cells as grey discs, R8 cells as
    green (R8y) or blue (R8p) dots whose intensity is the stimulus value."""
    img = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    half = width // 2
    draw.line([(half, 0), (half, height)], fill=(60, 60, 60), width=1)
    n16 = max(1, em.n_r16)
    radius = max(2, int(0.9 * math.sqrt(half * height / n16) / 2))

    def xy(uv: np.ndarray, eye: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.where(eye == 0, uv[:, 0] * (half - 1), half + uv[:, 0] * (half - 1))
        y = (1.0 - uv[:, 1]) * (height - 1)
        return x, y

    x16, y16 = xy(em.uv_r16, em.eye_r16)
    r16 = np.clip(np.asarray(stimulus.r16, dtype=np.float64), 0.0, 1.0)
    for i in range(em.n_r16):
        g = int(round(255 * r16[i])) if i < r16.size else 128
        draw.ellipse(
            [(x16[i] - radius, y16[i] - radius), (x16[i] + radius, y16[i] + radius)],
            fill=(g, g, g),
        )
    x8, y8 = xy(em.uv_r8, em.eye_r8)
    r8 = np.clip(np.asarray(stimulus.r8, dtype=np.float64), 0.0, 1.0)
    small = max(1, radius // 2)
    for i in range(em.n_r8):
        value = int(round(255 * r8[i])) if i < r8.size else 128
        colour = (0, value, 0) if em.r8_channel[i] == 1 else (0, 0, value)
        draw.ellipse(
            [(x8[i] - small, y8[i] - small), (x8[i] + small, y8[i] + small)],
            fill=colour,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_encoder(name: str, settings: dict | None = None, eye_map=None):
    """Build an encoder by name (A/chart, B/bars, C/features).

    `settings` is the merged settings dict; `settings["neural"]` may carry an
    `encoder_params` dict with keyword overrides for the chosen encoder, and
    `horizon_minutes` is passed to the feature encoder.
    """
    key = ENCODER_NAMES.get(str(name).strip())
    if key is None:
        key = ENCODER_NAMES.get(str(name).strip().upper())
    if key is None:
        raise ValueError(f"unknown encoder {name!r}; use A, B or C")
    neural = (settings or {}).get("neural", {}) if isinstance(settings, dict) else {}
    overrides = dict(neural.get("encoder_params", {}) or {})
    if key == "chart":
        return ChartEncoder(eye_map, **overrides)
    if key == "bars":
        return BarsEncoder(eye_map, **overrides)
    if "horizon_minutes" in neural and "horizon_minutes" not in overrides:
        overrides["horizon_minutes"] = int(neural["horizon_minutes"])
    return FeatureEncoder(eye_map, **overrides)
