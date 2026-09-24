# flybrain 使用手册（中文版）

> 版本：0.1.0 · 最后更新：2026-09-24
> 适用对象：使用 flybrain 做研究、二次封装、嵌入式集成、复现 OpenFly 全部或部分管线的开发者。

---

## 目录

- [第 1 章：项目简介与设计目标](#第-1-章项目简介与设计目标)
- [第 2 章：安装与可选扩展](#第-2-章安装与可选扩展)
- [第 3 章：核心概念（必读）](#第-3-章核心概念必读)
- [第 4 章：30 秒上手](#第-4-章30-秒上手)
- [第 5 章：完整 API 参考](#第-5-章完整-api-参考)
- [第 6 章：实战场景（7 个）](#第-6-章实战场景7-个)
- [第 7 章：数据路径与环境变量](#第-7-章数据路径与环境变量)
- [第 8 章：性能、调参与调试](#第-8-章性能调参与调试)
- [第 9 章：测试与开发](#第-9-章测试与开发)
- [第 10 章：常见问题（FAQ）](#第-10-章常见问题faq)
- [附录 A：完整管道图](#附录-a--a--完整管道图)

---

## 第 1 章：项目简介与设计目标

**flybrain** 是从 [OpenFly](https://github.com/marketcalls/openfly) 抽取出来的**独立可重用** Python 包。OpenFly 是一个用果蝇大脑连接组（MaleCNS v1.0）模拟量化交易决策的项目。flybrain 把其中**纯神经科学部分**完全隔离，可以脱离 broker / 市场数据 / 前端独立使用。

**核心能力**

- **Numba JIT LIF（Leaky Integrate-and-Fire）核**，1.8 万+ spikes / 秒
- **3 种编码器**：`ChartEncoder`（蜡烛图）、`BarsEncoder`（K 线柱）、`FeatureEncoder`（标量特征）
- **2 种读出器**：`FixedDecoder`（无需训练，直接比对左右脑半球 spike 计数）、`ReservoirReadout`（Ridge + Logistic，需要 `[readout]` extra）
- **可塑性强 KC→MBON 突触**（Ormond-style Oja）
- **MaleCNS v1.0 flat connectome** 完整工具链：断点续传下载 → sha256 校验 → 标准化 → 编译为 `.npz` graph → 装载到 Numba `Kernel`

**设计目标**

1. **零重型依赖默认**：`numpy + numba + pillow` 即装即用；只有用真脑图或 RC readout 才需 `pandas/pyarrow/sklearn`。
2. **接口稳定**：`interfaces.py` 中的 4 个 Protocol + 4 个 dataclass 视为契约，不向后不兼容改。
3. **可移植**：纯 Python 标准库优先（urllib 取代 requests）；`compile.py` 同时兼容 Linux/Mac/Windows（Windows 下自动跳过 `resource` 模块）。
4. **可追溯**：每个 encoder、decoder 都实现 `config_hash()`；每次 observe 都有 `ObservationResult(sim_ms, compute_seconds)`，可复算 wall-clock。

---

## 第 2 章：安装与可选扩展

### 2.1 基础安装

```bash
# 仅核心（默认）：numpy + numba + pillow
pip install flybrain
# 或本地可编辑安装（开发推荐）：
pip install -e D:/date/20260918/flybrain
```

依赖：Python ≥ 3.12，< 3.14；numpy ≥ 2.2，< 2.5；numba ≥ 0.61。

### 2.2 可选 extras（按需开启）

| Extra | 装的内容 | 用途 |
|---|---|---|
| `feather` | pandas + pyarrow | 解析 MaleCNS feather 文件（`connectome/normalize.py`） |
| `readout` | scikit-learn + scipy + joblib | `ReservoirReadout` 的分类器与岭回归 |
| `download` | requests | 替换 urllib 默认下载路径（仅当你希望 HTTP session 复用时才需要） |
| `dev` | pytest + pytest-cov + ruff | 跑测试与 lint |

```bash
# 全功能
pip install -e "D:/date/20260918/flybrain[feather,readout,download]"

# 仅开发
pip install -e "D:/date/20260918/flybrain[dev]"

# 组合：开发 + 真脑图 + RC readout
pip install -e "D:/date/20260918/flybrain[dev,feather,readout]"
```

### 2.3 验证安装

```python
import flybrain
print(flybrain.__version__)       # 0.1.0
print(len(flybrain.__all__))      # 31 个公开符号
```

---

## 第 3 章：核心概念（必读）

### 3.1 数据管道

flybrain 把"信号 → 决策"分解为 **4 步**，每一步都有明确的接口契约：

```
你的输入       编码            神经            解码            你的输出
──────        ────            ────            ────            ──────
SensorFrame ─[Encoder]→ Stimulus ─[Brain]→ counts ─[Decoder]→ Prediction
(K线/特征)   bars/chart      LIF 核          fixed/RC       (ENTER/EXIT/HOLD)
              features       (Numba @njit)
```

### 3.2 4 个核心契约（`flybrain.interfaces`）

| 名称 | 类型 | 作用 |
|---|---|---|
| `Stimulus` | `@dataclass(frozen)` | `r16`（R1-R6 亮度向量）+ `r8`（R8 亮度向量）+ 可选 `pulses`（多巴胺脉冲） |
| `SensorFrame` | `@dataclass(frozen)` | 一个 5 分钟 bar 的所有上下文：K 线、VIX、跨式权利金、持仓等 |
| `ObservationResult` | `@dataclass(frozen)` | `counts`（int32 spike 数）+ `neural_ms`（神经时间）+ `sim_ms`（累计）+ `compute_seconds`（墙钟） |
| `Prediction` | `@dataclass(frozen)` | `realized_over_implied`（实现/隐含波动比）+ `confidence`（0~1）+ `decision`（`ENTER`/`EXIT`/`HOLD`）+ `details` |

加上 3 个 `Protocol`：

- **`BrainProtocol`**：4 个方法 `observe` / `checkpoint` / `restore` / `provenance` + 属性 `n` / `populations`
- **`EncoderProtocol`**：`encode(sf, brain) → Stimulus` + `config_hash() → str`
- **`ReadoutProtocol`**：`predict(counts, brain, sf) → Prediction` + `config_hash() → str`

### 3.3 必须存在的 16 个脑区（`REQUIRED_POPULATIONS`）

```python
from flybrain import REQUIRED_POPULATIONS
# ('R1-R6', 'R8p', 'R8y', 'lamina', 'KC', 'PAM11', 'PPL101',
#  'MBON07', 'MBON11', 'MBON', 'DNp20_L', 'DNp20_R', 'DNpe017',
#  'DN', 'central_complex', 'random2000')
```

任何实现 `BrainProtocol` 的类都必须把这 16 个键放进 `self.populations`，值是 `int32` 的神经元编号数组。`FixedDecoder` 默认看 `DNp20_L` / `DNp20_R` / `DNpe017` 这三个决策区。

### 3.4 `Decision` 枚举

```python
class Decision(str, Enum):
    ENTER = "ENTER"   # 开仓（跨式卖出）
    EXIT  = "EXIT"    # 平仓
    HOLD  = "HOLD"    # 维持当前状态
```

由于 `Decision` 继承 `str`，可直接 `decision.value == "ENTER"`。

---

## 第 4 章：30 秒上手

下面这段**完整可运行**（不需要真脑图）。它构造一个 `BrainProtocol` 的最小桩，然后跑 3 步管道。

```python
"""
End-to-end flybrain example using a stub Brain.
Run: python examples/quickstart.py
"""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np

from flybrain import (
    REQUIRED_POPULATIONS, BrainProtocol, Decision,
    ObservationResult, Prediction, SensorFrame, Stimulus,
)
from flybrain.encoders import make_encoder
from flybrain.readout.fixed import FixedDecoder


# ① StubBrain：256 个神经元的最小桩，无真脑图也能跑通
@dataclass
class StubBrain(BrainProtocol):
    n_neurons: int = 256
    seed: int = 42

    def __post_init__(self):
        self.n = int(self.n_neurons)
        self._rng = np.random.default_rng(self.seed)
        self.populations = {}
        cur = 0
        for name in REQUIRED_POPULATIONS:                # 把 16 个脑区填进 dict
            sz = max(8, self.n // 32)
            if cur + sz > self.n:
                sz = max(1, self.n - cur)
            self.populations[name] = np.arange(cur, cur + sz, dtype=np.int32)
            cur += sz
            if cur >= self.n:
                cur = self.n - 1

    def observe(self, stimulus: Stimulus, neural_ms: float) -> ObservationResult:
        """伪 spike 数：均值驱动越强 → Poisson 越密"""
        r16 = float(np.mean(stimulus.r16)) if stimulus.r16.size else 0.0
        r8  = float(np.mean(stimulus.r8))  if stimulus.r8.size  else 0.0
        drive = 0.5 * (r16 + r8)
        sec = max(neural_ms, 1e-9) / 1000.0
        lam = (2.0 + 25.0 * drive) * sec
        counts = self._rng.poisson(lam, size=self.n).astype(np.int32)
        return ObservationResult(
            counts=counts, neural_ms=float(neural_ms),
            sim_ms=0.0, compute_seconds=0.0,
        )

    def checkpoint(self, path: str) -> None:  pass
    def restore(self, path: str)   -> None:  pass
    def provenance(self) -> dict:             return {"stub": True}


# ② 构造 SensorFrame（你只需要喂过去的 bar）
IST = timezone(timedelta(hours=5, minutes=30))
sf = SensorFrame(
    timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=IST),
    index_bars=(), vix=14.2, vix_bars=(),
    straddle_premium=120.5, entry_credit=None,
    days_to_expiry=4.0, minutes_since_open=10, position_lots=0,
)

# ③ 4 步管道
brain  = StubBrain()
stim   = make_encoder("bars").encode(sf, brain)
result = brain.observe(stim, neural_ms=200.0)
pred   = FixedDecoder(neural_ms=200.0).predict(result.counts, brain, sf)

print(f"Stimulus:   r16={stim.r16.shape}  r8={stim.r8.shape}")
print(f"Spikes:     {result.counts.sum()} over {brain.n} neurons in {result.neural_ms} ms")
print(f"Prediction: decision={pred.decision.value}  "
      f"roi={pred.realized_over_implied:.3f}  "
      f"conf={pred.confidence:.3f}")
# 输出（Poisson 随机）：
# Stimulus:   r16=(8,)  r8=(16,)
# Spikes:     ~488 over 256 neurons in 200.0 ms
# Prediction: decision=HOLD roi=1.000 conf=1.000
```

---

## 第 5 章：完整 API 参考

> 全部 31 个公开符号。除特别说明，所有 dataclass 都是 `frozen=True`。

### 5.1 契约层（`flybrain.interfaces`）

#### 5.1.1 `Stimulus`
```python
@dataclass(frozen=True)
class Stimulus:
    r16: np.ndarray                    # float32, len=n_r16, 范围 [0, 1]
    r8:  np.ndarray                    # float32, len=n_r8,  范围 [0, 1]
    pulses: tuple[tuple[str, float, float], ...] = ()
    #               (种群名, 电流 mV, 持续 ms)，用于多巴胺刺激
```

#### 5.1.2 `ObservationResult`
```python
@dataclass(frozen=True)
class ObservationResult:
    counts: np.ndarray      # int32, shape=(n,)
    neural_ms: float        # 本次模拟的神经时间（一般 100-300 ms）
    sim_ms: float           # 累计模拟时间（用于复算）
    compute_seconds: float    # 墙钟耗时
```

#### 5.1.3 `BrainProtocol`
```python
@runtime_checkable
class BrainProtocol(Protocol):
    n: int              # 神经元总数
    populations: dict[str, np.ndarray]   # 必须含 REQUIRED_POPULATIONS 全部 16 个

    def observe(self, stimulus: Stimulus, neural_ms: float) -> ObservationResult: ...
    def checkpoint(self, path: str) -> None: ...
    def restore(self, path: str) -> None: ...
    def provenance(self) -> dict: ...
```

#### 5.1.4 `SensorFrame`
```python
@dataclass(frozen=True)
class SensorFrame:
    timestamp: datetime                                  # tz-aware，Asia/Kolkata
    index_bars: tuple[Bar, ...]                          # NIFTY 5min K 线窗口，旧→新
    vix: float                                           # INDIAVIX 最新值
    vix_bars: tuple[Bar, ...]                            # VIX K 线窗口
    straddle_premium: float | None                       # ATM 跨式组合权利金
    entry_credit: float | None                           # 入场时收到的权利金
    days_to_expiry: float
    minutes_since_open: int
    position_lots: int                                   # 负=空跨式, 0=空仓
```

#### 5.1.5 `Prediction` / `Decision`
```python
@dataclass(frozen=True)
class Prediction:
    realized_over_implied: float     # 实现/隐含波动比，>1 表示超波动
    confidence: float                # 0~1
    decision: Decision
    details: dict = {}               # 解码器自填的调试字段

class Decision(str, Enum):
    ENTER = "ENTER"
    EXIT  = "EXIT"
    HOLD  = "HOLD"
```

#### 5.1.6 `EncoderProtocol` / `ReadoutProtocol`
两个都是运行时可检查的协议。`name`（str）和 `config_hash() → str` 是必须属性。

---

### 5.2 视网膜坐标（`flybrain.eyemap`）

#### 5.2.1 `EyeMap`
```python
@dataclass(frozen=True)
class EyeMap:
    uv_r16:     np.ndarray    # float32 (n_r16, 2)，每只眼 R1-R6 的 uv 坐标
    eye_r16:    np.ndarray    # int8   (n_r16,)，0=左眼 1=右眼
    uv_r8:      np.ndarray    # float32 (n_r8,  2)
    eye_r8:     np.ndarray    # int8   (n_r8,)
    r8_channel: np.ndarray    # int8   (n_r8,)，1=R8y(绿) 2=R8p(蓝)
    @property
    def n_r16(self) -> int
    @property
    def n_r8(self)  -> int
    def validate(self) -> None
```

#### 5.2.2 `default_eye_map(brain)` / `resolve_eye_map(brain)`
- `default_eye_map(brain)`：从脑的 populations 派生确定性默认映射（无外部坐标时使用）。
- `resolve_eye_map(brain)`：返回 `brain.eye_map()`，不存在则降级到 `default_eye_map`。
- `as_eye_map(obj)`：接受 `EyeMap` / 任何含五字段对象 / dict。

---

### 5.3 编码器（`flybrain.encoders`）

3 个内置 encoder + 1 个工厂：

#### 5.3.1 `ChartEncoder`（蜡烛图）
```python
ChartEncoder(eye_map=None, *,
             width=320, height=180, bars=60, line_width=3,
             atr_window=14, atr_multiple=3.0,
             background=(235,235,235),
             rising=(0,0,255), falling=(255,0,0),
             margin=6)
```
方法：
- `encode(sf, brain=None) → Stimulus`
- `render_png(stim, eye_map=None) → bytes`  ← PNG 字节流，便于可视化
- `config_hash() → str`

#### 5.3.2 `BarsEncoder`（K 线柱，柱状编码）
```python
BarsEncoder(eye_map=None, *,
            columns=30, std_window=60, atr_window=14,
            range_atr_multiple=2.0, vix_window=20)
```

#### 5.3.3 `FeatureEncoder`（标量特征）
```python
FeatureEncoder(eye_map=None, *,
               grid=3, std_window=60, realized_bars=12,
               horizon_minutes=60, session_minutes=375,
               expiry_scale_days=25.0)
```
特征顺序由 `FEATURE_NAMES` 常量给出。

#### 5.3.4 `make_encoder(name, settings=None, eye_map=None)`
`name` 可用别名（`ENCODER_NAMES`）：`"A" / "B" / "C"` 或 `"chart" / "bars" / "features"`。未知名 → 抛 `ValueError`。

#### 5.3.5 公共工具
- `stimulus_hash(stimulus) → str`：对 r16 + r8 算 sha256，用于复算验证。
- `atr(high, low, close, window=14) → float`：经典 ATR。
- `log_returns(close) → np.ndarray`
- `rolling_return_std(close, window=60, floor=1e-6) → float`
- `vix_zscore(sf, window=20) → float`

---

### 5.4 LIF 核（`flybrain.kernel`）

#### 5.4.1 常量
| 常量 | 值 | 含义 |
|---|---|---|
| `KERNEL_VERSION` | `"openfly-lif-1.1"` | 当前核版本，写入 provenance |
| `DT_MS` | `0.1` | 仿真步长 |
| `TAU_M_MS` | `20.0` | 膜时间常数 |
| `TAU_S_MS` | `5.0` | 突触时间常数 |
| `TAU_A_MS` | `200.0` | 适应时间常数 |
| `V_THRESH_MV` | `-45.0` | 阈值电压 |
| `KERNEL_PARAMETERS` | dict | 上面所有参数的字典 |

#### 5.4.2 `Kernel` 类
Numba `@njit(cache=True)` 编译的 LIF 核。一般用户**不需要直接构造**（`Brain.__init__` 会自动建），但你写自定义 Brain 时需要：
```python
from flybrain.kernel import Kernel
Kernel(ptr, post, weight, modulatory, is_kc)
```
- `ptr`：`np.int64`, shape `(n+1,)`，CSR 指针
- `post`：目标神经元编号
- `weight`：浮点权重（已按 NT 符号修正）
- `modulatory`：bool 数组，调制性连接不发放 spike
- `is_kc`：uint8 数组，标记 KC 种群

#### 5.4.3 `KernelState` 类
膜时间/种记忆 `(n, 8)` float64 数组。同样一般无需直接构造。

---

### 5.5 可塑性（`flybrain.plasticity`）

#### 5.5.1 `MV_PER_CONTACT`
`0.275` mV（一个突触接触的标准反应幅度）。

#### 5.5.2 `PlasticityConfig`
```python
@dataclass(frozen=True)
class PlasticityConfig:
    eta: float = 0.001                  # 学习率
    trace_tau_ms: float = 1000.0        # 资格迹时间常数
    memory_tau_ms: float = 1_800_000.0  # 长时记忆衰减（30 分钟）
    filter_tau_ms: float = 50.0         # 多巴胺滤波
    factor_min: float = 0.1
    factor_max: float = 2.0
    compartments: tuple[tuple[str, str], ...] = (("PAM11", "MBON07"),
                                                 ("PPL101", "MBON11"))
    def as_dict(self) -> dict
```

#### 5.5.3 `KCMBONPlasticity`
KC → MBON 突触的 Oja-style 可塑性。**仅当 `Brain(plastic=True)` 时启用**，需要 `Brain.checkpoint` 保存权重因子。

---

### 5.6 Brain（`flybrain.brain`）

#### 5.6.1 `Brain` 类
```python
Brain(graph_path=None,                # 默认 PATHS.graph
     half_saturation=DEFAULT_HALF_SATURATION,
     plastic=False,
     plasticity_config=None,          # 默认 PlasticityConfig()
     check_counts=True,
     graph=None,                      # 直接注入 dict，跳过文件装载
     r8_ame12_excitatory=True)        # 是否把 R8→aMe12 改为兴奋性
```
- 自动调用 `build_populations(g)` 构建 16 区映射
- 自动校验 16 区大小一致（`check_population_sizes`）
- `observe(stim, neural_ms)` 返回 spike counts
- `checkpoint(path)` / `restore(path)`：序列化权重 + 状态
- `provenance() → dict`：包含 graph_path、version、populations 规则、随机种子等
- `.populations`：dict[str, np.ndarray]，含全部 16 区
- `.kernel`：Numba `Kernel` 实例
- `.r16 / .r8 / .lamina`：常用区下标数组

#### 5.6.2 `load_graph(path)` 读 `.npz`
返回 `dict[str, np.ndarray]`，必需键：
- `ptr` / `post` / `weight` / `modulatory`（CSR 边表）
- `type` / `superclass`（str 数组）
- `r16_uv` / `r8_uv`（float32）
- `r16_eye` / `r8_eye` / `r8_channel`（int8）

#### 5.6.3 `build_populations(g)`
从 graph dict 推导 `populations` dict（16 个区）。算法见 brain.py 注释，可独立测试。

#### 5.6.4 `r8_ame12_edges(g)` 
返回 R8 → aMe12 边的下标数组，用于选择性改符号。

---

### 5.7 读出器（`flybrain.readout.*`）

#### 5.7.1 `FixedDecoder`
```python
FixedDecoder(neural_ms=200.0,
             threshold_hz=2.0,
             left="DNp20_L",        # 决策左半球
             right="DNp20_R",       # 决策右半球
             gate="DNpe017",        # 置信门控
             confidence_spikes=5.0)
```
方法：
- `bind_columns(neuron_ids)`：可选，绑定按 view 切片时的列。
- `decode(counts, brain) → dict`：返回 `{side, difference_hz, gate_spikes}`。
- `predict(counts, brain, sf=None) → Prediction`：返回最终 Prediction。**当 `sf.position_lots != 0` 而 signal=ENTER 时强制 HOLD**（持仓保护）。
- `predict_batch(counts, brain, neural_ms=None) → dict`：批量矢量化，返回 `{roi, signal, difference_hz, gate_spikes}`。
- `config_hash() → str`

#### 5.7.2 `ReservoirReadout`（需要 `[readout]`）
```python
ReservoirReadout(populations=DEFAULT_POPULATIONS,
                alpha=10.0,                # ridge 正则化
                tau=0.1,                   # 时间平滑
                horizon_minutes=60,
                neural_ms=200.0,
                brain=None,                # 可选，直接 resolve
                classifier=True,           # 是否同时训练 logistic
                std_floor=1e-6)
```
方法：
- `resolve(brain)`：从脑 populations 推导特征索引。
- `bind_columns(neuron_ids)` / `set_neuron_index(ids)`
- `features(X) → np.ndarray`：把 counts 转为时序平均特征。
- `fit(X, y, columns=None, classifier=None)`：训练。X 可以是 `(n_obs, n_neurons)` 的原始 spike；`columns` 给出列对应的神经元 ID。
- `refit_alpha(alpha)`：用缓存的 eigendecomp 重算不同 alpha，无需重训。
- `fit_classifier(Z, labels)` / `fit_classifier_from_counts(X, y)`
- `predict(counts, brain=None, sf=None) → Prediction`
- `predict_batch(X, brain=None) → dict`
- `config_hash() → str`

---

### 5.8 连接组工具链（`flybrain.connectome.*`）

#### 5.8.1 元数据（`sources`）
```python
from flybrain.connectome import SOURCES, Source, URL_PREFIX, LICENSE, DATASET
# SOURCES: 3 个 Source（annotations / neurotransmitters / weights）
# Source 属性：role, name, bytes, sha256
# Source 方法：.url, .path(root=None) → Path
```

#### 5.8.2 下载（`download`，无强制依赖）
```python
from flybrain.connectome import download_source, is_verified, DownloadError

path = download_source(src: Source,
                      root: Path | None = None,           # 默认 PATHS.malecns
                      progress: Callable[[int, int], None] | None = None,
                      verify_existing: bool = True) → Path
```
**特性**：断点续传（`.partial` + Range header）；失败自动重置；写入即校验。

#### 5.8.3 编译（`compile`，需 `[feather]`）
```python
from flybrain.connectome import (
    compile_graph, read_manifest, normalize_uv, photoreceptor_geometry,
    GRAPH_FORMAT_VERSION,
)

graph_path = compile_graph(malecns_dir: Path,             # 默认 PATHS.malecns
                          out: Path | None = None,        # 默认 PATHS.graph
                          *, snapshot: bool = True) → Path
# 返回 PATHS.graph 路径，并写入 manifest.json + .lock
manifest = read_manifest(graph_path) → dict | None
```

#### 5.8.4 标准化（`normalize`，需 `[feather]`）
```python
from flybrain.connectome import (
    sign_from_nt, tokenize_nt, load_annotations, node_arrays,
)
```
NT 符号规则（标准化神经元递质 → ±1 符号）见 `normalize.py` 注释。

#### 5.8.5 校验（`verify`）
```python
from flybrain.connectome.verify import (
    sha256_file, sha256_array,
    lock_path, manifest_path,
    array_hashes, graph_hashes,
    write_lock, read_lock,
    verify_sources, verify_graph,
)
```

---

### 5.9 路径管理（`flybrain.paths`）

```python
from flybrain.paths import PATHS, Paths

# PATHS 实例（frozen）
PATHS.root      # 默认 ~/.flybrain
PATHS.malecns   # 默认 ~/.flybrain/malecns   feather 源文件目录
PATHS.graph     # 默认 ~/.flybrain/graph.npz 编译产物
PATHS.ensure()  # 自动 mkdir -p
```

**覆盖方式**：设置环境变量 `FLYBRAIN_DATA`（在 import 之前）：
```python
import os
os.environ["FLYBRAIN_DATA"] = "/scratch/flybrain"
from flybrain.paths import PATHS
```

---

## 第 6 章：实战场景（7 个）

### 场景 1：跑通完整接口（不需要真脑图）

直接跑 `examples/quickstart.py`，见第 4 章。**这个例子是**真单测的子集，可作为你的开发起点。

---

### 场景 2：切换 3 种编码器

```python
from flybrain.encoders import make_encoder

for name in ["bars", "chart", "feature"]:
    enc = make_encoder(name)
    stim = enc.encode(sf, brain)
    print(f"{name:8s} → r16={stim.r16.shape}, r8={stim.r8.shape}, hash={enc.config_hash()[:12]}")
```

**输出**（按 encoder 配置决定）：
```
bars      → r16=(8,),  r8=(16,), hash=...
chart     → r16=(160,), r8=(160,), hash=...    # 320x180 网格映射后
feature   → r16=(15,), r8=(15,), hash=...
```

`chart` 编码器额外提供 PNG 字节流：
```python
png_bytes = enc.render_png(stim)   # 蜡烛图
with open("stimulus.png", "wb") as f:
    f.write(png_bytes)
```

---

### 场景 3：自定义 FixedDecoder 决策规则

```python
from flybrain.readout.fixed import FixedDecoder

decoder = FixedDecoder(
    neural_ms=300.0,            # 模拟更久（更稳）
    threshold_hz=1.5,           # 左右差 ≥ 1.5 Hz 才出信号
    left="MBON07",              # 用 MBON 区做决策
    right="MBON11",
    gate="PAM11",               # 用多巴胺区做置信
    confidence_spikes=8.0,
)
pred = decoder.predict(counts, brain, sf)
print(pred.decision, pred.confidence, pred.details)
```

**注意**：种群名必须在 `brain.populations` 里有，否则 `decode` 会抛 `KeyError`。

---

### 场景 4：完整接入 MaleCNS v1.0 真脑图

**前置**：`pip install flybrain[feather,download]`

```python
import flybrain
from flybrain.paths import PATHS
from flybrain.connectome import (
    SOURCES, download_source, is_verified, compile_graph,
)
from flybrain import Brain

# ① 准备目录
PATHS.ensure()

# ② 下载（首次 ~1.1 GB；自动断点续传 + sha256 校验）
for src in SOURCES:
    print(f"下载 {src.name} ({src.bytes/1e6:.1f} MB)...")
    path = download_source(src, progress=lambda d, t: print(f"  {d}/{t}"), )
    assert is_verified(src)
print("全部下载完成 ✓")

# ③ 编译成 .npz（5-10 分钟，结束后写入 PATHS.graph）
graph_path = compile_graph(PATHS.malecns)
print(f"编译完成 → {graph_path}")

# ④ 装载到 Brain（首次跑 Numba JIT 编译，慢；之后用 cache）
brain = Brain(graph_path=graph_path, plastic=False)
print(f"神经元 {brain.n:,} 个, 突触 {brain.edges:,} 个")

# ⑤ 跑一次观察
result = brain.observe(stim, neural_ms=200.0)
print(f"spike 数：{result.counts.sum()} in {result.neural_ms} ms")
```

**加速建议**：把 `cache=True`（`Kernel.__init__` 默认已是 `True`）保留；`Brain(half_saturation=...)` 调大 → spike 更快但更稀疏。

---

### 场景 5：用 ReservoirReadout 训练 + 推理

**前置**：`pip install flybrain[readout]`

```python
import numpy as np
from flybrain.readout.reservoir import ReservoirReadout

# ① 准备历史数据（伪）
n_obs, n_neurons = 1000, 256
X = np.random.poisson(2.0, size=(n_obs, n_neurons)).astype(np.int32)
y = (X[:, :50].sum(axis=1) > 110).astype(np.float64)   # 标签

# ② 构造 + 训练
readout = ReservoirReadout(brain=brain, alpha=10.0, classifier=True)
readout.fit(X, y)
print(f"fit 完成: n_features={readout.n_features}, n_obs={readout.fit_info['n_obs']}")

# ③ 推理
current_counts = X[0]
pred = readout.predict(current_counts, brain=brain, observation=sf)
print(pred.decision, pred.confidence, pred.details)

# ④ 重调 ridge alpha（无需重训）
readout.refit_alpha(1.0)

# ⑤ 批量推理
batch_pred = readout.predict_batch(X[:32], brain=brain)
print(batch_pred.keys())   # dict_keys(['roi', 'signal', 'decision', 'confidence'])
```

---

### 场景 6：自定义 EyeMap（覆盖脑自带）

```python
from flybrain.eyemap import EyeMap, as_eye_map, resolve_eye_map

# ① 自定义 5 字段
custom = EyeMap(
    uv_r16=np.array([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
    eye_r16=np.array([0, 0, 1], dtype=np.int8),
    uv_r8=np.array([[0.5, 0.5]], dtype=np.float32),
    eye_r8=np.array([1], dtype=np.int8),
    r8_channel=np.array([2], dtype=np.int8),
)
custom.validate()

# ② 让 encoder 用你的映射
encoder = make_encoder("chart", eye_map=custom)
stim = encoder.encode(sf, brain)

# ③ 或直接用默认（脑不带 .eye_map() 时）
eme = resolve_eye_map(brain)        # 优先 brain.eye_map()，否则 default_eye_map
print(eme.n_r16, eme.n_r8)
```

---

### 场景 7：checkpoint + restore（保留可塑性状态）

```python
brain = Brain(graph_path=graph_path, plastic=True)
brain.observe(stim_1, neural_ms=200.0)
brain.observe(stim_2, neural_ms=200.0)

# 保存当前状态（含权重因子 + KC/MBON traces）
brain.checkpoint("./brain_state/")

# 稍后恢复
brain2 = Brain(graph_path=graph_path, plastic=True)
brain2.restore("./brain_state/")
assert brain2.observations == brain.observations

# 追踪 provenance
print(brain.provenance())
# {'graph_path': '...', 'kernel_version': 'openfly-lif-1.1',
#  'n': ..., 'edges': ..., 'populations': {...规则...}, ...}
```

---

## 第 7 章：数据路径与环境变量

### 7.1 默认路径

```python
from flybrain.paths import PATHS
print(PATHS.root)        # Windows: C:\Users\<user>\.flybrain
                         # Linux/macOS: ~/.flybrain
print(PATHS.malecns)     # ~/.flybrain/malecns
print(PATHS.graph)       # ~/.flybrain/graph.npz
```

### 7.2 覆盖方式（优先级从高到低）

| 方式 | 何时生效 | 示例 |
|---|---|---|
| `os.environ["FLYBRAIN_DATA"]` | import flybrain 之前 | `os.environ["FLYBRAIN_DATA"] = "/data/fb"` |
| 直接修改 `PATHS` 实例 | 不行（frozen） | — |
| 临时目录 | `tempfile.mkdtemp()` + 改 `FLYBRAIN_DATA` | 见测试 `test_paths.py` |

**注意**：环境变量必须在 `from flybrain.paths import PATHS` **之前**设置；PATHS 是 module-import 时冻结的。

### 7.3 磁盘占用估算

| 阶段 | 内容 | 大小 |
|---|---|---|
| feather 源（malecns/） | 3 个 .feather 文件 | ~1.1 GB |
| 编译产物（graph.npz） | 数组 + manifest + lock | ~80 MB |
| Numba cache（~/.numba_cache/） | 编译产物 | ~30 MB |

---

## 第 8 章：性能、调参与调试

### 8.1 Numba JIT 加速

第一次跑 `Brain.observe` 会触发 JIT 编译（**慢 5-30 秒**），之后用 `~/.numba_cache/` 缓存（`cache=True`）。

**调试建议**：临时关闭缓存：
```python
import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba"
```
或在 Numba 报 `TypingError` 时开 debug：
```python
os.environ["NUMBA_DEBUG"] = "1"
```

### 8.2 减少观察时间

`neural_ms=200.0` 是默认值。要更快 → 调小：
```python
result = brain.observe(stim, neural_ms=100.0)
```
但**太小**会让 FixedDecoder 的左右频差被噪声淹没。建议 150-300 ms。

### 8.3 调整发放稀疏度

```python
brain = Brain(graph_path=graph_path, half_saturation=10.0)   # 默认值，见 DEFAULT_HALF_SATURATION
# half_saturation ↑ → 输入电流更难到达阈值 → spike 更稀疏
# 适合：低信号场景（夜间）
```

### 8.4 调试 timing：把 `ObservationResult.compute_seconds` 当 wall-clock

```python
import time
t0 = time.perf_counter()
result = brain.observe(stim, neural_ms=200.0)
elapsed = time.perf_counter() - t0
assert result.compute_seconds <= elapsed + 0.01
```

### 8.5 用 `Decision` 字符串 guard 当需求

```python
assert pred.decision.value in {"ENTER", "EXIT", "HOLD"}
if pred.decision == Decision.HOLD:
    ...   # 比 .value 比较快
```

---

## 第 9 章：测试与开发

### 9.1 跑测试

```bash
cd D:/date/20260918/flybrain
.venv/Scripts/python.exe -m pytest -v           # 22 个测试
.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v   # 仅端到端
```

**当前覆盖率**：
- `tests/test_imports.py` (19 个)：契约 + 构造 + encoder 工厂 + kernel 常量
- `tests/test_pipeline.py` (3 个)：BrainProtocol 满足 + encode→decode→predict + 持仓强制 HOLD

### 9.2 Lint

```bash
.venv/Scripts/python.exe -m ruff check src tests examples
.venv/Scripts/python.exe -m ruff format src tests examples    # 格式化
```

### 9.3 加新测试

```python
# tests/test_my_feature.py
from flybrain import ...

def test_my_new_encoder():
    enc = MyEncoder()
    assert enc.config_hash() == "expected_sha256"
    stim = enc.encode(my_sensor_frame, my_brain)
    assert stim.r16.shape == (16,)
```

### 9.4 添加新 extras

```toml
# pyproject.toml
[project.optional-dependencies]
my_new_extra = ["package1>=1.0", "package2"]

# 然后：
[tool.flybrain.extras]
my_new_extra = "描述何时使用"
```

---

## 第 10 章：常见问题（FAQ）

### Q1：装好 `flybrain` 但 `import flybrain.connectome` 抛 ImportError？
A：connectome 子模块大多用了 `__getattr__` 懒加载。`feather` extra 缺失时 `compile_graph` / `normalize_uv` 等会报 ImportError。解决：`pip install flybrain[feather,download]`。

### Q2：`Brain.load_graph()` 报 `FileNotFoundError`？
A：`PATHS.graph` 默认不存在。你必须先 `compile_graph(PATHS.malecns)` 生成它；或者直接传 `graph_path="your/path/to/graph.npz"`。

### Q3：StubBrain 跑出来的 `decision` 总是 HOLD？
A：StubBrain 的 spike 是 Poisson 随机，可能左右频差总 < `threshold_hz=2.0`。调小 `threshold_hz` 或增大 `lam`。

### Q4：`ReservoirReadout.fit()` 报 "need at least two observations"？
A：至少给 2 个样本（`X.shape[0] ≥ 2`）。

### Q5：能不能只装 `readout` extra 但不用真脑图？
A：可以。`ReservoirReadout.__init__` 接受 `brain=None`，之后 `predict(counts, brain=None)` 也行（但 `predict(counts, brain=stub)` 才能用 `bind_columns`）。

### Q6：Numba 报 `NoSuchModule` 或 `Cannot link program`？
A：Windows 下一般要先装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)。Linux/Mac 一般无此问题。

### Q7：`paths.PATHS` 是单例吗？
A：是。`from flybrain.paths import PATHS` 永远拿到同一个 frozen 实例。`FLYBRAIN_DATA` 只在 import 前有效。

### Q8：能不能打包成 wheel 发布？
A：可以。`pyproject.toml` 已经是 hatchling。`hatch build` → `dist/flybrain-0.1.0-py3-none-any.whl`。

### Q9：`Prediction.details` 是什么结构？
A：每种 decoder 自定义，但必有 `readout` 字段（decoder name）和 `signal` 字段（ENTER/EXIT/HOLD）。`FixedDecoder` 还填 `difference_hz` / `gate_spikes` / `threshold_hz`；`ReservoirReadout` 填 `ridge_value` / `clf_prob`。

### Q10：自定义 encoder 一定要继承 `_BaseEncoder` 吗？
A：不一定。只要实现 `EncoderProtocol`（`name: str` + `encode(sf, brain) → Stimulus` + `config_hash()`）就行。用 `_BaseEncoder` 是为了复用 `render_png` / 缓存 `stimulus_hash`。

---

## 附录 A：完整管道图

```
┌──────────┐  ┌────────────┐  ┌─────────┐  ┌──────────┐  ┌────────────┐
│ 你的数据 │→ │   Encoder   │→ │ Stimulus│→ │   Brain  │→ │  counts   │
│ (Bar /  │  │ (chart/bars │  │ (r16,r8,│  │ (Numba   │  │ (int32    │
│  Feature)│  │  /features) │  │ pulses) │  │  LIF)    │  │  shape=n) │
└──────────┘  └────────────┘  └─────────┘  └──────────┘  └────────────┘
  SensorFrame  EncoderProtocol  frozen       BrainProtocol   ObservationResult
                                                              │
                                                              ↓
                                            ┌──────────┐  ┌──────────┐
                                            │ Readout  │→ │Prediction│
                                            │(fixed /  │  │(decision,│
                                            │reservoir)│  │ conf,    │
                                            │          │  │ roi)     │
                                            └──────────┘  └──────────┘
                                            ReadoutProtocol
```

**最小调用栈**：
```python
stim = encoder.encode(sf)
result = brain.observe(stim)
pred = decoder.predict(result.counts, brain, sf)
```

**Brain 内部**：
```
Stimulus
  → drive = stimulus.r16 / r8 [长度外 L1-R6 / R8 数]
  → 步进 neural_ms / DT_MS 次 Numba run_steps(...)
  → counts[i] = spike 累积数
  → 返回 ObservationResult(counts, neural_ms, sim_ms, compute_seconds)
```

---

## 附录 B：文件清单

| 文件 | 作用 | 行数（约） |
|---|---|---|
| `src/flybrain/__init__.py` | 31 个公开符号聚合 | 90 |
| `src/flybrain/paths.py` | PATHS 实例 + `FLYBRAIN_DATA` 覆盖 | 60 |
| `src/flybrain/interfaces.py` | 4 dataclass + 3 Protocol + Decision | 280 |
| `src/flybrain/encoders.py` | 3 encoder + make_encoder + 工具 | 540 |
| `src/flybrain/eyemap.py` | EyeMap + 3 helper | 170 |
| `src/flybrain/kernel.py` | Numba `@njit` LIF 核 + 常量 | 460 |
| `src/flybrain/plasticity.py` | Oja-style KC→MBON + PlasticityConfig | 200 |
| `src/flybrain/brain.py` | `Brain` 类 + `load_graph` + `build_populations` | 510 |
| `src/flybrain/readout/fixed.py` | `FixedDecoder` + `ColumnBinding` | 200 |
| `src/flybrain/readout/reservoir.py` | `ReservoirReadout` + ridge + logistic | 290 |
| `src/flybrain/connectome/sources.py` | 3 个 Source + sha256 + URL | 100 |
| `src/flybrain/connectome/download.py` | 断点续传 + sha256 校验 | 150 |
| `src/flybrain/connectome/normalize.py` | NT → 符号规则 | 200 |
| `src/flybrain/connectome/compile.py` | feather → .npz | 470 |
| `src/flybrain/connectome/verify.py` | 哈希 + 锁文件 + 验证 | 130 |
| `tests/test_imports.py` | 19 个 import / 构造测试 | 200 |
| `tests/test_pipeline.py` | 3 个端到端测试 | 110 |
| `examples/quickstart.py` | 30 秒可运行 demo | 130 |

---

**祝你用得开心。** 🪰

如发现文档错误或 API 缺失，请编辑本仓库 `flybrain/docs/USAGE.zh.md`，或在 issue 中给出复现命令。