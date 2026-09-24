"""flybrain: a Numba-JIT accelerated fruit-fly connectome.

Extracted from OpenFly (github.com/marketcalls/openfly). Independent of
broker / market / front-end code.
"""
from flybrain import readout
from flybrain.brain import Brain, build_populations, load_graph, r8_ame12_edges
from flybrain.encoders import (
    ENCODER_NAMES,
    FEATURE_NAMES,
    BarsEncoder,
    ChartEncoder,
    FeatureEncoder,
    make_encoder,
)
from flybrain.eyemap import EyeMap, as_eye_map, default_eye_map, resolve_eye_map
from flybrain.interfaces import (
    REQUIRED_POPULATIONS,
    BrainProtocol,
    Decision,
    EncoderProtocol,
    ObservationResult,
    Prediction,
    ReadoutProtocol,
    SensorFrame,
    Stimulus,
)
from flybrain.kernel import (
    KERNEL_PARAMETERS,
    KERNEL_VERSION,
    Kernel,
    KernelState,
)
from flybrain.plasticity import MV_PER_CONTACT, KCMBONPlasticity, PlasticityConfig

__version__ = "0.1.0"

__all__ = [
    # contracts
    "Stimulus",
    "ObservationResult",
    "BrainProtocol",
    "REQUIRED_POPULATIONS",
    "SensorFrame",
    "Decision",
    "Prediction",
    "EncoderProtocol",
    "ReadoutProtocol",
    # eyemap
    "EyeMap",
    "as_eye_map",
    "default_eye_map",
    "resolve_eye_map",
    # encoders
    "ChartEncoder",
    "BarsEncoder",
    "FeatureEncoder",
    "make_encoder",
    "FEATURE_NAMES",
    "ENCODER_NAMES",
    # kernel
    "Kernel",
    "KernelState",
    "KERNEL_PARAMETERS",
    "KERNEL_VERSION",
    # plasticity
    "KCMBONPlasticity",
    "PlasticityConfig",
    "MV_PER_CONTACT",
    # brain
    "Brain",
    "load_graph",
    "build_populations",
    "r8_ame12_edges",
    # subpackage
    "readout",
]
