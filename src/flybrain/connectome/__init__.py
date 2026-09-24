"""MaleCNS v1.0 connectome: download, normalize, compile.

Requires `[feather]` for normalize.py; download.py uses urllib only.
"""
from flybrain.connectome.sources import (
    DATASET,
    LICENSE,
    SOURCES,
    URL_PREFIX,
    Source,
)

__all__ = [
    "SOURCES",
    "Source",
    "URL_PREFIX",
    "LICENSE",
    "DATASET",
]

# Lazy imports for sub-modules (require optional extras)
def __getattr__(name):
    if name == "download_source":
        from flybrain.connectome.download import download_source
        return download_source
    if name == "is_verified":
        from flybrain.connectome.download import is_verified
        return is_verified
    if name == "DownloadError":
        from flybrain.connectome.download import DownloadError
        return DownloadError
    if name == "compile_graph":
        from flybrain.connectome.compile import compile_graph
        return compile_graph
    if name == "read_manifest":
        from flybrain.connectome.compile import read_manifest
        return read_manifest
    if name == "normalize_uv":
        from flybrain.connectome.compile import normalize_uv
        return normalize_uv
    if name == "photoreceptor_geometry":
        from flybrain.connectome.compile import photoreceptor_geometry
        return photoreceptor_geometry
    if name == "GRAPH_FORMAT_VERSION":
        from flybrain.connectome.compile import GRAPH_FORMAT_VERSION
        return GRAPH_FORMAT_VERSION
    if name == "sign_from_nt":
        from flybrain.connectome.normalize import sign_from_nt
        return sign_from_nt
    if name == "tokenize_nt":
        from flybrain.connectome.normalize import tokenize_nt
        return tokenize_nt
    if name == "load_annotations":
        from flybrain.connectome.normalize import load_annotations
        return load_annotations
    if name == "node_arrays":
        from flybrain.connectome.normalize import node_arrays
        return node_arrays
    raise AttributeError(f"module 'flybrain.connectome' has no attribute {name!r}")
