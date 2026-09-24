"""The three MaleCNS v1.0 flat-connectome source files.

Sizes and SHA-256 digests are pinned here so a download is only accepted when
it matches byte for byte. License: CC-BY 4.0 (HHMI Janelia FlyEM, University
of Cambridge, MRC LMB, Google Research).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flybrain.paths import PATHS

URL_PREFIX = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
LICENSE = "CC-BY 4.0"
DATASET = "MaleCNS v1.0 flat connectome"


@dataclass(frozen=True)
class Source:
    role: str
    name: str
    bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return URL_PREFIX + self.name

    def path(self, root: Path | None = None) -> Path:
        return (root or PATHS.malecns) / self.name


SOURCES: tuple[Source, ...] = (
    Source(
        role="annotations",
        name="body-annotations-male-cns-v1.0-minconf-0.5.feather",
        bytes=14_483_314,
        sha256="2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    ),
    Source(
        role="neurotransmitters",
        name="body-neurotransmitters-male-cns-v1.0.feather",
        bytes=43_282_834,
        sha256="95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    ),
    Source(
        role="weights",
        name="connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        bytes=1_051_241_946,
        sha256="e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    ),
)


def source(role: str) -> Source:
    for s in SOURCES:
        if s.role == role:
            return s
    raise KeyError(f"unknown connectome source role: {role}")
