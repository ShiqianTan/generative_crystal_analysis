"""Portable defaults shared by all analysis scripts.

The repository contains no research data.  Put files under ``data/`` using the
layout documented in the README, set the environment variables below, or pass
the corresponding command-line options directly.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


DATA_ROOT = _path_from_env("CRYSTAL_BENCHMARK_DATA_DIR", REPOSITORY_ROOT / "data")
OUTPUT_ROOT = _path_from_env("CRYSTAL_BENCHMARK_OUTPUT_DIR", REPOSITORY_ROOT / "outputs")

FIELD_FORGE_NAME = "FieldForge (PT)"
CRYSTALITE_NAME = "Crystalite"

FIELD_FORGE_CIF_DIR = _path_from_env(
    "FIELD_FORGE_CIF_DIR", DATA_ROOT / "cifs" / "fieldforge"
)
CRYSTALITE_CIF_DIR = _path_from_env(
    "CRYSTALITE_CIF_DIR", DATA_ROOT / "cifs" / "crystalite"
)
MLIP_JSON_DIR = _path_from_env("MLIP_JSON_DIR", DATA_ROOT / "mlip_json")
MP20_CSV = _path_from_env("MP20_CSV", DATA_ROOT / "reference" / "mp20_total.csv")

CIF_MODELS = [
    (FIELD_FORGE_NAME, str(FIELD_FORGE_CIF_DIR)),
    (CRYSTALITE_NAME, str(CRYSTALITE_CIF_DIR)),
]

MLIP_MODEL_SPECS = [
    (
        FIELD_FORGE_NAME,
        "FF_1500k_aa4_lemat_10seed-retry-samples-task*.json",
        str(FIELD_FORGE_CIF_DIR),
    ),
    (
        CRYSTALITE_NAME,
        "Crystalite-PCA-16dim-relaxed-structure-matcher-samples*.json",
        str(CRYSTALITE_CIF_DIR),
    ),
]
