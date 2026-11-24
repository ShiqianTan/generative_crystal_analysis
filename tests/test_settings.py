from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import settings  # noqa: E402


def test_default_models_are_fieldforge_and_crystalite() -> None:
    assert [name for name, _ in settings.CIF_MODELS] == [
        "FieldForge (PT)",
        "Crystalite",
    ]


def test_default_paths_are_portable() -> None:
    repository = Path(__file__).resolve().parents[1]
    assert settings.REPOSITORY_ROOT == repository
    assert settings.DATA_ROOT.is_absolute()
    assert settings.OUTPUT_ROOT.is_absolute()
