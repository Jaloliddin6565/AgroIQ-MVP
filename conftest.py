"""Pytest uchun umumiy sozlamalar va fikstura'lar."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import load_crop_profiles, load_fertilizer_products, load_thresholds  # noqa: E402
from src.model_inference import load_artifact  # noqa: E402
from src.model_training import MODEL_PATH  # noqa: E402


@pytest.fixture(scope="session")
def thresholds() -> dict:
    return load_thresholds()


@pytest.fixture(scope="session")
def crop_profiles() -> dict:
    return load_crop_profiles()


@pytest.fixture(scope="session")
def fertilizer_products() -> dict:
    return load_fertilizer_products()


@pytest.fixture(scope="session")
def artifact() -> dict:
    """O'qitilgan model artefakti; mavjud bo'lmasa testlar o'tkazib yuboriladi."""

    if not MODEL_PATH.exists():
        pytest.skip("Model artefakti topilmadi — avval `python scripts/train_model.py` bajaring.")
    return load_artifact()
