"""Fosforni baholash: model yuklash, bashorat, noaniqlik va ishonch darajasi."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_validation import ConfigError, MODELS_DIR, load_thresholds
from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_color_features,
    check_within_calibration,
)

MODEL_PATH = MODELS_DIR / "phosphorus_model.joblib"

CONFIDENCE_LABELS_UZ = {
    "high": "Yuqori",
    "medium": "O'rtacha",
    "low": "Past",
}

OUT_OF_RANGE_MESSAGE_UZ = (
    "Ushbu namuna modelning validatsiya qilingan kalibrlash oralig'idan tashqarida. "
    "Laboratoriya tasdig'i tavsiya etiladi."
)


class ModelNotAvailableError(RuntimeError):
    """Model artefakti topilmaganda yoki buzilgan bo'lganda ko'tariladi."""


@dataclass
class PhosphorusEstimate:
    """Fosforni baholash natijasi."""

    olsen_p_mg_kg: float
    uncertainty_mg_kg: float
    ci95_low: float
    ci95_high: float
    confidence: str  # "high" | "medium" | "low"
    confidence_label_uz: str
    status_key: str
    status_label_uz: str
    status_color: str
    status_description_uz: str
    within_calibration: bool
    out_of_range_fields: list[str] = field(default_factory=list)
    tree_std: float = 0.0
    model_residual_std: float = 0.0
    notes_uz: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model yuklash
# ---------------------------------------------------------------------------


def load_artifact(path: Path | None = None) -> dict[str, Any]:
    """Saqlangan model artefaktini yuklaydi."""

    path = path or MODEL_PATH
    if not path.exists():
        raise ModelNotAvailableError(
            "Model fayli topilmadi (models/phosphorus_model.joblib). "
            "Iltimos, avval `python scripts/train_model.py` buyrug'ini bajaring."
        )
    try:
        artifact = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - joblib har xil xato turlarini beradi
        raise ModelNotAvailableError(
            "Model faylini o'qib bo'lmadi. Uni `python scripts/train_model.py` orqali qayta yarating."
        ) from exc

    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ModelNotAvailableError(
            "Model fayli formati noto'g'ri. `python scripts/train_model.py` orqali qayta yarating."
        )
    return artifact


# ---------------------------------------------------------------------------
# Fosfor sinfi
# ---------------------------------------------------------------------------


def classify_phosphorus(value: float, thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Olsen-P qiymatini konfiguratsiyadagi sinflarga ajratadi."""

    thresholds = thresholds or load_thresholds()
    classes = thresholds["classes"]
    for item in classes:
        if float(item["min"]) <= value < float(item["max"]):
            return item
    # Chegaradan yuqori qiymatlar eng yuqori sinfga tegishli.
    if value >= float(classes[-1]["max"]):
        return classes[-1]
    return classes[0]


# ---------------------------------------------------------------------------
# Noaniqlik va ishonch
# ---------------------------------------------------------------------------


def _tree_prediction_std(pipeline: Any, features: pd.DataFrame) -> float:
    """Random Forest daraxtlari bashoratlarining standart chetlanishi."""

    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    estimators = getattr(model, "estimators_", None)
    if not estimators:
        return 0.0

    transformed = features
    if hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            if name == "model":
                break
            transformed = step.transform(transformed)

    predictions = np.stack([tree.predict(np.asarray(transformed)) for tree in estimators])
    return float(np.std(predictions, axis=0)[0])


def _residual_sigma(prediction: float, report: Any) -> float:
    """Bashorat kattaligiga bog'liq qoldiq standart chetlanishi.

    O'qitish bosqichida moslangan `sigma = intercept + slope * bashorat` chizig'idan
    foydalanadi. Eski formatdagi artefaktlar uchun oddiy `residual_std` ga qaytadi.
    """

    intercept = float(getattr(report, "residual_intercept", 0.0) or 0.0)
    slope = float(getattr(report, "residual_slope", 0.0) or 0.0)
    fallback = float(getattr(report, "residual_std", 0.0) or 0.0)
    if intercept <= 0 and slope <= 0:
        return fallback
    return float(max(intercept + slope * max(prediction, 0.0), 0.1))


def assess_confidence(
    prediction: float,
    uncertainty: float,
    within_calibration: bool,
    target_range: dict[str, float] | None = None,
) -> str:
    """Konservativ ishonch qoidalari.

    - Kalibrlash oralig'idan tashqarida bo'lsa — har doim "past".
    - Nisbiy noaniqlik 15% dan kam bo'lsa — "yuqori", 30% dan kam bo'lsa — "o'rtacha".
    - Bashorat o'quv to'plamining maqsad oralig'idan chiqsa — daraja pasaytiriladi.
    """

    if not within_calibration:
        return "low"

    relative = uncertainty / max(abs(prediction), 1.0)
    if relative < 0.15:
        level = "high"
    elif relative < 0.30:
        level = "medium"
    else:
        level = "low"

    if target_range:
        low = float(target_range.get("min", 0.0))
        high = float(target_range.get("max", 0.0))
        if high > low and not (low <= prediction <= high):
            level = "medium" if level == "high" else "low"

    return level


def estimate_phosphorus(
    red: float,
    green: float,
    blue: float,
    reaction_time_sec: float,
    sample_temperature_c: float,
    artifact: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> PhosphorusEstimate:
    """Kolorimetrik o'lchovdan Olsen-P ni baholaydi."""

    artifact = artifact or load_artifact()
    pipeline = artifact["pipeline"]
    report = artifact.get("report")

    raw = {
        "red": float(red),
        "green": float(green),
        "blue": float(blue),
        "reaction_time_sec": float(reaction_time_sec),
        "sample_temperature_c": float(sample_temperature_c),
    }
    features = build_color_features(pd.DataFrame([raw]))
    features = features[list(artifact.get("feature_columns", FEATURE_COLUMNS))]

    prediction = float(pipeline.predict(features)[0])
    prediction = float(max(prediction, 0.0))

    # Noaniqlikning ikki manbasi birlashtiriladi:
    #  1. Model kelishmovchiligi — Random Forest daraxtlari tarqoqligi (agar mavjud bo'lsa);
    #  2. Qoldiq tarqoqlik — bashorat kattaligiga bog'liq (geteroskedastik) baho.
    tree_std = _tree_prediction_std(pipeline, features)
    residual_sigma = _residual_sigma(prediction, report)
    uncertainty = float(np.sqrt(tree_std**2 + residual_sigma**2))
    if uncertainty <= 0:
        uncertainty = max(0.1, 0.15 * prediction)

    calibration = getattr(report, "calibration", {}) or {}
    out_of_range = check_within_calibration(raw, calibration)
    within_calibration = not out_of_range

    target_range = getattr(report, "target_range", None)
    confidence = assess_confidence(prediction, uncertainty, within_calibration, target_range)

    status = classify_phosphorus(prediction, thresholds)

    notes: list[str] = []
    if out_of_range:
        notes.append(OUT_OF_RANGE_MESSAGE_UZ)
        notes.append("Oraliqdan chiqqan parametrlar: " + ", ".join(out_of_range) + ".")
    if getattr(report, "dataset_kind", "demo") == "demo":
        notes.append(
            "Model sintetik demo ma'lumotlar asosida o'qitilgan — real kalibrlash talab qilinadi."
        )

    return PhosphorusEstimate(
        olsen_p_mg_kg=round(prediction, 2),
        uncertainty_mg_kg=round(uncertainty, 2),
        ci95_low=round(max(prediction - 1.96 * uncertainty, 0.0), 2),
        ci95_high=round(prediction + 1.96 * uncertainty, 2),
        confidence=confidence,
        confidence_label_uz=CONFIDENCE_LABELS_UZ[confidence],
        status_key=str(status["key"]),
        status_label_uz=str(status["label_uz"]),
        status_color=str(status.get("color", "#2E7D32")),
        status_description_uz=str(status.get("description_uz", "")),
        within_calibration=within_calibration,
        out_of_range_fields=out_of_range,
        tree_std=round(tree_std, 3),
        model_residual_std=round(residual_sigma, 3),
        notes_uz=notes,
    )


def model_summary(artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Model va validatsiya sahifasi uchun qisqacha ma'lumot."""

    artifact = artifact or load_artifact()
    report = artifact.get("report")
    if report is None:
        raise ConfigError("Model artefaktida hisobot ma'lumoti yo'q.")
    return {
        "model_name": report.model_name,
        "metrics": report.metrics,
        "candidates": report.candidates,
        "dataset_kind": report.dataset_kind,
        "n_samples": report.n_samples,
        "n_train": report.n_train,
        "n_test": report.n_test,
        "split_strategy": report.split_strategy,
        "feature_columns": report.feature_columns,
        "calibration": report.calibration,
        "target_range": report.target_range,
        "residual_std": report.residual_std,
        "residual_intercept": getattr(report, "residual_intercept", 0.0),
        "residual_slope": getattr(report, "residual_slope", 0.0),
        "test_actual": getattr(report, "test_actual", []),
        "test_predicted": getattr(report, "test_predicted", []),
        "warnings": report.warnings,
        "trained_at": report.trained_at,
        "cv_metrics": report.cv_metrics,
    }
