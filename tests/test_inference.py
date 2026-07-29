"""Xususiyat hosil qilish, model bashorati, noaniqlik va tasniflash testlari."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_color_features,
    calibration_ranges,
    check_within_calibration,
    features_from_reading,
)
from src.model_inference import (
    ModelNotAvailableError,
    assess_confidence,
    classify_phosphorus,
    estimate_phosphorus,
    load_artifact,
    model_summary,
)
from src.model_training import build_demo_samples, generate_demo_dataset, simulate_color


# ---------------------------------------------------------------------------
# Rang xususiyatlari
# ---------------------------------------------------------------------------


def test_feature_frame_has_expected_columns():
    features = features_from_reading(120, 170, 215, 600, 24)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert len(features) == 1
    assert features.notna().all(axis=None)


def test_normalized_channels_sum_to_one():
    features = features_from_reading(120, 170, 215, 600, 24)
    total = (
        features["red_norm"].iloc[0]
        + features["green_norm"].iloc[0]
        + features["blue_norm"].iloc[0]
    )
    assert total == pytest.approx(1.0, abs=1e-6)


def test_absorbance_increases_as_channel_darkens():
    bright = features_from_reading(200, 200, 220, 600, 24)["absorbance_red"].iloc[0]
    dark = features_from_reading(40, 120, 200, 600, 24)["absorbance_red"].iloc[0]
    assert dark > bright


def test_zero_channels_do_not_produce_nan_or_inf():
    features = features_from_reading(0, 0, 0, 10, 0)
    values = features.to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_missing_columns_raise_value_error():
    with pytest.raises(ValueError):
        build_color_features(pd.DataFrame([{"red": 10, "green": 20}]))


def test_higher_phosphorus_produces_lower_red_channel():
    """Fizik tekshiruv: fosfor ko'paysa eritma ko'karadi, qizil kanal pasayadi."""

    low_red, _, _ = simulate_color(5.0, 600.0, 24.0)
    high_red, _, _ = simulate_color(40.0, 600.0, 24.0)
    assert high_red < low_red


# ---------------------------------------------------------------------------
# Kalibrlash oralig'i
# ---------------------------------------------------------------------------


def test_calibration_range_detection():
    frame = generate_demo_dataset(n_fields=6)
    ranges = calibration_ranges(frame)
    assert set(ranges) >= {"red", "green", "blue", "reaction_time_sec", "sample_temperature_c"}

    inside = {
        "red": float(frame["red"].median()),
        "green": float(frame["green"].median()),
        "blue": float(frame["blue"].median()),
        "reaction_time_sec": 600.0,
        "sample_temperature_c": 24.0,
    }
    assert check_within_calibration(inside, ranges) == []

    outside = dict(inside, reaction_time_sec=1750.0)
    assert check_within_calibration(outside, ranges)


# ---------------------------------------------------------------------------
# Fosfor tasnifi
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_key", "expected_label"),
    [
        (1.0, "very_low", "Juda past"),
        (4.99, "very_low", "Juda past"),
        (5.0, "low", "Past"),
        (9.9, "low", "Past"),
        (10.0, "medium", "O'rtacha"),
        (17.9, "medium", "O'rtacha"),
        (18.0, "sufficient", "Yetarli"),
        (29.9, "sufficient", "Yetarli"),
        (30.0, "high", "Yuqori"),
        (250.0, "high", "Yuqori"),
    ],
)
def test_classification_boundaries(value, expected_key, expected_label, thresholds):
    result = classify_phosphorus(value, thresholds)
    assert result["key"] == expected_key
    assert result["label_uz"] == expected_label


def test_classification_handles_zero(thresholds):
    assert classify_phosphorus(0.0, thresholds)["key"] == "very_low"


# ---------------------------------------------------------------------------
# Ishonch darajasi
# ---------------------------------------------------------------------------


def test_confidence_high_for_tight_uncertainty():
    assert assess_confidence(20.0, 1.0, within_calibration=True) == "high"


def test_confidence_medium_for_moderate_uncertainty():
    assert assess_confidence(20.0, 4.0, within_calibration=True) == "medium"


def test_confidence_low_for_wide_uncertainty():
    assert assess_confidence(20.0, 12.0, within_calibration=True) == "low"


def test_confidence_always_low_outside_calibration():
    assert assess_confidence(20.0, 0.2, within_calibration=False) == "low"


def test_confidence_downgraded_outside_target_range():
    target_range = {"min": 2.0, "max": 50.0}
    inside = assess_confidence(20.0, 1.0, True, target_range)
    outside = assess_confidence(90.0, 1.0, True, target_range)
    assert inside == "high"
    assert outside == "medium"


# ---------------------------------------------------------------------------
# Model bashorati
# ---------------------------------------------------------------------------


def test_estimate_returns_complete_result(artifact, thresholds):
    estimate = estimate_phosphorus(120, 170, 215, 600, 24, artifact=artifact, thresholds=thresholds)
    assert estimate.olsen_p_mg_kg >= 0
    assert estimate.uncertainty_mg_kg > 0
    assert estimate.ci95_low <= estimate.olsen_p_mg_kg <= estimate.ci95_high
    assert estimate.confidence in {"low", "medium", "high"}
    assert estimate.status_label_uz in {"Juda past", "Past", "O'rtacha", "Yetarli", "Yuqori"}


def test_estimate_is_deterministic(artifact):
    first = estimate_phosphorus(110, 165, 210, 600, 24, artifact=artifact)
    second = estimate_phosphorus(110, 165, 210, 600, 24, artifact=artifact)
    assert first.olsen_p_mg_kg == second.olsen_p_mg_kg
    assert first.uncertainty_mg_kg == second.uncertainty_mg_kg


def test_estimate_never_negative(artifact):
    estimate = estimate_phosphorus(255, 255, 255, 600, 24, artifact=artifact)
    assert estimate.olsen_p_mg_kg >= 0.0


def test_darker_red_channel_gives_higher_phosphorus(artifact):
    """Model fizik kutilmani saqlashi kerak: qizil kanal pasaysa, fosfor ortadi."""

    low = estimate_phosphorus(*simulate_color(6.0, 600, 24), 600, 24, artifact=artifact)
    high = estimate_phosphorus(*simulate_color(35.0, 600, 24), 600, 24, artifact=artifact)
    assert high.olsen_p_mg_kg > low.olsen_p_mg_kg


def test_uncertainty_grows_with_estimate(artifact):
    """Geteroskedastik noaniqlik: yuqori konsentratsiyada absolyut xato kattaroq."""

    low = estimate_phosphorus(*simulate_color(6.0, 600, 24), 600, 24, artifact=artifact)
    high = estimate_phosphorus(*simulate_color(35.0, 600, 24), 600, 24, artifact=artifact)
    assert high.uncertainty_mg_kg > low.uncertainty_mg_kg


def test_out_of_calibration_sample_is_flagged(artifact):
    estimate = estimate_phosphorus(120, 170, 215, 1795, 58, artifact=artifact)
    assert estimate.within_calibration is False
    assert estimate.confidence == "low"
    assert estimate.out_of_range_fields
    assert any("kalibrlash" in note.lower() for note in estimate.notes_uz)


def test_demo_samples_reproduce_expected_levels(artifact, thresholds):
    """Uchta demo ssenariy aniq farqlanadigan fosfor sinflarini berishi kerak."""

    demo = build_demo_samples()
    statuses = []
    for _, row in demo.iterrows():
        estimate = estimate_phosphorus(
            row["red"], row["green"], row["blue"],
            row["reaction_time_sec"], row["sample_temperature_c"],
            artifact=artifact, thresholds=thresholds,
        )
        # Baho kutilgan qiymatdan 25% dan ko'p farq qilmasligi kerak.
        assert estimate.olsen_p_mg_kg == pytest.approx(row["expected_p_mg_kg"], rel=0.25)
        statuses.append(estimate.status_key)

    assert len(set(statuses)) == 3, "Demo ssenariylar turli natijalar berishi kerak"
    assert statuses[-1] == "high"


def test_missing_model_file_raises_friendly_error(tmp_path):
    with pytest.raises(ModelNotAvailableError) as excinfo:
        load_artifact(tmp_path / "yoq.joblib")
    message = str(excinfo.value)
    assert "train_model.py" in message
    assert "Traceback" not in message


def test_corrupted_model_file_raises_friendly_error(tmp_path):
    path = tmp_path / "buzilgan.joblib"
    path.write_bytes(b"bu joblib fayl emas")
    with pytest.raises(ModelNotAvailableError):
        load_artifact(path)


def test_model_summary_reports_required_fields(artifact):
    summary = model_summary(artifact)
    for key in ("model_name", "metrics", "dataset_kind", "n_samples", "split_strategy", "calibration"):
        assert key in summary
    assert set(summary["metrics"]) == {"r2", "rmse", "mae"}
    assert summary["dataset_kind"] in {"real", "demo"}
    assert len(summary["candidates"]) >= 3
    assert any("Random Forest" in item["name"] for item in summary["candidates"])


def test_demo_dataset_is_reproducible():
    first = generate_demo_dataset(n_fields=8)
    second = generate_demo_dataset(n_fields=8)
    pd.testing.assert_frame_equal(first, second)
    assert (first["data_source"] == "SYNTHETIC_DEMO").all()
