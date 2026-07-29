"""Kirish validatsiyasi va konfiguratsiya yuklash testlari."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data_validation import (
    ConfigError,
    DataError,
    MIN_VALID_TRAINING_ROWS,
    is_synthetic_frame,
    load_crop_profiles,
    load_thresholds,
    load_training_data,
    validate_analysis_request,
    validate_training_frame,
)


def _valid_payload(**overrides) -> dict:
    payload = {
        "colorimetry": {
            "red": 120.0,
            "green": 170.0,
            "blue": 215.0,
            "reaction_time_sec": 600.0,
            "sample_temperature_c": 24.0,
        },
        "soil": {
            "ph": 7.6,
            "ec_ds_m": 2.0,
            "moisture_pct": 20.0,
            "crop_key": "paxta",
            "field_area_ha": 10.0,
            "target_yield_t_ha": 3.5,
            "fertilizer_key": "ammofos",
        },
        "sample_label": "Test",
    }
    for section, values in overrides.items():
        payload[section].update(values)
    return payload


# ---------------------------------------------------------------------------
# Kirish validatsiyasi
# ---------------------------------------------------------------------------


def test_valid_payload_passes():
    request, errors = validate_analysis_request(_valid_payload())
    assert errors == []
    assert request is not None
    assert request.colorimetry.red == 120.0
    assert request.soil.crop_key == "paxta"


@pytest.mark.parametrize("channel", ["red", "green", "blue"])
@pytest.mark.parametrize("bad_value", [-1.0, 256.0])
def test_rgb_outside_0_255_rejected(channel, bad_value):
    request, errors = validate_analysis_request(_valid_payload(colorimetry={channel: bad_value}))
    assert request is None
    assert any("0 dan 255 gacha" in message for message in errors)


@pytest.mark.parametrize("bad_ph", [1.5, 12.0])
def test_ph_outside_soil_range_rejected(bad_ph):
    request, errors = validate_analysis_request(_valid_payload(soil={"ph": bad_ph}))
    assert request is None
    assert any("pH" in message for message in errors)


def test_negative_ec_rejected():
    request, errors = validate_analysis_request(_valid_payload(soil={"ec_ds_m": -0.5}))
    assert request is None
    assert any("manfiy" in message for message in errors)


@pytest.mark.parametrize("bad_moisture", [-5.0, 140.0])
def test_moisture_outside_0_100_rejected(bad_moisture):
    request, errors = validate_analysis_request(_valid_payload(soil={"moisture_pct": bad_moisture}))
    assert request is None
    assert any("0 dan 100" in message for message in errors)


@pytest.mark.parametrize("field", ["field_area_ha", "target_yield_t_ha"])
def test_non_positive_field_values_rejected(field):
    request, errors = validate_analysis_request(_valid_payload(soil={field: 0.0}))
    assert request is None
    assert any("musbat" in message for message in errors)


def test_missing_required_field_reports_error():
    payload = _valid_payload()
    del payload["soil"]["ph"]
    request, errors = validate_analysis_request(payload)
    assert request is None
    assert errors


def test_error_messages_are_user_friendly_not_tracebacks():
    request, errors = validate_analysis_request(_valid_payload(colorimetry={"red": 999.0}))
    assert request is None
    assert errors
    for message in errors:
        assert "Traceback" not in message
        assert "pydantic" not in message.lower()


# ---------------------------------------------------------------------------
# Konfiguratsiya
# ---------------------------------------------------------------------------


def test_thresholds_classes_are_ordered_and_continuous(thresholds):
    classes = thresholds["classes"]
    assert len(classes) == 5
    keys = [item["key"] for item in classes]
    assert keys == ["very_low", "low", "medium", "sufficient", "high"]
    labels = [item["label_uz"] for item in classes]
    assert labels == ["Juda past", "Past", "O'rtacha", "Yetarli", "Yuqori"]
    for previous, current in zip(classes, classes[1:], strict=False):
        assert previous["max"] == current["min"], "Sinf chegaralari uzluksiz bo'lishi kerak"


def test_crop_profiles_contain_mvp_crops(crop_profiles):
    keys = {crop["key"] for crop in crop_profiles["crops"]}
    assert {"paxta", "bugdoy"}.issubset(keys)


def test_fertilizer_products_contain_mvp_products(fertilizer_products):
    keys = {item["key"] for item in fertilizer_products["products"]}
    assert {"ammofos", "superfosfat"}.issubset(keys)
    for product in fertilizer_products["products"]:
        assert 0 < product["p2o5_fraction"] <= 1


def test_invalid_json_raises_config_error(tmp_path):
    (tmp_path / "phosphorus_thresholds.json").write_text("{ bu yaroqsiz json", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_thresholds(str(tmp_path))
    assert "JSON" in str(excinfo.value)


def test_missing_config_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_crop_profiles(str(tmp_path))
    assert "topilmadi" in str(excinfo.value)


def test_config_with_missing_section_raises_config_error(tmp_path):
    (tmp_path / "phosphorus_thresholds.json").write_text(
        json.dumps({"unit": "mg/kg"}), encoding="utf-8"
    )
    with pytest.raises(ConfigError) as excinfo:
        load_thresholds(str(tmp_path))
    assert "classes" in str(excinfo.value)


def test_config_with_inverted_class_bounds_raises(tmp_path):
    payload = {
        "classes": [
            {"key": "low", "label_uz": "Past", "min": 20.0, "max": 5.0},
        ]
    }
    (tmp_path / "phosphorus_thresholds.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(str(tmp_path))


# ---------------------------------------------------------------------------
# Dataset validatsiyasi
# ---------------------------------------------------------------------------


def _training_frame(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"S{index}" for index in range(rows)],
            "field_id": ["F001"] * rows,
            "red": [120.0] * rows,
            "green": [170.0] * rows,
            "blue": [215.0] * rows,
            "reaction_time_sec": [600.0] * rows,
            "sample_temperature_c": [24.0] * rows,
            "lab_olsen_p_mg_kg": [12.0] * rows,
        }
    )


def test_validate_training_frame_drops_incomplete_rows():
    frame = _training_frame()
    frame.loc[0, "red"] = None
    cleaned, warnings = validate_training_frame(frame)
    assert len(cleaned) == 4
    assert any("to'liq bo'lmagani" in warning for warning in warnings)


def test_validate_training_frame_drops_out_of_range_rows():
    frame = _training_frame()
    frame.loc[1, "blue"] = 900.0
    cleaned, warnings = validate_training_frame(frame)
    assert len(cleaned) == 4
    assert any("oraliqdan" in warning for warning in warnings)


def test_validate_training_frame_requires_columns():
    frame = _training_frame().drop(columns=["lab_olsen_p_mg_kg"])
    with pytest.raises(DataError) as excinfo:
        validate_training_frame(frame)
    assert "lab_olsen_p_mg_kg" in str(excinfo.value)


def test_missing_dataset_is_reported_as_insufficient(tmp_path):
    frame, warnings, sufficient = load_training_data(tmp_path / "yoq.csv")
    assert frame.empty
    assert sufficient is False
    assert warnings


def test_small_dataset_is_insufficient(tmp_path):
    path = tmp_path / "soil_samples.csv"
    _training_frame(rows=MIN_VALID_TRAINING_ROWS - 1).to_csv(path, index=False)
    _, warnings, sufficient = load_training_data(path)
    assert sufficient is False
    assert any(str(MIN_VALID_TRAINING_ROWS) in warning for warning in warnings)


def test_synthetic_marker_is_detected():
    frame = _training_frame()
    assert is_synthetic_frame(frame) is False
    frame["data_source"] = "SYNTHETIC_DEMO"
    assert is_synthetic_frame(frame) is True
