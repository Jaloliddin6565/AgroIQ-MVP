"""Ma'lumotlarni birlashtirish, talqin qilish va provenance testlari (v0.2.0).

Ushbu fayl AgroIQ ning eng muhim ilmiy qoidalarini himoya qiladi:
  - kolorimetrik Olsen-P HAR DOIM asosiy fosfor manbai bo'lib qoladi;
  - sensor P indikatori u bilan hech qachon o'rtachalanmaydi;
  - azot va kaliy uchun miqdoriy me'yor berilmaydi.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.data_fusion import (
    compare_phosphorus_sources,
    compute_confidence,
    fuse,
)
from src.demo_scenarios import SCENARIOS, all_scenarios, build_scenario, scenario_ids
from src.sensor_schemas import (
    AnalyzerReading,
    DataSource,
    QualityFlag,
    SensorReading,
    utc_now,
)
from src.soil_interpretation import (
    assess_nutrient,
    condition_chart_data,
    diagnostic_index,
    interpret_parameter,
    interpret_soil_conditions,
)


def _analyzer(**overrides) -> AnalyzerReading:
    payload = {
        "red": 120.0, "green": 170.0, "blue": 215.0,
        "reaction_time_sec": 600.0, "analyzer_temperature_c": 24.0,
        "analyzer_device_id": "AGROIQ-P-001",
    }
    payload.update(overrides)
    return AnalyzerReading(**payload)


def _sensor(**overrides) -> SensorReading:
    payload = {
        "nitrogen_indicator": 60.0, "phosphorus_indicator": 20.0,
        "potassium_indicator": 180.0, "ph": 7.2, "ec_ds_m": 1.5,
        "soil_temperature_c": 22.0, "soil_moisture_percent": 24.0,
        "sensor_device_id": "SOIL-001", "measured_at": utc_now(),
    }
    payload.update(overrides)
    return SensorReading(**payload)


# ---------------------------------------------------------------------------
# pH / EC / namlik talqini
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_band"),
    [(5.0, "strongly_acidic"), (6.2, "acidic"), (7.0, "optimal"),
     (8.0, "alkaline"), (8.8, "strongly_alkaline")],
)
def test_ph_interpretation_bands(value, expected_band):
    assert interpret_parameter("ph", value).band_key == expected_band


@pytest.mark.parametrize(
    ("value", "expected_band"),
    [(0.2, "very_low"), (1.5, "normal"), (5.0, "moderate"), (12.0, "high")],
)
def test_ec_interpretation_bands(value, expected_band):
    assert interpret_parameter("ec_ds_m", value).band_key == expected_band


@pytest.mark.parametrize(
    ("value", "expected_band"),
    [(8.0, "very_low"), (15.0, "low"), (25.0, "optimal"), (36.0, "high"), (60.0, "saturated")],
)
def test_moisture_interpretation_bands(value, expected_band):
    assert interpret_parameter("soil_moisture_percent", value).band_key == expected_band


def test_missing_value_is_unknown_not_an_error():
    result = interpret_parameter("ph", None)
    assert result.band_key == "unknown"
    assert result.level == "unknown"


def test_interpret_soil_conditions_returns_all_parameters():
    conditions = interpret_soil_conditions(7.0, 1.2, 25.0, 21.0)
    assert set(conditions) == {"ph", "ec_ds_m", "soil_moisture_percent", "soil_temperature_c"}


def test_condition_chart_skips_missing_values():
    conditions = interpret_soil_conditions(7.0, None, 25.0, None)
    rows = condition_chart_data(conditions)
    assert {row["parameter"] for row in rows} == {"ph", "soil_moisture_percent"}


# ---------------------------------------------------------------------------
# N va K — faqat sifatiy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nutrient", "value", "expected"),
    [("nitrogen", 20.0, "likely_low"), ("nitrogen", 60.0, "acceptable"),
     ("nitrogen", 200.0, "likely_high"), ("potassium", 80.0, "likely_low"),
     ("potassium", 180.0, "acceptable"), ("potassium", 900.0, "likely_high")],
)
def test_nutrient_qualitative_status(nutrient, value, expected):
    assert assess_nutrient(nutrient, value).status_key == expected


def test_nutrient_assessment_never_enables_quantitative_by_default():
    """ENG MUHIM: kalibrlashsiz N va K uchun miqdoriy me'yor berilmaydi."""

    for nutrient in ("nitrogen", "potassium"):
        assessment = assess_nutrient(nutrient, 55.0)
        assert assessment.quantitative_enabled is False
        assert assessment.requires_lab_confirmation is True
        assert "laboratoriya" in assessment.advice_uz.lower()


def test_nutrient_assessment_handles_missing_value():
    assessment = assess_nutrient("nitrogen", None)
    assert assessment.value is None
    assert assessment.status_key == "unknown"


def test_assess_nutrient_rejects_phosphorus():
    """Fosfor bu funksiya orqali baholanmaydi — u kolorimetrik modeldan keladi."""

    with pytest.raises(ValueError):
        assess_nutrient("phosphorus", 20.0)


# ---------------------------------------------------------------------------
# Fosfor manbalarini taqqoslash — o'rtachalash TAQIQLANGAN
# ---------------------------------------------------------------------------


def test_close_phosphorus_values_agree():
    comparison = compare_phosphorus_sources(20.0, 22.0)
    assert comparison.agrees is True


def test_divergent_phosphorus_values_disagree():
    comparison = compare_phosphorus_sources(7.0, 46.0)
    assert comparison.agrees is False
    assert "tafovut" in comparison.explanation_uz
    assert "asosiy qiymat sifatida ishlatildi" in comparison.explanation_uz


def test_comparison_never_averages_the_two_sources():
    """Taqqoslash natijasida o'rtacha qiymat HOSIL BO'LMASLIGI kerak."""

    colorimetric, sensor_value = 8.0, 40.0
    comparison = compare_phosphorus_sources(colorimetric, sensor_value)
    average = (colorimetric + sensor_value) / 2
    assert comparison.colorimetric_olsen_p == colorimetric
    assert comparison.sensor_indicator == sensor_value
    # Hech qanday maydonda o'rtacha qiymat saqlanmaydi.
    for value in vars(comparison).values():
        if isinstance(value, (int, float)):
            assert value != pytest.approx(average)


def test_small_absolute_difference_at_low_concentration_still_agrees():
    """Past konsentratsiyada kichik absolyut farq katta nisbiy farq bermasligi kerak."""

    comparison = compare_phosphorus_sources(3.0, 7.0)
    assert comparison.agrees is True


def test_comparison_without_sensor_value():
    comparison = compare_phosphorus_sources(15.0, None)
    assert comparison.agrees is None
    assert "kolorimetrik" in comparison.explanation_uz


def test_primary_source_is_always_colorimetric():
    comparison = compare_phosphorus_sources(5.0, 90.0)
    assert "kolorimetrik" in comparison.primary_source_uz.lower()


# ---------------------------------------------------------------------------
# Ishonchlilik hisobi
# ---------------------------------------------------------------------------


def test_confidence_reduced_by_calibration_flag():
    assert compute_confidence("high", [QualityFlag.COLOR_OUTSIDE_CALIBRATION_RANGE.value]) == "low"


def test_confidence_reduced_by_stale_data():
    assert compute_confidence("high", [QualityFlag.SENSOR_DATA_STALE.value]) == "medium"


def test_confidence_not_penalised_by_high_ph():
    """Yuqori pH hisob-kitobda allaqachon tuzatilgan — ishonchlilikni pasaytirmaydi."""

    assert compute_confidence("high", [QualityFlag.PH_HIGH.value]) == "high"


def test_confidence_never_below_low():
    flags = [flag.value for flag in QualityFlag]
    assert compute_confidence("low", flags) == "low"


# ---------------------------------------------------------------------------
# To'liq birlashtirish
# ---------------------------------------------------------------------------


def test_fusion_produces_complete_result(artifact):
    result = fuse(_analyzer(), _sensor(), artifact=artifact)
    assert result.estimate is not None
    assert result.confidence in {"low", "medium", "high"}
    assert result.nitrogen.quantitative_enabled is False
    assert result.potassium.quantitative_enabled is False
    assert result.diagnostic_index is not None


def test_fusion_flags_stale_sensor_data(artifact):
    stale = _sensor(measured_at=utc_now() - timedelta(hours=48))
    result = fuse(_analyzer(), stale, artifact=artifact)
    assert QualityFlag.SENSOR_DATA_STALE.value in result.quality_flags


def test_fusion_flags_missing_device_id(artifact):
    result = fuse(_analyzer(analyzer_device_id=None), _sensor(sensor_device_id=None),
                  artifact=artifact)
    assert QualityFlag.DEVICE_ID_MISSING.value in result.quality_flags


def test_fusion_flags_high_ph(artifact):
    result = fuse(_analyzer(), _sensor(ph=8.6), artifact=artifact)
    assert QualityFlag.PH_HIGH.value in result.quality_flags


def test_fusion_flags_high_ec(artifact):
    result = fuse(_analyzer(), _sensor(ec_ds_m=9.0), artifact=artifact)
    assert QualityFlag.EC_HIGH.value in result.quality_flags


def test_fusion_flags_low_moisture(artifact):
    result = fuse(_analyzer(), _sensor(soil_moisture_percent=9.0), artifact=artifact)
    assert QualityFlag.MOISTURE_LOW.value in result.quality_flags


def test_fusion_flags_nk_lab_confirmation(artifact):
    result = fuse(_analyzer(), _sensor(), artifact=artifact)
    assert QualityFlag.N_REQUIRES_LAB_CONFIRMATION.value in result.quality_flags
    assert QualityFlag.K_REQUIRES_LAB_CONFIRMATION.value in result.quality_flags


def test_fusion_flags_synthetic_model(artifact):
    result = fuse(_analyzer(), _sensor(), artifact=artifact)
    assert QualityFlag.SYNTHETIC_MODEL.value in result.quality_flags


def test_fusion_without_sensor_uses_manual_values(artifact):
    """Sensor bo'lmasa, qo'lda kiritilgan qiymatlar ishlatiladi."""

    result = fuse(_analyzer(), None, artifact=artifact,
                  manual_ph=7.9, manual_ec=3.0, manual_moisture=20.0)
    assert result.ph == 7.9
    assert result.ec_ds_m == 3.0
    assert QualityFlag.SENSOR_DATA_MISSING.value in result.quality_flags


def test_sensor_values_take_precedence_over_manual(artifact):
    result = fuse(_analyzer(), _sensor(ph=6.9), artifact=artifact, manual_ph=8.5)
    assert result.ph == 6.9


def test_fusion_records_data_provenance(artifact):
    result = fuse(
        _analyzer(source=DataSource.DEMO),
        _sensor(source=DataSource.API),
        artifact=artifact,
    )
    assert DataSource.DEMO.value in result.data_sources
    assert DataSource.API.value in result.data_sources
    assert QualityFlag.DEMO_DATA.value in result.quality_flags


def test_fusion_flag_details_are_translated(artifact):
    result = fuse(_analyzer(), _sensor(ph=8.7, ec_ds_m=9.5), artifact=artifact)
    details = result.flag_details()
    assert details
    for detail in details:
        assert detail["title"] and detail["message"]
        assert detail["level"] in {"info", "caution", "attention"}


def test_fusion_keeps_colorimetric_as_primary_on_disagreement(artifact):
    """Tafovut bo'lganda ham tavsiyaga kolorimetrik qiymat uzatiladi."""

    result = fuse(_analyzer(), _sensor(phosphorus_indicator=95.0), artifact=artifact)
    assert QualityFlag.P_INDICATORS_DISAGREE.value in result.quality_flags
    assert result.estimate.olsen_p_mg_kg == result.phosphorus_comparison.colorimetric_olsen_p
    # Sensor qiymati bahoni o'zgartirmagan.
    assert result.estimate.olsen_p_mg_kg != pytest.approx(95.0)


# ---------------------------------------------------------------------------
# Diagnostika indeksi
# ---------------------------------------------------------------------------


def test_diagnostic_index_has_disclaimer():
    conditions = interpret_soil_conditions(7.0, 1.2, 25.0, 22.0)
    index = diagnostic_index("sufficient", conditions)
    assert index is not None
    assert 0 <= index["score"] <= 100
    assert "sertifikatlangan" in index["disclaimer_uz"].lower()


def test_diagnostic_index_none_without_any_component():
    empty = interpret_soil_conditions(None, None, None, None)
    assert diagnostic_index(None, empty) is None


def test_diagnostic_index_better_for_good_conditions():
    good = interpret_soil_conditions(7.0, 1.2, 25.0, 22.0)
    bad = interpret_soil_conditions(9.0, 12.0, 6.0, 40.0)
    assert diagnostic_index("sufficient", good)["score"] > diagnostic_index("very_low", bad)["score"]


# ---------------------------------------------------------------------------
# Demo ssenariylar
# ---------------------------------------------------------------------------


def test_four_demo_scenarios_exist():
    assert len(SCENARIOS) == 4
    assert "demo_disagreement" in scenario_ids()


def test_every_scenario_has_both_devices_and_field_data():
    for scenario in all_scenarios():
        assert scenario["analyzer"]["analyzer_device_id"]
        assert scenario["sensor"]["sensor_device_id"]
        assert scenario["sensor"]["measured_at"] is not None
        assert scenario["field"]["crop_key"]
        for channel in ("red", "green", "blue"):
            assert 0 <= scenario["analyzer"][channel] <= 255


def test_scenarios_produce_distinct_phosphorus_classes(artifact):
    statuses = []
    for scenario in all_scenarios():
        result = fuse(
            AnalyzerReading(**scenario["analyzer"], source=DataSource.DEMO),
            SensorReading(**scenario["sensor"], source=DataSource.DEMO),
            artifact=artifact,
        )
        statuses.append(result.estimate.status_key)
    assert len(set(statuses)) >= 3


def test_only_fourth_scenario_triggers_disagreement(artifact):
    """Tafovut bayrog'i FAQAT 4-ssenariyda paydo bo'lishi kerak."""

    disagreements = []
    for scenario in all_scenarios():
        result = fuse(
            AnalyzerReading(**scenario["analyzer"], source=DataSource.DEMO),
            SensorReading(**scenario["sensor"], source=DataSource.DEMO),
            artifact=artifact,
        )
        if QualityFlag.P_INDICATORS_DISAGREE.value in result.quality_flags:
            disagreements.append(scenario["demo_id"])
    assert disagreements == ["demo_disagreement"]


def test_high_phosphorus_scenario_classified_high(artifact):
    scenario = build_scenario("demo_high")
    result = fuse(
        AnalyzerReading(**scenario["analyzer"], source=DataSource.DEMO),
        SensorReading(**scenario["sensor"], source=DataSource.DEMO),
        artifact=artifact,
    )
    assert result.estimate.status_key == "high"


def test_scenarios_are_not_stale(artifact):
    """Demo ssenariylar har doim yangi vaqt belgisiga ega bo'lishi kerak."""

    for scenario in all_scenarios():
        result = fuse(
            AnalyzerReading(**scenario["analyzer"], source=DataSource.DEMO),
            SensorReading(**scenario["sensor"], source=DataSource.DEMO),
            artifact=artifact,
        )
        assert QualityFlag.SENSOR_DATA_STALE.value not in result.quality_flags


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        build_scenario("mavjud_emas")
