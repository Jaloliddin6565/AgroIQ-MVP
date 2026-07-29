"""Tavsiya dvigateli, ogohlantirishlar, tushuntirishlar va PDF hisobot testlari."""

from __future__ import annotations

import pytest

from src.data_validation import ConfigError, get_product
from src.explanations import build_explanations, technical_details
from src.recommendation_engine import (
    Recommendation,
    build_recommendation,
    build_soil_warnings,
    fertilizer_from_p2o5,
    ph_adjustment,
    recommendation_confidence,
    status_multiplier,
)
from src.report_generator import ReportError, build_pdf_report, make_analysis_id


def make_recommendation(**overrides) -> Recommendation:
    kwargs = {
        "olsen_p_mg_kg": 7.0,
        "status_key": "low",
        "status_label_uz": "Past",
        "ph": 7.2,
        "ec_ds_m": 1.5,
        "moisture_pct": 22.0,
        "crop_key": "paxta",
        "field_area_ha": 10.0,
        "target_yield_t_ha": 3.5,
        "fertilizer_key": "ammofos",
        "model_confidence": "high",
        "within_calibration": True,
    }
    kwargs.update(overrides)
    return build_recommendation(**kwargs)


# ---------------------------------------------------------------------------
# O'g'it konversiyasi
# ---------------------------------------------------------------------------


def test_fertilizer_conversion_is_transparent():
    assert fertilizer_from_p2o5(46.0, 0.46) == pytest.approx(100.0)
    assert fertilizer_from_p2o5(19.0, 0.19) == pytest.approx(100.0)
    assert fertilizer_from_p2o5(0.0, 0.46) == 0.0


def test_lower_p2o5_fraction_needs_more_product():
    ammofos = fertilizer_from_p2o5(80.0, 0.46)
    superfosfat = fertilizer_from_p2o5(80.0, 0.19)
    assert superfosfat > ammofos


@pytest.mark.parametrize("bad_fraction", [0.0, -0.2, 1.5])
def test_invalid_p2o5_fraction_rejected(bad_fraction):
    with pytest.raises(ValueError):
        fertilizer_from_p2o5(50.0, bad_fraction)


def test_negative_requirement_rejected():
    with pytest.raises(ValueError):
        fertilizer_from_p2o5(-10.0, 0.46)


def test_conversion_matches_engine_output():
    """Dvigatel chiqishi oshkora formula bilan mos kelishi kerak."""

    recommendation = make_recommendation()
    fraction = get_product("ammofos")["p2o5_fraction"]
    assert recommendation.p2o5_fraction == fraction
    assert recommendation.product_kg_ha_low == pytest.approx(
        recommendation.required_p2o5_low / fraction, rel=1e-3
    )
    assert recommendation.product_kg_ha_high == pytest.approx(
        recommendation.required_p2o5_high / fraction, rel=1e-3
    )


def test_total_scales_with_field_area():
    small = make_recommendation(field_area_ha=5.0)
    large = make_recommendation(field_area_ha=20.0)
    assert large.total_product_kg_high == pytest.approx(small.total_product_kg_high * 4.0, rel=1e-6)


def test_higher_target_yield_increases_requirement():
    low = make_recommendation(target_yield_t_ha=2.0)
    high = make_recommendation(target_yield_t_ha=5.0)
    assert high.required_p2o5_high > low.required_p2o5_high


def test_lower_phosphorus_status_increases_requirement():
    sufficient = make_recommendation(status_key="sufficient", status_label_uz="Yetarli")
    very_low = make_recommendation(status_key="very_low", status_label_uz="Juda past")
    assert very_low.required_p2o5_high > sufficient.required_p2o5_high


def test_calculation_steps_are_exposed():
    recommendation = make_recommendation()
    assert len(recommendation.calculation_steps) == 6
    for step in recommendation.calculation_steps:
        assert {"step", "formula", "result"} <= set(step)


def test_invalid_crop_or_product_raises_config_error():
    with pytest.raises(ConfigError):
        make_recommendation(crop_key="mavjud_emas")
    with pytest.raises(ConfigError):
        make_recommendation(fertilizer_key="mavjud_emas")


@pytest.mark.parametrize("field", ["field_area_ha", "target_yield_t_ha"])
def test_non_positive_inputs_rejected(field):
    with pytest.raises(ValueError):
        make_recommendation(**{field: 0.0})


def test_status_multiplier_missing_key_raises():
    with pytest.raises(ConfigError):
        status_multiplier("mavjud_emas")


# ---------------------------------------------------------------------------
# Yuqori fosfor ssenariysi
# ---------------------------------------------------------------------------


def test_high_phosphorus_recommends_no_additional_fertilizer():
    recommendation = make_recommendation(
        olsen_p_mg_kg=40.0, status_key="high", status_label_uz="Yuqori"
    )
    assert recommendation.no_phosphorus_needed is True
    assert recommendation.required_p2o5_high == 0.0
    assert recommendation.total_product_kg_high == 0.0
    assert "tavsiya etilmaydi" in recommendation.application_stage_uz


def test_high_phosphorus_produces_informational_warning():
    recommendation = make_recommendation(
        olsen_p_mg_kg=40.0, status_key="high", status_label_uz="Yuqori"
    )
    keys = {warning.key for warning in recommendation.warnings}
    assert "phosphorus_high" in keys
    message = next(w.message_uz for w in recommendation.warnings if w.key == "phosphorus_high")
    assert "iqtisodiy" in message


# ---------------------------------------------------------------------------
# Tuproq sharoiti ogohlantirishlari
# ---------------------------------------------------------------------------


def test_high_ph_warning_is_raised():
    recommendation = make_recommendation(ph=8.5)
    keys = {warning.key for warning in recommendation.warnings}
    assert "ph_very_high" in keys


def test_moderately_high_ph_warning_is_raised():
    recommendation = make_recommendation(ph=8.0)
    keys = {warning.key for warning in recommendation.warnings}
    assert "ph_high" in keys


def test_high_ph_increases_required_dose():
    neutral = make_recommendation(ph=7.0)
    alkaline = make_recommendation(ph=8.6)
    assert alkaline.required_p2o5_high > neutral.required_p2o5_high


def test_acidic_ph_warning_is_raised():
    recommendation = make_recommendation(ph=5.4)
    keys = {warning.key for warning in recommendation.warnings}
    assert "ph_low" in keys


def test_high_ec_warning_is_raised():
    recommendation = make_recommendation(ec_ds_m=9.0)
    keys = {warning.key for warning in recommendation.warnings}
    assert "ec_high" in keys
    warning = next(w for w in recommendation.warnings if w.key == "ec_high")
    assert warning.level == "attention"
    assert "sho'rlanish" in warning.message_uz


def test_moderate_ec_warning_is_raised():
    recommendation = make_recommendation(ec_ds_m=5.0)
    keys = {warning.key for warning in recommendation.warnings}
    assert "ec_moderate" in keys


def test_normal_conditions_produce_no_soil_warnings():
    recommendation = make_recommendation(ph=7.0, ec_ds_m=1.2, moisture_pct=25.0)
    keys = {warning.key for warning in recommendation.warnings}
    assert keys == set()


def test_very_low_moisture_warning_is_raised():
    recommendation = make_recommendation(moisture_pct=8.0)
    keys = {warning.key for warning in recommendation.warnings}
    assert "moisture_very_low" in keys


def test_crop_specific_ec_threshold(crop_profiles):
    """Bug'doy paxtaga qaraganda sho'rga sezgirroq — chegara pastroqda ishlaydi."""

    wheat = build_soil_warnings(7.0, 6.5, 25.0, "low", crop_profiles["crops"][1], crop_profiles)
    cotton = build_soil_warnings(7.0, 6.5, 25.0, "low", crop_profiles["crops"][0], crop_profiles)
    assert "ec_crop_threshold" in {warning.key for warning in wheat}
    assert "ec_crop_threshold" not in {warning.key for warning in cotton}


def test_warnings_use_neutral_agronomic_wording():
    recommendation = make_recommendation(ph=8.6, ec_ds_m=9.5, moisture_pct=6.0)
    assert recommendation.warnings
    for warning in recommendation.warnings:
        text = warning.message_uz.lower()
        assert warning.level in {"info", "caution", "attention"}
        for forbidden in ("xavfli", "halokat", "zaharli", "o'ladi"):
            assert forbidden not in text


def test_ph_adjustment_selects_correct_band():
    assert ph_adjustment(7.0)["factor_high"] == 1.0
    assert ph_adjustment(8.6)["factor_high"] > 1.0
    assert ph_adjustment(5.0)["factor_high"] > 1.0


# ---------------------------------------------------------------------------
# Ishonchlilik
# ---------------------------------------------------------------------------


def test_recommendation_confidence_downgraded_by_attention_warnings():
    recommendation = make_recommendation(ph=8.6, ec_ds_m=9.5, model_confidence="high")
    assert recommendation.confidence == "low"


def test_recommendation_confidence_low_outside_calibration():
    assert recommendation_confidence("high", within_calibration=False, warnings=[]) == "low"


def test_recommendation_confidence_preserved_when_conditions_normal():
    recommendation = make_recommendation(ph=7.0, ec_ds_m=1.0, moisture_pct=25.0, model_confidence="high")
    assert recommendation.confidence == "high"


# ---------------------------------------------------------------------------
# Tushuntirishlar
# ---------------------------------------------------------------------------


def test_explanations_are_generated(artifact):
    from src.model_inference import estimate_phosphorus

    estimate = estimate_phosphorus(160, 200, 228, 600, 24, artifact=artifact)
    recommendation = make_recommendation(
        olsen_p_mg_kg=estimate.olsen_p_mg_kg,
        status_key=estimate.status_key,
        status_label_uz=estimate.status_label_uz,
        ph=8.1,
    )
    reasons = build_explanations(estimate, recommendation, 8.1, 1.8, 21.0)
    assert 3 <= len(reasons) <= 6
    assert any("pH" in reason for reason in reasons)
    assert any(estimate.status_label_uz in reason for reason in reasons)


def test_technical_details_include_metrics(artifact):
    from src.model_inference import estimate_phosphorus, model_summary

    estimate = estimate_phosphorus(120, 170, 215, 600, 24, artifact=artifact)
    recommendation = make_recommendation()
    details = technical_details(estimate, recommendation, model_summary(artifact))
    assert "R²" in details
    assert "Baholangan Olsen-P" in details


# ---------------------------------------------------------------------------
# PDF hisobot
# ---------------------------------------------------------------------------


def test_pdf_report_is_generated_in_memory(artifact):
    from src.model_inference import estimate_phosphorus, model_summary

    estimate = estimate_phosphorus(120, 170, 215, 600, 24, artifact=artifact)
    recommendation = make_recommendation(
        olsen_p_mg_kg=estimate.olsen_p_mg_kg,
        status_key=estimate.status_key,
        status_label_uz=estimate.status_label_uz,
    )
    measurements = {
        "red": 120, "green": 170, "blue": 215,
        "reaction_time_sec": 600, "sample_temperature_c": 24,
        "ph": 7.2, "ec_ds_m": 1.5, "moisture_pct": 22.0,
    }
    explanations = build_explanations(estimate, recommendation, 7.2, 1.5, 22.0)
    pdf = build_pdf_report(
        estimate, recommendation, measurements, explanations, model_summary(artifact)
    )
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2000


def test_pdf_report_error_is_wrapped(artifact):
    from src.model_inference import estimate_phosphorus

    estimate = estimate_phosphorus(120, 170, 215, 600, 24, artifact=artifact)
    with pytest.raises(ReportError):
        build_pdf_report(estimate, None, {}, [])  # type: ignore[arg-type]


def test_analysis_id_format():
    assert make_analysis_id().startswith("AGQ-")
