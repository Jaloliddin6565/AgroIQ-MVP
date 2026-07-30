"""Tushuntiriladigan ma'lumotlarni birlashtirish (data fusion) qatlami — v0.2.0.

Bu qatlam ikkita mustaqil ma'lumot manbasini birlashtiradi, LEKIN ularni
ARALASHTIRMAYDI:

  AgroIQ kolorimetrik analizatori ──► Olsen-P bahosi  (ASOSIY fosfor manbai)
  Universal tuproq sensori        ──► pH, EC, namlik, harorat (kontekst)
                                  └─► N, P, K indikatorlari (skrining)

ENG MUHIM ILMIY QOIDA
---------------------
Sensorning fosfor indikatori Olsen-P bahosining o'rnini HECH QACHON bosmaydi va
u bilan O'RTACHALANMAYDI. Ikki qiymat farq qilsa, faqat MOSLIK TEKSHIRUVI
sifatida bayroq qo'yiladi va kolorimetrik natija asosiy bo'lib qoladi.

Bosqichlar:
  1. Vaqt belgilari va o'lchov oraliqlarini tekshirish
  2. Yetishmayotgan yoki ziddiyatli kirishlarni aniqlash
  3. Kolorimetrik ma'lumotdan Olsen-P baholash
  4. pH, EC, namlik va haroratni sinflash
  5. N va K indikatorlarini konservativ talqin qilish
  6. Sensor P va Olsen-P moslik tekshiruvi
  7. Sifat bayroqlarini shakllantirish
  8. Umumiy tavsiya ishonchliligini hisoblash
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.data_validation import (
    load_recommendation_rules,
    load_sensor_thresholds,
    load_thresholds,
)
from src.model_inference import PhosphorusEstimate, estimate_phosphorus
from src.sensor_schemas import (
    AnalyzerReading,
    DataSource,
    QUALITY_FLAG_LABELS_UZ,
    QualityFlag,
    SensorReading,
    is_in_future,
    is_stale,
    utc_now,
)
from src.soil_interpretation import (
    NutrientAssessment,
    ParameterInterpretation,
    assess_nutrient,
    diagnostic_index,
    interpret_soil_conditions,
)

CONFIDENCE_ORDER = ["low", "medium", "high"]
CONFIDENCE_LABELS_UZ = {"high": "Yuqori", "medium": "O'rtacha", "low": "Past"}


@dataclass
class PhosphorusComparison:
    """Kolorimetrik Olsen-P va sensor P indikatorining moslik tekshiruvi."""

    colorimetric_olsen_p: float | None
    sensor_indicator: float | None
    difference: float | None
    relative_difference: float | None
    agrees: bool | None
    primary_source_uz: str = "AgroIQ kolorimetrik Olsen-P bahosi"
    explanation_uz: str = ""


@dataclass
class FusionResult:
    """Birlashtirish natijasi — tavsiya dvigateliga uzatiladi."""

    estimate: PhosphorusEstimate | None
    conditions: dict[str, ParameterInterpretation]
    nitrogen: NutrientAssessment
    potassium: NutrientAssessment
    phosphorus_comparison: PhosphorusComparison
    quality_flags: list[str]
    confidence: str
    confidence_label_uz: str
    data_sources: list[str]
    diagnostic_index: dict[str, Any] | None = None
    notes_uz: list[str] = field(default_factory=list)
    # Tavsiya dvigateli uchun yakuniy, tekshirilgan qiymatlar
    ph: float | None = None
    ec_ds_m: float | None = None
    soil_moisture_percent: float | None = None
    soil_temperature_c: float | None = None

    def flag_details(self) -> list[dict[str, str]]:
        """Bayroqlarni foydalanuvchiga ko'rsatish uchun matnga aylantiradi."""

        details: list[dict[str, str]] = []
        for flag in self.quality_flags:
            meta = QUALITY_FLAG_LABELS_UZ.get(flag)
            if meta:
                details.append({"code": flag, **meta})
        return details


# ---------------------------------------------------------------------------
# 6-bosqich: fosfor moslik tekshiruvi
# ---------------------------------------------------------------------------


def compare_phosphorus_sources(
    colorimetric_olsen_p: float | None,
    sensor_indicator: float | None,
    thresholds: dict[str, Any] | None = None,
) -> PhosphorusComparison:
    """Ikki fosfor manbasini TAQQOSLAYDI, lekin hech qachon birlashtirmaydi.

    Bu faqat moslik tekshiruvi. Natija tavsiyada ishlatiladigan qiymatni
    o'zgartirmaydi — kolorimetrik baho har doim asosiy bo'lib qoladi.
    """

    thresholds = thresholds or load_sensor_thresholds()
    spec = thresholds.get("phosphorus_agreement", {})
    relative_tolerance = float(spec.get("relative_tolerance", 0.60))
    absolute_tolerance = float(spec.get("absolute_tolerance_mg_kg", 6.0))

    if colorimetric_olsen_p is None or sensor_indicator is None:
        return PhosphorusComparison(
            colorimetric_olsen_p=colorimetric_olsen_p,
            sensor_indicator=sensor_indicator,
            difference=None,
            relative_difference=None,
            agrees=None,
            explanation_uz=(
                "Taqqoslash uchun ikkala fosfor qiymati ham kerak. Tavsiyada AgroIQ "
                "kolorimetrik bahosi ishlatiladi."
            ),
        )

    difference = float(sensor_indicator) - float(colorimetric_olsen_p)
    base = max(abs(float(colorimetric_olsen_p)), 1.0)
    relative = abs(difference) / base

    # Ikkala shart ham buzilgandagina tafovut deb hisoblanadi — kichik absolyut
    # farqlar past konsentratsiyalarda katta nisbiy farq berishi mumkin.
    agrees = not (relative > relative_tolerance and abs(difference) > absolute_tolerance)

    if agrees:
        explanation = (
            "Sensor fosfor indikatori kolorimetrik baho bilan umumiy mos keladi. "
            "Bu qo'shimcha ishonch beradi, ammo tavsiya baribir kolorimetrik natijaga asoslanadi."
        )
    else:
        explanation = (
            "Fosfor bo'yicha qurilmalar o'rtasida tafovut aniqlandi. AgroIQ kolorimetrik "
            "natijasi tavsiyada asosiy qiymat sifatida ishlatildi. Laboratoriya tasdig'i "
            "tavsiya etiladi."
        )

    return PhosphorusComparison(
        colorimetric_olsen_p=float(colorimetric_olsen_p),
        sensor_indicator=float(sensor_indicator),
        difference=round(difference, 2),
        relative_difference=round(relative, 3),
        agrees=agrees,
        explanation_uz=explanation,
    )


# ---------------------------------------------------------------------------
# 8-bosqich: umumiy ishonchlilik
# ---------------------------------------------------------------------------


def compute_confidence(
    model_confidence: str,
    quality_flags: list[str],
    rules: dict[str, Any] | None = None,
) -> str:
    """Model ishonchidan boshlanadi va bayroqlar bo'yicha KONSERVATIV pasaytiriladi."""

    rules = rules or load_recommendation_rules()
    penalties = rules.get("confidence_penalties", {})

    level_index = (
        CONFIDENCE_ORDER.index(model_confidence)
        if model_confidence in CONFIDENCE_ORDER
        else 0
    )
    total_penalty = sum(int(penalties.get(flag, 0)) for flag in quality_flags)
    return CONFIDENCE_ORDER[max(level_index - total_penalty, 0)]


# ---------------------------------------------------------------------------
# Asosiy birlashtirish funksiyasi
# ---------------------------------------------------------------------------


def fuse(
    analyzer: AnalyzerReading | None,
    sensor: SensorReading | None,
    artifact: dict[str, Any] | None = None,
    manual_ph: float | None = None,
    manual_ec: float | None = None,
    manual_moisture: float | None = None,
    reference_time: datetime | None = None,
    sensor_thresholds: dict[str, Any] | None = None,
    phosphorus_thresholds: dict[str, Any] | None = None,
    rules: dict[str, Any] | None = None,
    extra_flags: list[str] | None = None,
) -> FusionResult:
    """Barcha manbalarni tekshiradi, talqin qiladi va birlashtiradi.

    `manual_*` qiymatlari sensor o'lchovi bo'lmaganda ishlatiladi (zaxira manba).
    Sensor qiymati mavjud bo'lsa, u ustunlikka ega.
    """

    sensor_thresholds = sensor_thresholds or load_sensor_thresholds()
    phosphorus_thresholds = phosphorus_thresholds or load_thresholds()
    rules = rules or load_recommendation_rules()
    reference_time = reference_time or utc_now()

    flags: list[str] = list(extra_flags or [])
    notes: list[str] = []
    sources: list[str] = []

    # --- 3-bosqich: kolorimetrik Olsen-P bahosi (ASOSIY fosfor manbai) ---
    estimate: PhosphorusEstimate | None = None
    if analyzer is not None:
        sources.append(analyzer.source.value)
        estimate = estimate_phosphorus(
            red=analyzer.red,
            green=analyzer.green,
            blue=analyzer.blue,
            reaction_time_sec=analyzer.reaction_time_sec,
            sample_temperature_c=analyzer.analyzer_temperature_c,
            artifact=artifact,
            thresholds=phosphorus_thresholds,
        )
        if not estimate.within_calibration:
            flags.append(QualityFlag.COLOR_OUTSIDE_CALIBRATION_RANGE.value)
        if not analyzer.analyzer_device_id:
            flags.append(QualityFlag.DEVICE_ID_MISSING.value)

    # --- 1-bosqich: sensor vaqt belgilarini tekshirish ---
    freshness = sensor_thresholds.get("data_freshness", {})
    stale_hours = float(freshness.get("stale_after_hours", 24))
    future_tolerance = float(freshness.get("future_tolerance_minutes", 10))

    if sensor is not None:
        if sensor.source.value not in sources:
            sources.append(sensor.source.value)
        if is_stale(sensor.measured_at, stale_hours, reference_time):
            flags.append(QualityFlag.SENSOR_DATA_STALE.value)
        if is_in_future(sensor.measured_at, future_tolerance, reference_time):
            flags.append(QualityFlag.TIMESTAMP_IN_FUTURE.value)
        if not sensor.sensor_device_id:
            if QualityFlag.DEVICE_ID_MISSING.value not in flags:
                flags.append(QualityFlag.DEVICE_ID_MISSING.value)
        flags.extend(
            code for code in sensor.quality_flags if code not in flags
        )
    else:
        flags.append(QualityFlag.SENSOR_DATA_MISSING.value)

    # --- 2-bosqich: tuproq sharoiti qiymatlarini tanlash (sensor ustun) ---
    ph = sensor.ph if sensor and sensor.ph is not None else manual_ph
    ec = sensor.ec_ds_m if sensor and sensor.ec_ds_m is not None else manual_ec
    moisture = (
        sensor.soil_moisture_percent
        if sensor and sensor.soil_moisture_percent is not None
        else manual_moisture
    )
    soil_temperature = sensor.soil_temperature_c if sensor else None

    # --- 4-bosqich: tuproq sharoitini sinflash ---
    conditions = interpret_soil_conditions(
        ph=ph,
        ec_ds_m=ec,
        soil_moisture_percent=moisture,
        soil_temperature_c=soil_temperature,
        thresholds=sensor_thresholds,
    )

    ph_band = conditions["ph"].band_key
    if ph_band in {"alkaline", "strongly_alkaline"}:
        flags.append(QualityFlag.PH_HIGH.value)
    elif ph_band in {"acidic", "strongly_acidic"}:
        flags.append(QualityFlag.PH_LOW.value)

    if conditions["ec_ds_m"].band_key in {"moderate", "high"}:
        flags.append(QualityFlag.EC_HIGH.value)

    moisture_band = conditions["soil_moisture_percent"].band_key
    if moisture_band in {"very_low", "low"}:
        flags.append(QualityFlag.MOISTURE_LOW.value)
    elif moisture_band == "saturated":
        flags.append(QualityFlag.MOISTURE_HIGH.value)

    if conditions["soil_temperature_c"].level == "attention":
        flags.append(QualityFlag.SOIL_TEMPERATURE_EXTREME.value)

    # --- 5-bosqich: N va K ni konservativ talqin qilish ---
    nitrogen = assess_nutrient(
        "nitrogen",
        sensor.nitrogen_indicator if sensor else None,
        thresholds=sensor_thresholds,
    )
    potassium = assess_nutrient(
        "potassium",
        sensor.potassium_indicator if sensor else None,
        thresholds=sensor_thresholds,
    )
    if nitrogen.value is not None and nitrogen.requires_lab_confirmation:
        flags.append(QualityFlag.N_REQUIRES_LAB_CONFIRMATION.value)
    if potassium.value is not None and potassium.requires_lab_confirmation:
        flags.append(QualityFlag.K_REQUIRES_LAB_CONFIRMATION.value)

    # --- 6-bosqich: fosfor moslik tekshiruvi (o'rtachalash YO'Q) ---
    comparison = compare_phosphorus_sources(
        colorimetric_olsen_p=estimate.olsen_p_mg_kg if estimate else None,
        sensor_indicator=sensor.phosphorus_indicator if sensor else None,
        thresholds=sensor_thresholds,
    )
    if comparison.agrees is False:
        flags.append(QualityFlag.P_INDICATORS_DISAGREE.value)
        notes.append(comparison.explanation_uz)

    # --- Demo / sintetik model bayroqlari ---
    if DataSource.DEMO.value in sources:
        flags.append(QualityFlag.DEMO_DATA.value)
    if artifact is not None:
        report = artifact.get("report")
        if getattr(report, "dataset_kind", "demo") == "demo":
            flags.append(QualityFlag.SYNTHETIC_MODEL.value)

    # Takrorlarni olib tashlash (tartibni saqlagan holda)
    flags = list(dict.fromkeys(flags))

    # --- 8-bosqich: umumiy ishonchlilik ---
    model_confidence = estimate.confidence if estimate else "low"
    confidence = compute_confidence(model_confidence, flags, rules)

    index = diagnostic_index(
        estimate.status_key if estimate else None, conditions, rules
    )

    if sensor is None:
        notes.append(
            "Universal sensor ma'lumoti kiritilmagan — tuproq sharoiti qo'lda kiritilgan "
            "qiymatlarga asoslandi."
        )

    return FusionResult(
        estimate=estimate,
        conditions=conditions,
        nitrogen=nitrogen,
        potassium=potassium,
        phosphorus_comparison=comparison,
        quality_flags=flags,
        confidence=confidence,
        confidence_label_uz=CONFIDENCE_LABELS_UZ[confidence],
        data_sources=list(dict.fromkeys(sources)),
        diagnostic_index=index,
        notes_uz=notes,
        ph=ph,
        ec_ds_m=ec,
        soil_moisture_percent=moisture,
        soil_temperature_c=soil_temperature,
    )
