"""Tuproq sharoiti o'lchovlarini talqin qilish (v0.2.0).

Bu modul sensor qiymatlarini konfiguratsiyadagi chegaralar bo'yicha sinflarga
ajratadi. Hech qanday chegara kod ichida qat'iy yozilmagan — barchasi
`config/sensor_thresholds.json` faylida.

Azot va kaliy uchun FAQAT sifatiy baho beriladi. Miqdoriy me'yor
`config/nutrient_calibration.json` da kalibrlash tasdiqlanmaguncha berilmaydi.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.data_validation import (
    load_nutrient_calibration,
    load_sensor_thresholds,
)

# Sifatiy baho kalitlari (N va K uchun)
QUALITATIVE_LABELS_UZ: dict[str, str] = {
    "likely_low": "Ehtimol past",
    "acceptable": "Qoniqarli",
    "likely_high": "Ehtimol yuqori",
    "unknown": "Ma'lumot yo'q",
}


@dataclass
class ParameterInterpretation:
    """Bitta parametrning talqini."""

    parameter: str
    label_uz: str
    value: float | None
    unit: str
    band_key: str
    band_label_uz: str
    level: str  # "ok" | "caution" | "attention" | "unknown"
    color: str
    screening_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NutrientAssessment:
    """Azot yoki kaliy uchun sifatiy baho."""

    nutrient: str
    label_uz: str
    value: float | None
    unit: str
    status_key: str
    status_label_uz: str
    level: str
    color: str
    advice_uz: str
    quantitative_enabled: bool
    requires_lab_confirmation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PARAMETER_LABELS_UZ: dict[str, str] = {
    "ph": "pH",
    "ec_ds_m": "Elektr o'tkazuvchanlik (EC)",
    "soil_moisture_percent": "Namlik",
    "soil_temperature_c": "Tuproq harorati",
    "nitrogen_indicator": "Azot indikatori (N)",
    "phosphorus_indicator": "Sensor P indikatori",
    "potassium_indicator": "Kaliy indikatori (K)",
}

_UNKNOWN = ParameterInterpretation(
    parameter="",
    label_uz="",
    value=None,
    unit="",
    band_key="unknown",
    band_label_uz="Ma'lumot yo'q",
    level="unknown",
    color="#9AA69C",
)


def interpret_parameter(
    parameter: str,
    value: float | None,
    thresholds: dict[str, Any] | None = None,
) -> ParameterInterpretation:
    """Qiymatni konfiguratsiyadagi diapazonlar bo'yicha sinflaydi."""

    thresholds = thresholds or load_sensor_thresholds()
    spec = thresholds.get(parameter)
    label = PARAMETER_LABELS_UZ.get(parameter, parameter)

    if spec is None:
        result = ParameterInterpretation(**{**asdict(_UNKNOWN), "parameter": parameter})
        result.label_uz = label
        return result

    unit = str(spec.get("unit", ""))
    screening = bool(spec.get("screening_only", False))

    if value is None:
        return ParameterInterpretation(
            parameter=parameter,
            label_uz=label,
            value=None,
            unit=unit,
            band_key="unknown",
            band_label_uz="Ma'lumot yo'q",
            level="unknown",
            color="#9AA69C",
            screening_only=screening,
        )

    bands = sorted(spec["bands"], key=lambda band: float(band["max"]))
    chosen = bands[-1]
    for band in bands:
        if value <= float(band["max"]):
            chosen = band
            break

    return ParameterInterpretation(
        parameter=parameter,
        label_uz=label,
        value=float(value),
        unit=unit,
        band_key=str(chosen["key"]),
        band_label_uz=str(chosen["label_uz"]),
        level=str(chosen.get("level", "ok")),
        color=str(chosen.get("color", "#27AE60")),
        screening_only=screening,
    )


def interpret_soil_conditions(
    ph: float | None,
    ec_ds_m: float | None,
    soil_moisture_percent: float | None,
    soil_temperature_c: float | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, ParameterInterpretation]:
    """Asosiy tuproq sharoiti parametrlarini birgalikda talqin qiladi."""

    thresholds = thresholds or load_sensor_thresholds()
    return {
        "ph": interpret_parameter("ph", ph, thresholds),
        "ec_ds_m": interpret_parameter("ec_ds_m", ec_ds_m, thresholds),
        "soil_moisture_percent": interpret_parameter(
            "soil_moisture_percent", soil_moisture_percent, thresholds
        ),
        "soil_temperature_c": interpret_parameter(
            "soil_temperature_c", soil_temperature_c, thresholds
        ),
    }


def assess_nutrient(
    nutrient: str,
    value: float | None,
    thresholds: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> NutrientAssessment:
    """Azot yoki kaliy uchun KONSERVATIV sifatiy baho.

    Miqdoriy me'yor faqat `nutrient_calibration.json` da tegishli bayroq
    yoqilgan bo'lsa beriladi. Standart holatda — berilmaydi.
    """

    if nutrient not in {"nitrogen", "potassium"}:
        raise ValueError("assess_nutrient faqat 'nitrogen' va 'potassium' uchun ishlaydi.")

    thresholds = thresholds or load_sensor_thresholds()
    calibration = calibration or load_nutrient_calibration()

    parameter = "nitrogen_indicator" if nutrient == "nitrogen" else "potassium_indicator"
    interpretation = interpret_parameter(parameter, value, thresholds)

    entry = calibration.get(nutrient, {})
    quantitative = bool(entry.get("quantitative_enabled", False))
    requires_lab = bool(entry.get("requires_lab_confirmation", True))
    advice_map = entry.get("qualitative_advice_uz", {})

    status_key = interpretation.band_key if value is not None else "unknown"
    advice = advice_map.get(
        status_key,
        "Ma'lumot yetarli emas. Laboratoriya tahlili tavsiya etiladi."
        if value is None
        else "Aniq me'yor uchun laboratoriya tasdig'i tavsiya etiladi.",
    )

    return NutrientAssessment(
        nutrient=nutrient,
        label_uz="Azot (N)" if nutrient == "nitrogen" else "Kaliy (K)",
        value=None if value is None else float(value),
        unit=interpretation.unit,
        status_key=status_key,
        status_label_uz=QUALITATIVE_LABELS_UZ.get(status_key, interpretation.band_label_uz),
        level=interpretation.level,
        color=interpretation.color,
        advice_uz=advice,
        quantitative_enabled=quantitative,
        requires_lab_confirmation=requires_lab,
    )


#: Holat darajasining vizual "maqbullik" ballari (0-100).
LEVEL_SCORES: dict[str, float] = {"ok": 88.0, "caution": 55.0, "attention": 25.0}


def condition_chart_data(
    conditions: dict[str, ParameterInterpretation],
    order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Tuproq sharoiti grafigi uchun ma'lumot tayyorlaydi.

    `normalized` — qiymatning maqbul oraliqqa nisbatan "maqbullik" ko'rsatkichi
    (mutlaq qiymat emas), `display` — foydalanuvchiga ko'rsatiladigan haqiqiy qiymat.
    """

    order = order or ["ph", "ec_ds_m", "soil_moisture_percent", "soil_temperature_c"]
    rows: list[dict[str, Any]] = []
    for parameter in order:
        interpretation = conditions.get(parameter)
        if interpretation is None or interpretation.value is None:
            continue
        unit = interpretation.unit
        if parameter == "ph":
            display = f"{interpretation.value:.1f} · {interpretation.band_label_uz}"
        elif parameter == "soil_moisture_percent":
            display = f"{interpretation.value:.0f}% · {interpretation.band_label_uz}"
        elif parameter == "soil_temperature_c":
            display = f"{interpretation.value:.1f} °C · {interpretation.band_label_uz}"
        else:
            display = f"{interpretation.value:.2f} {unit} · {interpretation.band_label_uz}"
        rows.append(
            {
                "parameter": parameter,
                "label_uz": interpretation.label_uz,
                "normalized": LEVEL_SCORES.get(interpretation.level, 50.0),
                "display": display,
                "color": interpretation.color,
                "level": interpretation.level,
            }
        )
    return rows


def diagnostic_index(
    phosphorus_status_key: str | None,
    conditions: dict[str, ParameterInterpretation],
    rules: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Dastlabki AgroIQ diagnostika indeksi (0-100).

    DIQQAT: bu sertifikatlangan tuproq unumdorligi bahosi EMAS. U faqat
    kiritilgan o'lchovlarning maqbul oraliqdan chetlanishini umumlashtiradi.
    Hisobga olinmagan parametrlar indeksga qo'shilmaydi va og'irliklar
    mavjud komponentlar bo'yicha qayta normallashtiriladi.
    """

    from src.data_validation import load_recommendation_rules

    rules = rules or load_recommendation_rules()
    spec = rules.get("diagnostic_index", {})
    if not spec.get("enabled", False):
        return None

    phosphorus_scores = {
        "very_low": 15.0,
        "low": 40.0,
        "medium": 65.0,
        "sufficient": 92.0,
        "high": 78.0,  # Yuqori fosfor ham ideal emas (ortiqcha zaxira).
    }
    level_scores = {"ok": 92.0, "caution": 58.0, "attention": 25.0}

    component_values: dict[str, float] = {}
    if phosphorus_status_key in phosphorus_scores:
        component_values["phosphorus"] = phosphorus_scores[phosphorus_status_key]
    for component, parameter in (
        ("ph", "ph"),
        ("ec", "ec_ds_m"),
        ("moisture", "soil_moisture_percent"),
    ):
        interpretation = conditions.get(parameter)
        if interpretation is not None and interpretation.level in level_scores:
            component_values[component] = level_scores[interpretation.level]

    if not component_values:
        return None

    components = spec.get("components", [])
    total_weight = 0.0
    weighted = 0.0
    breakdown: list[dict[str, Any]] = []
    for component in components:
        key = component["key"]
        if key not in component_values:
            continue
        weight = float(component.get("weight", 0.0))
        score = component_values[key]
        weighted += weight * score
        total_weight += weight
        breakdown.append(
            {"key": key, "label_uz": component.get("label_uz", key), "score": round(score, 1)}
        )

    if total_weight <= 0:
        return None

    score = weighted / total_weight
    if score >= 80:
        label, color = "Yaxshi", "#27AE60"
    elif score >= 60:
        label, color = "Qoniqarli", "#8BBF3D"
    elif score >= 40:
        label, color = "E'tibor talab qiladi", "#E67E22"
    else:
        label, color = "Past", "#C0392B"

    return {
        "score": round(score, 1),
        "label_uz": label,
        "color": color,
        "title_uz": spec.get("label_uz", "Dastlabki AgroIQ diagnostika indeksi"),
        "disclaimer_uz": spec.get("disclaimer_uz", ""),
        "components": breakdown,
        "components_used": len(breakdown),
        "components_total": len(components),
    }
