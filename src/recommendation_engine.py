"""Shaffof (oq quti) o'g'itlash tavsiyalari dvigateli.

Bu modul QORA QUTI EMAS. Har bir raqam agronomik qoidalar va JSON
konfiguratsiyasidagi koeffitsientlardan kelib chiqib hisoblanadi, va har bir
hisoblash qadami `calculation_steps` ro'yxatida saqlanadi.

Asosiy formula:

    kerakli_P2O5 (kg/ga) = maqsadli_hosil (t/ga)
                           x P2O5_olib_chiqish (kg/t)
                           x fosfor_holati_koeffitsienti
                           x pH_tuzatish_koeffitsienti

    o'g'it_mahsuloti (kg/ga) = kerakli_P2O5 / mahsulotdagi_P2O5_ulushi
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src import PILOT_RECOMMENDATION_NOTICE_UZ
from src.data_validation import (
    ConfigError,
    get_crop,
    get_product,
    load_crop_profiles,
    load_fertilizer_products,
)

BAG_WEIGHT_KG = 50.0

CONFIDENCE_ORDER = ["low", "medium", "high"]
CONFIDENCE_LABELS_UZ = {"high": "Yuqori", "medium": "O'rtacha", "low": "Past"}


@dataclass
class SoilWarning:
    """Agronomik ogohlantirish (neytral, tavsiyaviy uslubda)."""

    key: str
    level: str  # "info" | "caution" | "attention"
    title_uz: str
    message_uz: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    """To'liq o'g'itlash tavsiyasi."""

    crop_key: str
    crop_name_uz: str
    status_key: str
    status_label_uz: str
    product_key: str
    product_name_uz: str
    p2o5_fraction: float
    required_p2o5_low: float
    required_p2o5_high: float
    product_kg_ha_low: float
    product_kg_ha_high: float
    total_product_kg_low: float
    total_product_kg_high: float
    total_bags_low: float
    total_bags_high: float
    field_area_ha: float
    target_yield_t_ha: float
    application_stage_uz: str
    application_method_uz: str
    warnings: list[SoilWarning]
    confidence: str
    confidence_label_uz: str
    calculation_steps: list[dict[str, Any]]
    notice_uz: str = PILOT_RECOMMENDATION_NOTICE_UZ
    estimated_cost_uzs_low: float | None = None
    estimated_cost_uzs_high: float | None = None
    no_phosphorus_needed: bool = False
    agronomic_notes_uz: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = [warning.to_dict() for warning in self.warnings]
        return payload


# ---------------------------------------------------------------------------
# Yordamchi funksiyalar
# ---------------------------------------------------------------------------


def _round_to(value: float, step: float = 5.0) -> float:
    """Dala amaliyoti uchun qulay qadamga yaxlitlash."""

    if value <= 0:
        return 0.0
    return round(round(value / step) * step, 1)


def ph_adjustment(ph: float, profiles: dict[str, Any] | None = None) -> dict[str, Any]:
    """pH bo'yicha tuzatish koeffitsientini qaytaradi."""

    profiles = profiles or load_crop_profiles()
    rules = sorted(profiles["global_ph_adjustment"], key=lambda item: float(item["max_ph"]))
    for rule in rules:
        if ph <= float(rule["max_ph"]):
            return rule
    return rules[-1]


def status_multiplier(status_key: str, profiles: dict[str, Any] | None = None) -> dict[str, float]:
    """Fosfor holati bo'yicha koeffitsient oralig'ini qaytaradi."""

    profiles = profiles or load_crop_profiles()
    multipliers = profiles["status_multipliers"]
    if status_key not in multipliers:
        raise ConfigError(
            f"'crop_profiles.json': '{status_key}' fosfor sinfi uchun koeffitsient aniqlanmagan."
        )
    entry = multipliers[status_key]
    return {"low": float(entry["low"]), "high": float(entry["high"])}


def fertilizer_from_p2o5(required_p2o5_kg_ha: float, p2o5_fraction: float) -> float:
    """Shaffof konversiya: o'g'it (kg/ga) = kerakli P2O5 / mahsulotdagi P2O5 ulushi."""

    if not 0 < p2o5_fraction <= 1:
        raise ValueError("P2O5 ulushi 0 va 1 orasida bo'lishi kerak.")
    if required_p2o5_kg_ha < 0:
        raise ValueError("Kerakli P2O5 manfiy bo'lishi mumkin emas.")
    return required_p2o5_kg_ha / p2o5_fraction


# ---------------------------------------------------------------------------
# Tuproq sharoiti ogohlantirishlari
# ---------------------------------------------------------------------------


def build_soil_warnings(
    ph: float,
    ec_ds_m: float,
    moisture_pct: float,
    status_key: str,
    crop: dict[str, Any],
    profiles: dict[str, Any] | None = None,
) -> list[SoilWarning]:
    """Tuproq sharoiti bo'yicha neytral agronomik ogohlantirishlarni shakllantiradi."""

    profiles = profiles or load_crop_profiles()
    rules = profiles["soil_condition_rules"]
    warnings: list[SoilWarning] = []

    # --- pH ---
    if ph >= float(rules["ph_very_high"]):
        warnings.append(
            SoilWarning(
                key="ph_very_high",
                level="attention",
                title_uz="Yuqori pH (kuchli ishqoriy muhit)",
                message_uz=(
                    f"pH {ph:.1f} — ishqoriy va karbonatga boy tuproqlarda fosforning o'simlik "
                    "tomonidan o'zlashtirilishi kamayishi mumkin. Lokal (tasmali) o'g'itlash va "
                    "organik o'g'it qo'shish samaradorlikni oshirishi mumkin."
                ),
            )
        )
    elif ph >= float(rules["ph_high"]):
        warnings.append(
            SoilWarning(
                key="ph_high",
                level="caution",
                title_uz="Ishqoriy muhitga moyillik",
                message_uz=(
                    f"pH {ph:.1f} — bu qiymatda fosforning bir qismi kalsiy birikmalariga "
                    "bog'lanib, o'simlik uchun kamroq o'zlashadigan holatga o'tishi mumkin."
                ),
            )
        )
    elif ph <= float(rules["ph_low"]):
        warnings.append(
            SoilWarning(
                key="ph_low",
                level="caution",
                title_uz="Kislotali muhit",
                message_uz=(
                    f"pH {ph:.1f} — kislotali tuproqda fosfor temir va alyuminiy birikmalari bilan "
                    "bog'lanishi mumkin. Ohaklash zaruriyatini agronom bilan baholang."
                ),
            )
        )

    # --- EC (sho'rlanish) ---
    if ec_ds_m >= float(rules["ec_high"]):
        warnings.append(
            SoilWarning(
                key="ec_high",
                level="attention",
                title_uz="Yuqori elektr o'tkazuvchanlik (sho'rlanish xavfi)",
                message_uz=(
                    f"EC {ec_ds_m:.1f} dS/m — sho'rlanish darajasi yuqori. Bu oziq moddalarning "
                    "o'zlashtirilishini cheklashi mumkin; qo'shimcha agronomik baholash va "
                    "yuvish (drenaj) tadbirlari ko'rib chiqilishi tavsiya etiladi."
                ),
            )
        )
    elif ec_ds_m >= float(rules["ec_moderate"]):
        warnings.append(
            SoilWarning(
                key="ec_moderate",
                level="caution",
                title_uz="O'rtacha sho'rlanish",
                message_uz=(
                    f"EC {ec_ds_m:.1f} dS/m — o'rtacha sho'rlanish belgilari mavjud. "
                    "O'g'it me'yorini bir yo'la emas, bosqichma-bosqich berish maqsadga muvofiq."
                ),
            )
        )

    crop_ec_limit = crop.get("ec_sensitivity_ds_m")
    if crop_ec_limit is not None and ec_ds_m >= float(crop_ec_limit):
        warnings.append(
            SoilWarning(
                key="ec_crop_threshold",
                level="caution",
                title_uz=f"{crop['name_uz']} uchun sho'rlanish chegarasi",
                message_uz=str(crop.get("ec_note_uz", "")),
            )
        )

    # --- Namlik ---
    if moisture_pct <= float(rules["moisture_very_low"]):
        warnings.append(
            SoilWarning(
                key="moisture_very_low",
                level="attention",
                title_uz="Juda past namlik",
                message_uz=(
                    f"Namlik {moisture_pct:.0f}% — quruq tuproqda o'g'itning erishi va ildiz orqali "
                    "o'zlashtirilishi cheklanishi mumkin. O'g'itlashni sug'orish bilan "
                    "muvofiqlashtirish tavsiya etiladi."
                ),
            )
        )
    elif moisture_pct <= float(rules["moisture_low"]):
        warnings.append(
            SoilWarning(
                key="moisture_low",
                level="caution",
                title_uz="Past namlik",
                message_uz=(
                    f"Namlik {moisture_pct:.0f}% — o'g'it samaradorligini oshirish uchun uni "
                    "sug'orishdan oldin yoki sug'orish bilan birga berish maqsadga muvofiq."
                ),
            )
        )
    elif moisture_pct >= float(rules["moisture_high"]):
        warnings.append(
            SoilWarning(
                key="moisture_high",
                level="info",
                title_uz="Yuqori namlik",
                message_uz=(
                    f"Namlik {moisture_pct:.0f}% — ortiqcha nam sharoitda oziq moddalarning "
                    "yuvilib ketishi ehtimoli ortadi. Drenaj holatini tekshiring."
                ),
            )
        )

    # --- Fosfor holati ---
    if status_key == "high":
        warnings.append(
            SoilWarning(
                key="phosphorus_high",
                level="info",
                title_uz="Fosfor zaxirasi yuqori",
                message_uz=(
                    "Tuproqdagi fosfor darajasi yuqori baholandi. Qo'shimcha fosforli o'g'it "
                    "berish iqtisodiy va ekologik jihatdan asoslanmasligi mumkin; mablag'ni azot "
                    "yoki kaliy me'yorini optimallashtirishga yo'naltirish ko'rib chiqilishi mumkin."
                ),
            )
        )
    elif status_key == "very_low":
        warnings.append(
            SoilWarning(
                key="phosphorus_very_low",
                level="attention",
                title_uz="Fosfor darajasi juda past",
                message_uz=(
                    "Fosfor zaxirasi juda cheklangan deb baholandi. Bu holatda o'g'itlash me'yorini "
                    "bir necha mavsumga taqsimlab, tuproq zaxirasini bosqichma-bosqich tiklash "
                    "yondashuvi ko'rib chiqilishi mumkin."
                ),
            )
        )

    return warnings


# ---------------------------------------------------------------------------
# Asosiy tavsiya funksiyasi
# ---------------------------------------------------------------------------


def _downgrade(level: str, steps: int = 1) -> str:
    index = max(CONFIDENCE_ORDER.index(level) - steps, 0)
    return CONFIDENCE_ORDER[index]


def recommendation_confidence(
    model_confidence: str,
    within_calibration: bool,
    warnings: list[SoilWarning],
) -> str:
    """Tavsiya ishonchliligi: model ishonchidan boshlanadi va konservativ pasaytiriladi."""

    level = model_confidence if model_confidence in CONFIDENCE_ORDER else "low"
    if not within_calibration:
        level = "low"
    attention_count = sum(1 for warning in warnings if warning.level == "attention")
    if attention_count >= 2:
        level = _downgrade(level, 2)
    elif attention_count == 1:
        level = _downgrade(level, 1)
    return level


def build_recommendation(
    olsen_p_mg_kg: float,
    status_key: str,
    status_label_uz: str,
    ph: float,
    ec_ds_m: float,
    moisture_pct: float,
    crop_key: str,
    field_area_ha: float,
    target_yield_t_ha: float,
    fertilizer_key: str,
    model_confidence: str = "medium",
    within_calibration: bool = True,
    profiles: dict[str, Any] | None = None,
    products: dict[str, Any] | None = None,
) -> Recommendation:
    """Shaffof agronomik qoidalar asosida o'g'itlash tavsiyasini quradi."""

    profiles = profiles or load_crop_profiles()
    products = products or load_fertilizer_products()
    crop = get_crop(crop_key, profiles)
    product = get_product(fertilizer_key, products)

    if field_area_ha <= 0:
        raise ValueError("Dala maydoni musbat bo'lishi kerak.")
    if target_yield_t_ha <= 0:
        raise ValueError("Maqsadli hosildorlik musbat bo'lishi kerak.")

    steps: list[dict[str, Any]] = []

    # 1-qadam: hosil bilan olib chiqiladigan P2O5.
    removal = float(crop["p2o5_removal_kg_per_ton"])
    base_p2o5 = target_yield_t_ha * removal
    steps.append(
        {
            "step": "1. Hosil bilan olib chiqiladigan fosfor",
            "formula": f"{target_yield_t_ha:.2f} t/ga x {removal:.1f} kg P2O5/t",
            "result": f"{base_p2o5:.1f} kg P2O5/ga",
        }
    )

    # 2-qadam: tuproqdagi fosfor holati koeffitsienti.
    multiplier = status_multiplier(status_key, profiles)
    steps.append(
        {
            "step": "2. Tuproqdagi fosfor holati koeffitsienti",
            "formula": f"Holat: {status_label_uz} (Olsen-P ≈ {olsen_p_mg_kg:.1f} mg/kg)",
            "result": f"x {multiplier['low']:.2f} … {multiplier['high']:.2f}",
        }
    )

    # 3-qadam: pH tuzatishi.
    ph_rule = ph_adjustment(ph, profiles)
    ph_low = float(ph_rule["factor_low"])
    ph_high = float(ph_rule["factor_high"])
    steps.append(
        {
            "step": "3. pH tuzatish koeffitsienti",
            "formula": f"pH = {ph:.2f} → {ph_rule['reason_uz']}",
            "result": f"x {ph_low:.2f} … {ph_high:.2f}",
        }
    )

    # 4-qadam: kerakli P2O5 oralig'i.
    required_low = base_p2o5 * multiplier["low"] * ph_low
    required_high = base_p2o5 * multiplier["high"] * ph_high
    min_cap = float(crop.get("min_p2o5_kg_ha", 0.0))
    max_cap = float(crop.get("max_p2o5_kg_ha", 250.0))
    required_low = _round_to(min(max(required_low, min_cap), max_cap))
    required_high = _round_to(min(max(required_high, min_cap), max_cap))
    if required_high < required_low:
        required_low, required_high = required_high, required_low
    steps.append(
        {
            "step": "4. Kerakli fosfor me'yori",
            "formula": "1-qadam x 2-qadam x 3-qadam (ekin uchun ruxsat etilgan chegaralarda)",
            "result": f"{required_low:.0f} … {required_high:.0f} kg P2O5/ga",
        }
    )

    # 5-qadam: mahsulot miqdoriga konversiya.
    fraction = float(product["p2o5_fraction"])
    product_low = fertilizer_from_p2o5(required_low, fraction)
    product_high = fertilizer_from_p2o5(required_high, fraction)
    steps.append(
        {
            "step": "5. O'g'it mahsuloti miqdori",
            "formula": (
                f"kerakli P2O5 / mahsulotdagi P2O5 ulushi = "
                f"{required_low:.0f}…{required_high:.0f} / {fraction:.2f}"
            ),
            "result": f"{product_low:.0f} … {product_high:.0f} kg/ga ({product['name_uz']})",
        }
    )

    # 6-qadam: dala maydoniga ko'paytirish.
    total_low = product_low * field_area_ha
    total_high = product_high * field_area_ha
    steps.append(
        {
            "step": "6. Dala uchun umumiy miqdor",
            "formula": f"{product_low:.0f}…{product_high:.0f} kg/ga x {field_area_ha:.2f} ga",
            "result": f"{total_low:.0f} … {total_high:.0f} kg",
        }
    )

    warnings = build_soil_warnings(ph, ec_ds_m, moisture_pct, status_key, crop, profiles)
    confidence = recommendation_confidence(model_confidence, within_calibration, warnings)

    no_phosphorus_needed = required_high <= 0.0
    stage = str(crop.get("application_stage_uz", ""))
    method = str(crop.get("application_method_uz", ""))
    if no_phosphorus_needed:
        stage = "Ushbu mavsumda fosforli o'g'it berish tavsiya etilmaydi."
        method = (
            "Tuproq zaxirasi yetarli — keyingi mavsumda takroriy tahlil o'tkazish tavsiya etiladi."
        )

    price = product.get("typical_price_uzs_per_kg")
    cost_low = float(price) * total_low if price else None
    cost_high = float(price) * total_high if price else None

    return Recommendation(
        crop_key=crop["key"],
        crop_name_uz=crop["name_uz"],
        status_key=status_key,
        status_label_uz=status_label_uz,
        product_key=product["key"],
        product_name_uz=product["name_uz"],
        p2o5_fraction=fraction,
        required_p2o5_low=round(required_low, 1),
        required_p2o5_high=round(required_high, 1),
        product_kg_ha_low=round(product_low, 1),
        product_kg_ha_high=round(product_high, 1),
        total_product_kg_low=round(total_low, 1),
        total_product_kg_high=round(total_high, 1),
        total_bags_low=round(total_low / BAG_WEIGHT_KG, 1),
        total_bags_high=round(total_high / BAG_WEIGHT_KG, 1),
        field_area_ha=field_area_ha,
        target_yield_t_ha=target_yield_t_ha,
        application_stage_uz=stage,
        application_method_uz=method,
        warnings=warnings,
        confidence=confidence,
        confidence_label_uz=CONFIDENCE_LABELS_UZ[confidence],
        calculation_steps=steps,
        estimated_cost_uzs_low=round(cost_low) if cost_low is not None else None,
        estimated_cost_uzs_high=round(cost_high) if cost_high is not None else None,
        no_phosphorus_needed=no_phosphorus_needed,
        agronomic_notes_uz=list(crop.get("agronomic_notes_uz", [])),
    )
