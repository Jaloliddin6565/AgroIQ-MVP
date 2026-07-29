"""Tushuntirishlar: "Nima uchun bu tavsiya berildi?" bo'limi uchun matnlar.

Tushuntirishlar generativ model tomonidan EMAS, balki hisoblash jarayonidagi
aniq faktlardan shakllantiriladi. Shu sababli ular har doim hisob-kitob bilan
mos keladi.
"""

from __future__ import annotations

from typing import Any

from src.model_inference import PhosphorusEstimate
from src.recommendation_engine import Recommendation


def _ph_sentence(ph: float) -> str | None:
    if ph >= 8.3:
        return f"Tuproq pH qiymati {ph:.1f} — kuchli ishqoriy muhitni ko'rsatmoqda."
    if ph >= 7.8:
        return f"Tuproq pH qiymati {ph:.1f} — ishqoriy muhitga moyillik aniqlandi."
    if ph <= 5.8:
        return f"Tuproq pH qiymati {ph:.1f} — kislotali muhitni ko'rsatmoqda."
    return f"Tuproq pH qiymati {ph:.1f} — fosfor o'zlashtirilishi uchun qulay oraliqda."


def build_explanations(
    estimate: PhosphorusEstimate,
    recommendation: Recommendation,
    ph: float,
    ec_ds_m: float,
    moisture_pct: float,
) -> list[str]:
    """3-5 ta qisqa, tushunarli tushuntirish qaytaradi."""

    reasons: list[str] = []

    # 1. Fosfor holati — har doim birinchi.
    reasons.append(
        f"Kolorimetrik o'lchov asosida o'simlik o'zlashtira oladigan fosfor "
        f"{estimate.olsen_p_mg_kg:.1f} mg/kg deb baholandi — bu \"{estimate.status_label_uz}\" darajaga to'g'ri keladi."
    )

    # 2. pH ta'siri.
    ph_sentence = _ph_sentence(ph)
    if ph_sentence:
        reasons.append(ph_sentence)
    if ph >= 7.8:
        reasons.append(
            "Yuqori pH fosforning o'simlik tomonidan o'zlashtirilishini cheklashi mumkin, "
            "shuning uchun me'yorga tuzatish koeffitsienti qo'llanildi."
        )

    # 3. Ekin va maqsadli hosildorlik.
    reasons.append(
        f"Tanlangan ekin ({recommendation.crop_name_uz}) va maqsadli hosildorlik "
        f"({recommendation.target_yield_t_ha:g} t/ga) hisobga olindi."
    )

    # 4. Sho'rlanish yoki namlik — faqat ahamiyatli bo'lsa.
    if ec_ds_m >= 4.0:
        reasons.append(
            f"Elektr o'tkazuvchanlik {ec_ds_m:.1f} dS/m — sho'rlanish oziq moddalar "
            "o'zlashtirilishini cheklashi mumkinligi inobatga olindi."
        )
    elif moisture_pct <= 12.0:
        reasons.append(
            f"Tuproq namligi {moisture_pct:.0f}% — quruq sharoitda o'g'it samaradorligi "
            "pasayishi mumkinligi hisobga olindi."
        )

    # 5. Konversiya izohi.
    if recommendation.no_phosphorus_needed:
        reasons.append(
            "Fosfor zaxirasi yuqori bo'lgani uchun ushbu mavsumda qo'shimcha fosforli o'g'it "
            "tavsiya etilmadi."
        )
    else:
        reasons.append(
            f"O'g'it miqdori mahsulotdagi P2O5 ulushiga ko'ra hisoblandi: "
            f"{recommendation.product_name_uz} tarkibida {recommendation.p2o5_fraction * 100:.0f}% P2O5."
        )

    # 6. Ishonch darajasi past bo'lsa — ochiq aytiladi.
    if estimate.confidence == "low" or not estimate.within_calibration:
        reasons.append(
            "Model ishonchliligi past baholandi — natijani laboratoriya tahlili bilan "
            "tasdiqlash tavsiya etiladi."
        )

    return reasons[:6]


def technical_details(
    estimate: PhosphorusEstimate,
    recommendation: Recommendation,
    model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Agronom uchun texnik tafsilotlar to'plami."""

    details: dict[str, Any] = {
        "Baholangan Olsen-P": f"{estimate.olsen_p_mg_kg:.2f} mg/kg",
        "Standart noaniqlik (±1σ)": f"±{estimate.uncertainty_mg_kg:.2f} mg/kg",
        "Taxminiy 95% oraliq": f"{estimate.ci95_low:.1f} … {estimate.ci95_high:.1f} mg/kg",
        "Model qoldiq tarqoqligi": f"{estimate.model_residual_std:.2f} mg/kg",
        "Model ishonchliligi": estimate.confidence_label_uz,
        "Tavsiya ishonchliligi": recommendation.confidence_label_uz,
        "Kalibrlash oralig'ida": "Ha" if estimate.within_calibration else "Yo'q",
    }
    # Daraxtlar tarqoqligi faqat Random Forest tanlanganda ma'noga ega.
    if estimate.tree_std > 0:
        details["Daraxtlar tarqoqligi (RF)"] = f"{estimate.tree_std:.2f} mg/kg"
    if model_info:
        metrics = model_info.get("metrics", {})
        details["Model"] = model_info.get("model_name", "-")
        details["R²"] = f"{metrics.get('r2', 0):.3f}"
        details["RMSE"] = f"{metrics.get('rmse', 0):.2f} mg/kg"
        details["MAE"] = f"{metrics.get('mae', 0):.2f} mg/kg"
        details["Dataset turi"] = (
            "Demo (sintetik)" if model_info.get("dataset_kind") == "demo" else "Real"
        )
    return details
