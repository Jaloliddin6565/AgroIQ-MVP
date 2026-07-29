"""AgroIQ — tuproq diagnostikasi va aqlli o'g'itlash platformasi.

Streamlit ilovasi. Ishga tushirish:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    DEMO_MODEL_BANNER_UZ,
    DISCLAIMER_UZ,
    PILOT_RECOMMENDATION_NOTICE_UZ,
    __app_name__,
    __version__,
)
from src.data_validation import (  # noqa: E402
    ConfigError,
    DATA_DIR,
    load_crop_profiles,
    load_fertilizer_products,
    load_thresholds,
    validate_analysis_request,
)
from src.explanations import build_explanations, technical_details  # noqa: E402
from src.model_inference import (  # noqa: E402
    ModelNotAvailableError,
    estimate_phosphorus,
    load_artifact,
    model_summary,
)
from src.model_training import DEMO_SCENARIOS, build_demo_samples, feature_importances  # noqa: E402
from src.recommendation_engine import build_recommendation  # noqa: E402
from src.report_generator import ReportError, build_pdf_report, make_analysis_id  # noqa: E402
from src.ui_components import (  # noqa: E402
    ACCENT,
    INK,
    MUTED,
    PRIMARY,
    alert_card,
    banner,
    color_swatch,
    confidence_badge,
    disclaimer_box,
    feature_importance_chart,
    info_card,
    inject_css,
    metric_card,
    model_comparison_chart,
    phosphorus_gauge,
    prediction_scatter,
    render_logo,
    step_card,
)

PAGE_HOME = "Bosh sahifa"
PAGE_ANALYSIS = "Yangi tahlil"
PAGE_RESULTS = "Natijalar"
PAGE_MODEL = "Model va validatsiya"
PAGE_DEMO = "Demo rejimi"
PAGE_ABOUT = "Loyiha haqida"

PAGES = [PAGE_HOME, PAGE_ANALYSIS, PAGE_RESULTS, PAGE_MODEL, PAGE_DEMO, PAGE_ABOUT]
PAGE_ICONS = {
    PAGE_HOME: "🏠",
    PAGE_ANALYSIS: "🧪",
    PAGE_RESULTS: "📊",
    PAGE_MODEL: "🤖",
    PAGE_DEMO: "🎬",
    PAGE_ABOUT: "💡",
}

st.set_page_config(
    page_title="AgroIQ — tuproq diagnostikasi",
    page_icon="🌱",
    layout="wide",
    # "auto" — mobil ekranlarda yon panel avtomatik yig'iladi va kontentni to'smaydi.
    initial_sidebar_state="auto",
)


# ---------------------------------------------------------------------------
# Keshlangan resurslar
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_artifact() -> dict[str, Any]:
    return load_artifact()


@st.cache_data(show_spinner=False)
def get_configs() -> dict[str, Any]:
    return {
        "thresholds": load_thresholds(),
        "crops": load_crop_profiles(),
        "products": load_fertilizer_products(),
    }


@st.cache_data(show_spinner=False)
def get_demo_samples() -> pd.DataFrame:
    """Demo namunalarni CSV dan o'qiydi; fayl bo'lmasa — kodda tayyorlangan ssenariylar."""

    path = DATA_DIR / "demo_samples.csv"
    if path.exists():
        try:
            frame = pd.read_csv(path)
            if not frame.empty and "demo_id" in frame.columns:
                return frame
        except Exception:  # noqa: BLE001 - fayl buzilgan bo'lsa zaxira variantga o'tamiz
            pass
    return build_demo_samples()


@st.cache_data(show_spinner=False)
def get_model_info() -> dict[str, Any]:
    return model_summary(get_artifact())


@st.cache_data(show_spinner=False)
def get_feature_importances() -> pd.DataFrame | None:
    return feature_importances(get_artifact()["pipeline"])


# ---------------------------------------------------------------------------
# Navigatsiya
# ---------------------------------------------------------------------------


def goto(page: str) -> None:
    """Boshqa bo'limga o'tish.

    Streamlit widget kaliti (`nav`) widget yaratilgandan keyin o'zgartirilmaydi,
    shuning uchun o'tish so'rovi vaqtincha saqlanadi va keyingi ishga tushirishda,
    yon panel chizilishidan oldin qo'llaniladi (`apply_pending_navigation`).
    """

    st.session_state["_pending_nav"] = page
    st.rerun()


def apply_pending_navigation() -> None:
    """Kutilayotgan o'tishni yon panel widget'i yaratilishidan oldin qo'llaydi."""

    pending = st.session_state.pop("_pending_nav", None)
    if pending in PAGES:
        st.session_state["nav"] = pending


def _crop_options(configs: dict[str, Any]) -> dict[str, str]:
    return {crop["key"]: crop["name_uz"] for crop in configs["crops"]["crops"]}


def _product_options(configs: dict[str, Any]) -> dict[str, str]:
    return {item["key"]: item["name_uz"] for item in configs["products"]["products"]}


# ---------------------------------------------------------------------------
# Tahlilni bajarish
# ---------------------------------------------------------------------------


def run_analysis(payload: dict[str, Any], sample_label: str = "Namuna") -> bool:
    """Kirishni tekshiradi, modelni ishga tushiradi va natijani sessiyaga saqlaydi."""

    request, errors = validate_analysis_request(
        {
            "colorimetry": {
                "red": payload["red"],
                "green": payload["green"],
                "blue": payload["blue"],
                "reaction_time_sec": payload["reaction_time_sec"],
                "sample_temperature_c": payload["sample_temperature_c"],
            },
            "soil": {
                "ph": payload["ph"],
                "ec_ds_m": payload["ec_ds_m"],
                "moisture_pct": payload["moisture_pct"],
                "crop_key": payload["crop_key"],
                "field_area_ha": payload["field_area_ha"],
                "target_yield_t_ha": payload["target_yield_t_ha"],
                "fertilizer_key": payload["fertilizer_key"],
            },
            "sample_label": sample_label,
        }
    )
    if request is None:
        st.error("Kiritilgan ma'lumotlarda xatolik bor:")
        for message in errors:
            st.markdown(f"- {message}")
        return False

    try:
        configs = get_configs()
        artifact = get_artifact()
        colorimetry = request.colorimetry
        soil = request.soil

        estimate = estimate_phosphorus(
            red=colorimetry.red,
            green=colorimetry.green,
            blue=colorimetry.blue,
            reaction_time_sec=colorimetry.reaction_time_sec,
            sample_temperature_c=colorimetry.sample_temperature_c,
            artifact=artifact,
            thresholds=configs["thresholds"],
        )
        recommendation = build_recommendation(
            olsen_p_mg_kg=estimate.olsen_p_mg_kg,
            status_key=estimate.status_key,
            status_label_uz=estimate.status_label_uz,
            ph=soil.ph,
            ec_ds_m=soil.ec_ds_m,
            moisture_pct=soil.moisture_pct,
            crop_key=soil.crop_key,
            field_area_ha=soil.field_area_ha,
            target_yield_t_ha=soil.target_yield_t_ha,
            fertilizer_key=soil.fertilizer_key,
            model_confidence=estimate.confidence,
            within_calibration=estimate.within_calibration,
            profiles=configs["crops"],
            products=configs["products"],
        )
        explanations = build_explanations(
            estimate, recommendation, soil.ph, soil.ec_ds_m, soil.moisture_pct
        )
    except ModelNotAvailableError as exc:
        st.error(str(exc))
        return False
    except ConfigError as exc:
        st.error(f"Konfiguratsiya xatosi: {exc}")
        return False
    except (ValueError, KeyError) as exc:
        st.error(f"Tahlilni bajarib bo'lmadi: {exc}")
        return False

    st.session_state["result"] = {
        "estimate": estimate,
        "recommendation": recommendation,
        "explanations": explanations,
        "measurements": {
            "red": colorimetry.red,
            "green": colorimetry.green,
            "blue": colorimetry.blue,
            "reaction_time_sec": colorimetry.reaction_time_sec,
            "sample_temperature_c": colorimetry.sample_temperature_c,
            "ph": soil.ph,
            "ec_ds_m": soil.ec_ds_m,
            "moisture_pct": soil.moisture_pct,
        },
        "sample_label": sample_label,
        "analysis_id": make_analysis_id(),
    }
    return True


# ---------------------------------------------------------------------------
# Sahifalar
# ---------------------------------------------------------------------------


def page_home(configs: dict[str, Any], model_info: dict[str, Any] | None) -> None:
    st.markdown(
        """
        <div class="agro-hero">
            <span class="eyebrow">AI · Tuproq diagnostikasi · AgriTech</span>
            <h1>AgroIQ — tuproq diagnostikasi va aniq o'g'itlash uchun aqlli platforma</h1>
            <p>O'simlik o'zlashtira oladigan fosforni tezkor baholang va dalangiz uchun
            tushunarli o'g'itlash tavsiyasini oling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Muammo")
        st.markdown(
            "Fermerlar uchun dala sharoitida o'simlik o'zlashtira oladigan fosforni tez va "
            "arzon o'lchash imkoniyati deyarli yo'q. Laboratoriya tahlili qimmat va uzoq davom "
            "etadi, natijada o'g'it ko'pincha **taxminiy me'yorda** beriladi. Bu ikki tomonlama "
            "yo'qotishga olib keladi: keraksiz xarajat yoki hosildorlikning pasayishi."
        )
        st.markdown("#### Yechim")
        st.markdown(
            "AgroIQ portativ kolorimetrik o'lchov va sun'iy intellektni birlashtiradi. "
            "Qurilma rang o'zgarishini o'lchaydi, AI modeli fosfor miqdorini baholaydi, "
            "shaffof agronomik qoidalar esa aniq o'g'itlash me'yorini hisoblab beradi — "
            "**tushuntirish va ishonch darajasi bilan birga**."
        )
    with right:
        st.markdown("#### Asosiy afzalliklar")
        for icon, title, body in [
            ("⚡", "Tezkor natija", "Laboratoriya kutishisiz, dala sharoitida bir necha daqiqada."),
            ("🎯", "Aniq me'yor", "Ekin, hosildorlik va tuproq sharoitiga moslashtirilgan hisob."),
            ("🔍", "Tushuntiriladigan AI", "Har bir tavsiya ortidagi sabab ochiq ko'rsatiladi."),
            ("💰", "Xarajat nazorati", "Ortiqcha o'g'it va keraksiz sarf-xarajatning oldini olish."),
        ]:
            info_card(title, body, icon)
            st.write("")

    st.markdown("### Qanday ishlaydi?")
    steps = st.columns(4, gap="medium")
    workflow = [
        ("Tuproq namunasini tayyorlash", "Daladan namuna olinadi va reagent bilan aralashtiriladi."),
        ("Kolorimetrik o'lchov", "Qurilma rang qiymatlarini (RGB), vaqt va haroratni qayd etadi."),
        ("AI tahlili", "Model o'simlik o'zlashtira oladigan fosforni va noaniqlikni baholaydi."),
        ("O'g'itlash tavsiyasi", "Ekin va dalaga mos o'g'it turi hamda miqdori hisoblanadi."),
    ]
    for index, (column, (title, body)) in enumerate(zip(steps, workflow, strict=True), start=1):
        with column:
            step_card(index, title, body)

    st.write("")
    action_left, action_right = st.columns([1, 3])
    with action_left:
        if st.button("Tahlilni boshlash", type="primary", width="stretch"):
            goto(PAGE_ANALYSIS)
    with action_right:
        if st.button("Demo ssenariylarni ko'rish", width="stretch"):
            goto(PAGE_DEMO)

    st.write("")
    if model_info:
        metrics = model_info["metrics"]
        columns = st.columns(4, gap="medium")
        with columns[0]:
            metric_card("Model", model_info["model_name"], sub="Avtomatik tanlangan")
        with columns[1]:
            metric_card("Aniqlik (R²)", f"{metrics['r2']:.3f}", sub="Ajratilgan test to'plamida", accent=ACCENT)
        with columns[2]:
            metric_card("RMSE", f"{metrics['rmse']:.2f}", "mg/kg", sub="O'rtacha kvadratik xato", accent=ACCENT)
        with columns[3]:
            metric_card("Namunalar", f"{model_info['n_samples']}", sub="O'quv to'plami hajmi")

    st.write("")
    disclaimer_box(DISCLAIMER_UZ)


def page_analysis(configs: dict[str, Any]) -> None:
    st.markdown("## 🧪 Yangi tahlil")
    st.caption(
        "Kolorimetrik o'lchov qiymatlarini kiriting yoki tayyor demo namunani tanlang. "
        "Fosforni baholashda faqat rang o'lchovi ishlatiladi; pH, EC va namlik o'g'itlash "
        "tavsiyasini shakllantirish uchun kerak."
    )

    crops = _crop_options(configs)
    products = _product_options(configs)
    demo_frame = get_demo_samples()

    mode = st.radio(
        "Ma'lumot kiritish usuli",
        ["Demo namuna", "Qo'lda kiritish"],
        horizontal=True,
        key="input_mode",
    )

    defaults: dict[str, Any] = {
        "red": 120.0,
        "green": 170.0,
        "blue": 215.0,
        "reaction_time_sec": 600.0,
        "sample_temperature_c": 24.0,
        "ph": 7.6,
        "ec_ds_m": 2.0,
        "moisture_pct": 20.0,
        "crop_key": next(iter(crops)),
        "field_area_ha": 10.0,
        "target_yield_t_ha": 3.5,
        "fertilizer_key": next(iter(products)),
    }
    sample_label = "Qo'lda kiritilgan namuna"

    if mode == "Demo namuna":
        labels = {row["demo_id"]: row["scenario_uz"] for _, row in demo_frame.iterrows()}
        chosen = st.selectbox(
            "Demo namunani tanlang",
            options=list(labels.keys()),
            format_func=lambda key: labels[key],
            key="demo_pick",
        )
        row = demo_frame.loc[demo_frame["demo_id"] == chosen].iloc[0]
        st.info(str(row.get("description_uz", "")))
        for key in defaults:
            if key in row.index and pd.notna(row[key]):
                defaults[key] = row[key]
        sample_label = str(row["scenario_uz"])
        preview_left, preview_right = st.columns([1, 2])
        with preview_left:
            color_swatch(float(row["red"]), float(row["green"]), float(row["blue"]))
        with preview_right:
            st.caption(
                "Demo namuna qiymatlari quyidagi shaklga avtomatik yuklandi — "
                "ularni o'zgartirib, natija qanday o'zgarishini sinab ko'rishingiz mumkin."
            )

    with st.form("analysis_form"):
        st.markdown("##### 1. Kolorimetrik o'lchov")
        color_columns = st.columns(3)
        with color_columns[0]:
            red = st.number_input(
                "Qizil (R)", min_value=0.0, max_value=255.0,
                value=float(defaults["red"]), step=1.0,
                help="Qurilma sensori qayd etgan qizil kanal qiymati (0-255).",
            )
        with color_columns[1]:
            green = st.number_input(
                "Yashil (G)", min_value=0.0, max_value=255.0,
                value=float(defaults["green"]), step=1.0,
                help="Yashil kanal qiymati (0-255).",
            )
        with color_columns[2]:
            blue = st.number_input(
                "Ko'k (B)", min_value=0.0, max_value=255.0,
                value=float(defaults["blue"]), step=1.0,
                help="Ko'k kanal qiymati (0-255). Fosfor ko'p bo'lsa eritma ko'kroq bo'ladi.",
            )

        condition_columns = st.columns(2)
        with condition_columns[0]:
            reaction_time = st.number_input(
                "Reaksiya vaqti (sekund)", min_value=10.0, max_value=1800.0,
                value=float(defaults["reaction_time_sec"]), step=10.0,
                help="Reagent qo'shilgandan keyin o'tgan vaqt. Odatda 8-12 daqiqa (480-720 sek).",
            )
        with condition_columns[1]:
            temperature = st.number_input(
                "Namuna harorati (°C)", min_value=0.0, max_value=60.0,
                value=float(defaults["sample_temperature_c"]), step=0.5,
                help="Harorat rang hosil bo'lish tezligiga ta'sir qiladi.",
            )

        st.markdown("##### 2. Tuproq holati")
        soil_columns = st.columns(3)
        with soil_columns[0]:
            ph = st.number_input(
                "pH", min_value=3.0, max_value=10.0,
                value=float(defaults["ph"]), step=0.1,
                help="Tuproq muhiti. 6.5-7.5 fosfor uchun eng qulay oraliq.",
            )
        with soil_columns[1]:
            ec = st.number_input(
                "Elektr o'tkazuvchanlik, EC (dS/m)", min_value=0.0, max_value=30.0,
                value=float(defaults["ec_ds_m"]), step=0.1,
                help="Sho'rlanish ko'rsatkichi. 4 dS/m dan yuqori qiymat sho'rlanishdan darak beradi.",
            )
        with soil_columns[2]:
            moisture = st.number_input(
                "Namlik (%)", min_value=0.0, max_value=100.0,
                value=float(defaults["moisture_pct"]), step=1.0,
                help="Tuproqdagi namlik ulushi. Quruq tuproqda o'g'it sekin o'zlashadi.",
            )

        st.markdown("##### 3. Ekin va dala")
        crop_columns = st.columns(2)
        crop_keys = list(crops.keys())
        with crop_columns[0]:
            crop_key = st.selectbox(
                "Ekin turi", options=crop_keys,
                index=crop_keys.index(defaults["crop_key"]) if defaults["crop_key"] in crop_keys else 0,
                format_func=lambda key: crops[key],
                help="Ekin turi fosforga bo'lgan talabni belgilaydi.",
            )
        product_keys = list(products.keys())
        with crop_columns[1]:
            fertilizer_key = st.selectbox(
                "O'g'it mahsuloti", options=product_keys,
                index=product_keys.index(defaults["fertilizer_key"])
                if defaults["fertilizer_key"] in product_keys else 0,
                format_func=lambda key: products[key],
                help="Mahsulotdagi P2O5 ulushi asosida o'g'it miqdori hisoblanadi.",
            )

        field_columns = st.columns(2)
        with field_columns[0]:
            area = st.number_input(
                "Dala maydoni (gektar)", min_value=0.01, max_value=100000.0,
                value=float(defaults["field_area_ha"]), step=0.5,
                help="Umumiy o'g'it miqdorini hisoblash uchun kerak.",
            )
        with field_columns[1]:
            target_yield = st.number_input(
                "Maqsadli hosildorlik (t/ga)", min_value=0.1, max_value=50.0,
                value=float(defaults["target_yield_t_ha"]), step=0.1,
                help="Rejalashtirilgan hosil. Qanchalik yuqori bo'lsa, oziq modda talabi ham shuncha yuqori.",
            )

        submitted = st.form_submit_button("AI tahlilini boshlash", type="primary", width="stretch")

    if submitted:
        payload = {
            "red": red, "green": green, "blue": blue,
            "reaction_time_sec": reaction_time, "sample_temperature_c": temperature,
            "ph": ph, "ec_ds_m": ec, "moisture_pct": moisture,
            "crop_key": crop_key, "field_area_ha": area,
            "target_yield_t_ha": target_yield, "fertilizer_key": fertilizer_key,
        }
        with st.spinner("AI tahlili bajarilmoqda..."):
            if run_analysis(payload, sample_label=sample_label):
                goto(PAGE_RESULTS)


def _render_results(configs: dict[str, Any], model_info: dict[str, Any] | None) -> None:
    result = st.session_state.get("result")
    if not result:
        st.info("Hozircha natija yo'q. Avval tahlil o'tkazing.")
        if st.button("Yangi tahlilga o'tish", type="primary"):
            goto(PAGE_ANALYSIS)
        return

    estimate = result["estimate"]
    recommendation = result["recommendation"]
    explanations = result["explanations"]
    measurements = result["measurements"]

    st.markdown(f"## 📊 Tahlil natijasi")
    st.caption(
        f"Namuna: **{result['sample_label']}** · Tahlil ID: `{result['analysis_id']}`"
    )

    if model_info and model_info.get("dataset_kind") == "demo":
        banner(f"⚠️ {DEMO_MODEL_BANNER_UZ}")

    if not estimate.within_calibration:
        banner(
            "⚠️ " + " ".join(estimate.notes_uz[:2])
            if estimate.notes_uz
            else "Namuna kalibrlash oralig'idan tashqarida."
        )

    # --- Asosiy ko'rsatkichlar ---
    top = st.columns(4, gap="medium")
    with top[0]:
        metric_card(
            "Baholangan Olsen-P",
            f"{estimate.olsen_p_mg_kg:.1f}",
            "mg/kg",
            sub=f"±{estimate.uncertainty_mg_kg:.1f} mg/kg noaniqlik",
            accent=PRIMARY,
        )
    with top[1]:
        metric_card(
            "Fosfor holati",
            estimate.status_label_uz,
            sub=f"{estimate.ci95_low:.1f}–{estimate.ci95_high:.1f} mg/kg (95%)",
            accent=estimate.status_color,
        )
    with top[2]:
        metric_card(
            "Model ishonchliligi",
            estimate.confidence_label_uz,
            sub="Konservativ baholash qoidalari",
            accent=ACCENT,
        )
    with top[3]:
        if recommendation.no_phosphorus_needed:
            metric_card(
                "Tavsiya",
                "Fosfor kerak emas",
                sub="Zaxira yetarli — bu mavsumda qo'shimcha fosfor tavsiya etilmaydi",
                accent="#1F6FB2",
            )
        else:
            metric_card(
                "Tavsiya etilgan o'g'it",
                recommendation.product_name_uz,
                sub=f"{recommendation.product_kg_ha_low:.0f}–{recommendation.product_kg_ha_high:.0f} kg/ga",
                accent=ACCENT,
            )

    st.write("")
    st.markdown("#### Fosfor darajasi shkalasi")
    st.plotly_chart(
        phosphorus_gauge(
            estimate.olsen_p_mg_kg, configs["thresholds"], estimate.uncertainty_mg_kg
        ),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(estimate.status_description_uz)

    # --- O'g'itlash tavsiyasi ---
    st.markdown("---")
    st.markdown("### 🌾 O'g'itlash tavsiyasi")

    if recommendation.no_phosphorus_needed:
        alert_card(
            "info",
            "Ushbu mavsumda fosforli o'g'it tavsiya etilmaydi",
            "Tuproqdagi fosfor zaxirasi yuqori baholandi. Qo'shimcha fosfor berish hosildorlikni "
            "oshirmasligi, ammo xarajat va ekologik yukni oshirishi mumkin. Keyingi mavsumda "
            "takroriy tahlil o'tkazish tavsiya etiladi.",
        )
    else:
        rec_columns = st.columns(4, gap="medium")
        with rec_columns[0]:
            metric_card(
                "Kerakli oziq modda",
                f"{recommendation.required_p2o5_low:.0f}–{recommendation.required_p2o5_high:.0f}",
                "kg P₂O₅/ga",
            )
        with rec_columns[1]:
            metric_card(
                f"{recommendation.product_name_uz} me'yori",
                f"{recommendation.product_kg_ha_low:.0f}–{recommendation.product_kg_ha_high:.0f}",
                "kg/ga",
                sub=f"Tarkibida {recommendation.p2o5_fraction * 100:.0f}% P₂O₅",
                accent=ACCENT,
            )
        with rec_columns[2]:
            metric_card(
                f"Dala uchun jami ({recommendation.field_area_ha:g} ga)",
                f"{recommendation.total_product_kg_low:.0f}–{recommendation.total_product_kg_high:.0f}",
                "kg",
                sub=f"≈ {recommendation.total_bags_low:.0f}–{recommendation.total_bags_high:.0f} qop (50 kg)",
                accent=ACCENT,
            )
        with rec_columns[3]:
            if recommendation.estimated_cost_uzs_low is not None:
                low = recommendation.estimated_cost_uzs_low / 1_000_000
                high = recommendation.estimated_cost_uzs_high / 1_000_000
                metric_card(
                    "Taxminiy xarajat",
                    f"{low:.1f}–{high:.1f}",
                    "mln",
                    sub=f"so'm · {recommendation.field_area_ha:g} ga uchun indikativ narx",
                )
            else:
                metric_card("Tavsiya ishonchliligi", recommendation.confidence_label_uz)

    timing_columns = st.columns(2, gap="large")
    with timing_columns[0]:
        info_card("Qo'llash muddati", recommendation.application_stage_uz, "🗓️")
    with timing_columns[1]:
        info_card("Qo'llash usuli", recommendation.application_method_uz, "🚜")

    st.write("")
    st.markdown(
        f"Tavsiya ishonchliligi: {confidence_badge(recommendation.confidence, recommendation.confidence_label_uz)}",
        unsafe_allow_html=True,
    )

    # --- Ogohlantirishlar ---
    if recommendation.warnings:
        st.markdown("### ⚠️ Agronomik ogohlantirishlar")
        for warning in recommendation.warnings:
            alert_card(warning.level, warning.title_uz, warning.message_uz)

    # --- Tushuntirish ---
    st.markdown("### 🔍 Nima uchun bu tavsiya berildi?")
    for index, reason in enumerate(explanations, start=1):
        st.markdown(
            f"""<div style="display:flex;gap:.7rem;margin-bottom:.55rem;align-items:flex-start;">
            <div style="min-width:26px;height:26px;border-radius:8px;background:{PRIMARY};color:#fff;
                        display:flex;align-items:center;justify-content:center;font-size:.8rem;
                        font-weight:700;">{index}</div>
            <div style="color:{INK};font-size:.95rem;line-height:1.55;padding-top:.15rem;">{reason}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # --- Texnik tafsilotlar ---
    with st.expander("🔬 Agronom uchun texnik tafsilotlar"):
        details = technical_details(estimate, recommendation, model_info)
        detail_columns = st.columns(2, gap="large")
        items = list(details.items())
        midpoint = (len(items) + 1) // 2
        for column, chunk in zip(detail_columns, [items[:midpoint], items[midpoint:]], strict=True):
            with column:
                for key, value in chunk:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;gap:1rem;"
                        f"padding:.35rem 0;border-bottom:1px solid #EDF2EE;'>"
                        f"<span style='color:{MUTED};font-size:.88rem;'>{key}</span>"
                        f"<b style='font-size:.88rem;'>{value}</b></div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("###### Hisoblash qadamlari")
        st.dataframe(
            pd.DataFrame(recommendation.calculation_steps).rename(
                columns={"step": "Qadam", "formula": "Formula", "result": "Natija"}
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("###### Kiritilgan o'lchovlar")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Parametr": "Qizil (R)", "Qiymat": f"{measurements['red']:.0f}"},
                    {"Parametr": "Yashil (G)", "Qiymat": f"{measurements['green']:.0f}"},
                    {"Parametr": "Ko'k (B)", "Qiymat": f"{measurements['blue']:.0f}"},
                    {"Parametr": "Reaksiya vaqti", "Qiymat": f"{measurements['reaction_time_sec']:.0f} sek"},
                    {"Parametr": "Harorat", "Qiymat": f"{measurements['sample_temperature_c']:.1f} °C"},
                    {"Parametr": "pH", "Qiymat": f"{measurements['ph']:.2f}"},
                    {"Parametr": "EC", "Qiymat": f"{measurements['ec_ds_m']:.2f} dS/m"},
                    {"Parametr": "Namlik", "Qiymat": f"{measurements['moisture_pct']:.1f} %"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        if recommendation.agronomic_notes_uz:
            st.markdown("###### Ekin bo'yicha agronomik izohlar")
            for note in recommendation.agronomic_notes_uz:
                st.markdown(f"- {note}")

    # --- PDF va qayta tahlil ---
    st.markdown("---")
    action_columns = st.columns([2, 1, 1])
    with action_columns[0]:
        try:
            pdf_bytes = build_pdf_report(
                estimate=estimate,
                recommendation=recommendation,
                measurements=measurements,
                explanations=explanations,
                model_info=model_info,
                analysis_id=result["analysis_id"],
                sample_label=result["sample_label"],
            )
            st.download_button(
                "📄 PDF hisobotni yuklab olish",
                data=pdf_bytes,
                file_name=f"AgroIQ_hisobot_{result['analysis_id']}.pdf",
                mime="application/pdf",
                width="stretch",
            )
        except ReportError as exc:
            st.error(str(exc))
    with action_columns[1]:
        if st.button("Yangi tahlil", width="stretch"):
            goto(PAGE_ANALYSIS)
    with action_columns[2]:
        if st.button("Demo rejimi", width="stretch"):
            goto(PAGE_DEMO)

    st.write("")
    disclaimer_box(f"{PILOT_RECOMMENDATION_NOTICE_UZ}<br/><br/>{DISCLAIMER_UZ}")


def page_results(configs: dict[str, Any], model_info: dict[str, Any] | None) -> None:
    _render_results(configs, model_info)


def page_model(model_info: dict[str, Any] | None) -> None:
    st.markdown("## 🤖 Model va validatsiya")
    if model_info is None:
        st.error(
            "Model yuklanmagan. `python scripts/train_model.py` buyrug'ini bajaring va sahifani yangilang."
        )
        return

    st.caption(
        "Ushbu sahifa mustaqil ekspertlar uchun: model qanday o'qitilgani, qanday sinovdan "
        "o'tgani va qanday cheklovlarga ega ekani ochiq ko'rsatilgan."
    )

    if model_info["dataset_kind"] == "demo":
        banner(f"⚠️ {DEMO_MODEL_BANNER_UZ}")

    metrics = model_info["metrics"]
    columns = st.columns(4, gap="medium")
    with columns[0]:
        metric_card("Tanlangan model", model_info["model_name"], sub=f"O'qitilgan: {model_info['trained_at']}")
    with columns[1]:
        metric_card("R²", f"{metrics['r2']:.3f}", sub="Aniqlanish koeffitsienti", accent=ACCENT)
    with columns[2]:
        metric_card("RMSE", f"{metrics['rmse']:.2f}", "mg/kg", sub="O'rtacha kvadratik xato", accent=ACCENT)
    with columns[3]:
        metric_card("MAE", f"{metrics['mae']:.2f}", "mg/kg", sub="O'rtacha absolyut xato", accent=ACCENT)

    st.write("")
    info_columns = st.columns(2, gap="large")
    with info_columns[0]:
        st.markdown("#### Ma'lumotlar to'plami")
        dataset_label = (
            "Demo (sintetik, takrorlanuvchi seed=42)"
            if model_info["dataset_kind"] == "demo"
            else "Real laboratoriya ma'lumoti"
        )
        st.markdown(
            f"""
            - **Turi:** {dataset_label}
            - **Namunalar soni:** {model_info['n_samples']} ta
            - **O'quv / test:** {model_info['n_train']} / {model_info['n_test']}
            - **Bo'linish usuli:** {model_info['split_strategy']}
            """
        )
        if model_info.get("cv_metrics"):
            cv = model_info["cv_metrics"]
            st.markdown(
                f"- **GroupKFold cross-validation:** R² = {cv['r2']:.3f}, "
                f"RMSE = {cv['rmse']:.2f} mg/kg ({int(cv['n_splits'])} fold)"
            )
        st.caption(
            "Dala bo'yicha guruhlangan bo'linish (GroupShuffleSplit) tanlandi, chunki bitta "
            "daladan olingan namunalar o'zaro bog'liq. Bu baholashni real sharoitga yaqinlashtiradi."
        )

    with info_columns[1]:
        st.markdown("#### Modelning kirish o'zgaruvchilari")
        st.markdown(
            "Model **faqat optik va o'lchov sharoiti** ma'lumotlaridan foydalanadi:"
        )
        pills = "".join(
            f'<span class="agro-pill">{name}</span>'
            for name in ["RGB kanallar", "Normallashgan RGB", "RGB nisbatlari", "Hue / Saturation / Value",
                         "Absorbsiya indekslari", "Rang indekslari", "Reaksiya vaqti", "Harorat"]
        )
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown(
            f"Jami **{len(model_info['feature_columns'])}** ta xususiyat hosil qilinadi."
        )
        alert_card(
            "info",
            "Ilmiy chegaralanish",
            "pH, EC va namlik fosfor konsentratsiyasini bashorat qilishda <b>ishlatilmaydi</b>. Ular "
            "faqat o'g'itlash tavsiyasi bosqichida — fosforning o'zlashtirilishi va qo'llash "
            "sharoitini baholash uchun qo'llaniladi.",
        )

    st.markdown("---")
    chart_columns = st.columns(2, gap="large")
    with chart_columns[0]:
        st.markdown("#### Model nomzodlarini taqqoslash")
        st.plotly_chart(
            model_comparison_chart(model_info["candidates"], model_info["model_name"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.dataframe(
            pd.DataFrame(model_info["candidates"]).rename(
                columns={"name": "Model", "r2": "R²", "rmse": "RMSE", "mae": "MAE"}
            ).round(4),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Uchta nomzod bir xil test to'plamida taqqoslanadi. Random Forest asosiy chiziqli "
            "bo'lmagan nomzod sifatida har doim sinovdan o'tkaziladi; natijasi eng yaxshi modelga "
            "yaqin bo'lsa (0.02 R² ichida), daraxtlar orqali noaniqlikni baholash imkoniyati "
            "uchun unga ustunlik beriladi."
        )

    with chart_columns[1]:
        importances = get_feature_importances()
        if importances is not None:
            st.markdown("#### Xususiyatlarning hissasi")
            st.plotly_chart(
                feature_importance_chart(importances),
                width="stretch",
                config={"displayModeBar": False},
            )
        elif model_info.get("test_actual"):
            st.markdown("#### Laboratoriya va model qiymatlari")
            st.plotly_chart(
                prediction_scatter(
                    pd.Series(model_info["test_actual"]), pd.Series(model_info["test_predicted"])
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption(
                "Tanlangan model daraxtlarga asoslanmagani uchun xususiyat muhimligi diagrammasi "
                "o'rniga test to'plamidagi bashoratlar taqqoslanmoqda."
            )

    st.markdown("---")
    range_columns = st.columns(2, gap="large")
    with range_columns[0]:
        st.markdown("#### Kalibrlash oralig'i")
        st.caption(
            "Model quyidagi oraliqlarda o'qitilgan. Bu chegaradan tashqaridagi o'lchovlar uchun "
            "natija ishonchsiz deb belgilanadi va laboratoriya tasdig'i so'raladi."
        )
        calibration_rows = [
            {
                "Parametr": name,
                "Minimal": f"{bounds['min']:.1f}",
                "Maksimal": f"{bounds['max']:.1f}",
            }
            for name, bounds in model_info["calibration"].items()
        ]
        st.dataframe(pd.DataFrame(calibration_rows), width="stretch", hide_index=True)
        target = model_info["target_range"]
        st.markdown(
            f"**Maqsad (Olsen-P) oralig'i:** {target['min']:.1f} – {target['max']:.1f} mg/kg "
            f"(o'rtacha {target['mean']:.1f} mg/kg)"
        )

    with range_columns[1]:
        st.markdown("#### Noaniqlikni baholash usuli")
        st.markdown(
            f"""
            Har bir natija uchun standart noaniqlik ikki manbadan birlashtiriladi:

            1. **Model kelishmovchiligi** — Random Forest tanlanganda alohida daraxtlar
               bashoratlarining standart chetlanishi.
            2. **Qoldiq tarqoqlik** — test to'plamidagi xatolikning bashorat kattaligiga
               bog'liq (geteroskedastik) bahosi:
               σ ≈ {model_info['residual_intercept']:.2f} + {model_info['residual_slope']:.3f} × bashorat.

            Yakuniy noaniqlik: σ_umumiy = √(σ_model² + σ_qoldiq²).
            """
        )
        st.markdown("#### Ishonch darajasi qoidalari")
        st.markdown(
            """
            - **Yuqori** — nisbiy noaniqlik 15% dan kam va namuna kalibrlash oralig'ida.
            - **O'rtacha** — nisbiy noaniqlik 30% dan kam.
            - **Past** — boshqa barcha holatlar; kalibrlash oralig'idan chiqqan har qanday
              namuna avtomatik ravishda "past" deb belgilanadi.
            """
        )

    st.markdown("---")
    st.markdown("#### Modelning cheklovlari")
    limit_columns = st.columns(2, gap="large")
    limitations = [
        ("Laboratoriya o'rnini bosmaydi",
         "AgroIQ tezkor dala bahosini beradi. Rasmiy agrokimyoviy xulosa uchun sertifikatlangan "
         "laboratoriya tahlili talab qilinadi."),
        ("Hududiy kalibrlash zarur",
         "Model tuproq turi, reagent partiyasi va qurilma optikasiga sezgir. Har bir hudud uchun "
         "mahalliy namunalar bilan qayta kalibrlash kerak."),
        ("Faqat fosfor",
         "Hozirgi versiya faqat o'simlik o'zlashtira oladigan fosforni baholaydi. Azot, kaliy va "
         "mikroelementlar keyingi modullarda rejalashtirilgan."),
        ("Dala validatsiyasi yakunlanmagan",
         "O'g'itlash me'yorlari agronomik adabiyot va amaliyotga asoslangan pilot qiymatlar. "
         "Ular hosildorlik bo'yicha dala tajribalari bilan tasdiqlanishi kerak."),
    ]
    for index, (title, body) in enumerate(limitations):
        with limit_columns[index % 2]:
            info_card(title, body, "•")
            st.write("")

    if model_info.get("warnings"):
        with st.expander("O'qitish jarayonidagi ogohlantirishlar"):
            for warning in model_info["warnings"]:
                st.markdown(f"- {warning}")


def page_demo(configs: dict[str, Any]) -> None:
    st.markdown("## 🎬 Demo rejimi")
    st.caption(
        "Uchta tayyorlangan ssenariy platformaning turli sharoitlarda qanday javob berishini "
        "ko'rsatadi. Hech qanday fayl yuklash talab etilmaydi."
    )
    banner(f"⚠️ {DEMO_MODEL_BANNER_UZ}")

    demo_frame = get_demo_samples()
    scenario_hint = {row["demo_id"]: row for _, row in demo_frame.iterrows()}
    order = list(scenario_hint.keys())

    if "demo_index" not in st.session_state:
        st.session_state["demo_index"] = 0

    columns = st.columns(len(order), gap="medium")
    for index, (column, demo_id) in enumerate(zip(columns, order, strict=True)):
        row = scenario_hint[demo_id]
        with column:
            info_card(str(row["scenario_uz"]), str(row["description_uz"]), "🧪")
            st.write("")
            if st.button("Shu ssenariyni ishga tushirish", key=f"run_{demo_id}", width="stretch"):
                st.session_state["demo_index"] = index
                payload = {
                    "red": float(row["red"]), "green": float(row["green"]), "blue": float(row["blue"]),
                    "reaction_time_sec": float(row["reaction_time_sec"]),
                    "sample_temperature_c": float(row["sample_temperature_c"]),
                    "ph": float(row["ph"]), "ec_ds_m": float(row["ec_ds_m"]),
                    "moisture_pct": float(row["moisture_pct"]),
                    "crop_key": str(row["crop_key"]),
                    "field_area_ha": float(row["field_area_ha"]),
                    "target_yield_t_ha": float(row["target_yield_t_ha"]),
                    "fertilizer_key": str(row["fertilizer_key"]),
                }
                with st.spinner("AI tahlili bajarilmoqda..."):
                    if run_analysis(payload, sample_label=str(row["scenario_uz"])):
                        goto(PAGE_RESULTS)

    st.write("")
    next_columns = st.columns([1, 3])
    with next_columns[0]:
        if st.button("Keyingi demo namuna →", type="primary", width="stretch"):
            next_index = (st.session_state["demo_index"] + 1) % len(order)
            st.session_state["demo_index"] = next_index
            row = scenario_hint[order[next_index]]
            payload = {
                "red": float(row["red"]), "green": float(row["green"]), "blue": float(row["blue"]),
                "reaction_time_sec": float(row["reaction_time_sec"]),
                "sample_temperature_c": float(row["sample_temperature_c"]),
                "ph": float(row["ph"]), "ec_ds_m": float(row["ec_ds_m"]),
                "moisture_pct": float(row["moisture_pct"]),
                "crop_key": str(row["crop_key"]),
                "field_area_ha": float(row["field_area_ha"]),
                "target_yield_t_ha": float(row["target_yield_t_ha"]),
                "fertilizer_key": str(row["fertilizer_key"]),
            }
            with st.spinner("AI tahlili bajarilmoqda..."):
                if run_analysis(payload, sample_label=str(row["scenario_uz"])):
                    goto(PAGE_RESULTS)

    st.markdown("---")
    st.markdown("#### Ssenariylar parametrlari")
    display = demo_frame[
        [
            "scenario_uz", "red", "green", "blue", "reaction_time_sec",
            "sample_temperature_c", "ph", "ec_ds_m", "moisture_pct",
            "crop_key", "field_area_ha", "target_yield_t_ha",
        ]
    ].rename(
        columns={
            "scenario_uz": "Ssenariy", "red": "R", "green": "G", "blue": "B",
            "reaction_time_sec": "Vaqt (sek)", "sample_temperature_c": "T (°C)",
            "ph": "pH", "ec_ds_m": "EC", "moisture_pct": "Namlik (%)",
            "crop_key": "Ekin", "field_area_ha": "Maydon (ga)",
            "target_yield_t_ha": "Hosil (t/ga)",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)


def page_about() -> None:
    st.markdown("## 💡 Loyiha haqida")

    columns = st.columns(2, gap="large")
    with columns[0]:
        st.markdown("### Muammo")
        st.markdown(
            "Fermerlarda dala darajasida o'simlik o'zlashtira oladigan fosforni tez va arzon "
            "o'lchash imkoniyati yo'q. Laboratoriya tahlili qimmat, uzoq va ko'p hollarda "
            "mavsumiy qarorlar uchun kech keladi. Natijada o'g'it me'yori taxminan belgilanadi."
        )
        st.markdown("### Yechim")
        st.markdown(
            "Portativ kolorimetrik tahlil qurilmasi va AI asosidagi agronomik tavsiya platformasi. "
            "Sun'iy intellekt optik kalibrlash va fosforni baholash uchun ishlatiladi; o'g'it "
            "me'yori esa shaffof, tekshirilishi mumkin bo'lgan agronomik qoidalar orqali hisoblanadi."
        )
        st.markdown("### Maqsadli mijozlar")
        for item in [
            "Fermer xo'jaliklari",
            "Paxta va g'alla klasterlari",
            "Agrolaboratoriyalar",
            "Agronomik konsalting xizmatlari",
            "O'g'it distribyutorlari",
            "Sayyor tuproq tahlili xizmatlari",
        ]:
            st.markdown(f"- {item}")

    with columns[1]:
        st.markdown("### Biznes modeli")
        for icon, title, body in [
            ("📦", "Qurilma savdosi", "Portativ AgroIQ o'qish qurilmasini sotish."),
            ("🧪", "Test kartrijlari", "Takroriy sotiladigan reagent kartrijlari — barqaror daromad."),
            ("🚐", "Xizmat sifatida tahlil", "Sayyor tuproq tahlili xizmati (test uchun to'lov)."),
            ("☁️", "Platforma obunasi", "AI tahlil platformasiga yillik obuna."),
            ("🏢", "B2B xizmatlar", "Klasterlar uchun integratsiya va ma'lumot xizmatlari."),
        ]:
            info_card(title, body, icon)
            st.write("")

    st.markdown("---")
    st.markdown("### Kelgusi modullar")
    roadmap = st.columns(3, gap="medium")
    modules = [
        ("Kaliy (K₂O)", "Kaliy bo'yicha kolorimetrik modul."),
        ("Nitrat azot (NO₃-N)", "Azot bilan ta'minlanganlikni tezkor baholash."),
        ("Sug'orish suvi tahlili", "Suvning sho'rligi va tarkibini baholash."),
        ("Dron va sun'iy yo'ldosh", "Masofaviy zondlash ma'lumotlari bilan integratsiya."),
        ("O'zgaruvchan me'yorli xaritalar", "Dala ichidagi farqlarga mos o'g'itlash xaritalari."),
        ("Mobil ilova", "Oflayn rejimda ishlaydigan fermer ilovasi."),
    ]
    for index, (title, body) in enumerate(modules):
        with roadmap[index % 3]:
            info_card(title, body, "→")
            st.write("")

    st.markdown("---")
    st.markdown("### Jamoa")
    team = st.columns(3, gap="medium")
    roles = [
        ("Loyiha rahbari", "[Ism Familiya]", "Mahsulot strategiyasi va biznes rivojlantirish"),
        ("AI / ML muhandisi", "[Ism Familiya]", "Model, kalibrlash va ma'lumotlar tahlili"),
        ("Agronom-maslahatchi", "[Ism Familiya]", "Agronomik qoidalar va dala validatsiyasi"),
    ]
    for column, (role, name, description) in zip(team, roles, strict=True):
        with column:
            info_card(f"{name}", f"<b>{role}</b><br/>{description}", "👤")

    st.write("")
    disclaimer_box(DISCLAIMER_UZ)


# ---------------------------------------------------------------------------
# Asosiy oqim
# ---------------------------------------------------------------------------


def main() -> None:
    inject_css()
    apply_pending_navigation()

    # Konfiguratsiyalar — ularsiz ilova ishlay olmaydi.
    try:
        configs = get_configs()
    except ConfigError as exc:
        st.error(f"Konfiguratsiya xatosi: {exc}")
        st.stop()
        return

    # Model — bo'lmasa ilova ishlaydi, lekin tahlil qilmaydi.
    model_info: dict[str, Any] | None = None
    model_error: str | None = None
    try:
        model_info = get_model_info()
    except ModelNotAvailableError as exc:
        model_error = str(exc)
    except Exception:  # noqa: BLE001 - foydalanuvchiga traceback ko'rsatilmaydi
        model_error = (
            "Model faylini yuklashda kutilmagan xatolik. "
            "`python scripts/train_model.py` buyrug'ini bajaring."
        )

    with st.sidebar:
        render_logo(compact=True)
        st.markdown(
            f"<div style='color:{MUTED};font-size:.82rem;margin-bottom:1rem;'>"
            f"Tuproq diagnostikasi va aqlli o'g'itlash platformasi</div>",
            unsafe_allow_html=True,
        )
        st.radio(
            "Bo'limlar",
            PAGES,
            key="nav",
            format_func=lambda page: f"{PAGE_ICONS[page]}  {page}",
            label_visibility="collapsed",
        )
        st.markdown("---")
        if model_info and model_info["dataset_kind"] == "demo":
            st.markdown(
                "<div style='background:#FFF8E1;border:1px solid #EBD08A;border-radius:10px;"
                "padding:.6rem .7rem;font-size:.76rem;color:#6B4B00;line-height:1.45;'>"
                "<b>Demo model</b><br/>Sintetik ma'lumotda o'qitilgan. Dala uchun kalibrlash zarur."
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='color:{MUTED};font-size:.72rem;margin-top:1rem;'>"
            f"{__app_name__} MVP v{__version__}<br/>President AI Award prototipi</div>",
            unsafe_allow_html=True,
        )

    if model_error:
        st.error(model_error)
        st.info(
            "Modelni yaratish uchun terminalda quyidagi buyruqni bajaring:\n\n"
            "```\npython scripts/train_model.py\n```"
        )

    page = st.session_state.get("nav", PAGE_HOME)

    if page == PAGE_HOME:
        page_home(configs, model_info)
    elif page == PAGE_ANALYSIS:
        if model_error:
            st.warning("Model tayyor bo'lmaguncha tahlil o'tkazib bo'lmaydi.")
        else:
            page_analysis(configs)
    elif page == PAGE_RESULTS:
        page_results(configs, model_info)
    elif page == PAGE_MODEL:
        page_model(model_info)
    elif page == PAGE_DEMO:
        if model_error:
            st.warning("Model tayyor bo'lmaguncha demo rejimini ishga tushirib bo'lmaydi.")
        else:
            page_demo(configs)
    elif page == PAGE_ABOUT:
        page_about()


if __name__ == "__main__":
    main()
