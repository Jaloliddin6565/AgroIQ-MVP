"""AgroIQ — integratsiyalashgan tuproq intellekti va aniq o'g'itlash platformasi.

v0.2.0: AgroIQ kolorimetrik fosfor analizatori + universal ko'p parametrli tuproq
sensori + tushuntiriladigan ma'lumotlarni birlashtirish qatlami.

Ishga tushirish:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, time as dtime, timezone
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
from src.data_fusion import fuse  # noqa: E402
from src.data_validation import (  # noqa: E402
    ConfigError,
    load_crop_profiles,
    load_device_profiles,
    load_fertilizer_products,
    load_nutrient_calibration,
    load_recommendation_rules,
    load_sensor_thresholds,
    load_thresholds,
)
from src.demo_scenarios import SCENARIOS, build_scenario  # noqa: E402
from src.device_integration import (  # noqa: E402
    api_defaults,
    example_sensor_file,
    fetch_from_gateway,
    get_device_profile,
    parse_uploaded_sensor_file,
    check_connection,
)
from src.explanations import build_explanations, technical_details  # noqa: E402
from src.model_inference import (  # noqa: E402
    ModelNotAvailableError,
    load_artifact,
    model_summary,
)
from src.model_training import feature_importances  # noqa: E402
from src.recommendation_engine import build_recommendation  # noqa: E402
from src.report_generator import ReportError, build_pdf_report, make_analysis_id  # noqa: E402
from src.sensor_schemas import (  # noqa: E402
    DATA_SOURCE_LABELS_UZ,
    AnalyzerReading,
    DataSource,
    SensorReading,
)
from src.soil_interpretation import condition_chart_data  # noqa: E402
from src.ui_components import (  # noqa: E402
    ACCENT,
    INK,
    MUTED,
    PRIMARY,
    alert_card,
    architecture_row,
    banner,
    color_swatch,
    confidence_badge,
    device_card,
    diagnostic_index_gauge,
    disclaimer_box,
    feature_importance_chart,
    info_card,
    inject_css,
    metric_card,
    model_comparison_chart,
    nutrient_card,
    phosphorus_comparison_chart,
    phosphorus_gauge,
    prediction_scatter,
    render_logo,
    section_header,
    soil_condition_chart,
    source_badges,
    status_pill,
    step_card,
    top_navigation,
)

# ---------------------------------------------------------------------------
# Bo'limlar
# ---------------------------------------------------------------------------

PAGE_HOME = "Bosh sahifa"
PAGE_ANALYSIS = "Yangi tahlil"
PAGE_RESULTS = "Natijalar"
PAGE_DEVICES = "Qurilmalar"
PAGE_MODEL = "Model va validatsiya"
PAGE_DEMO = "Demo rejimi"
PAGE_ABOUT = "Loyiha haqida"

PAGES = [
    PAGE_HOME,
    PAGE_ANALYSIS,
    PAGE_RESULTS,
    PAGE_DEVICES,
    PAGE_MODEL,
    PAGE_DEMO,
    PAGE_ABOUT,
]

st.set_page_config(
    page_title="AgroIQ — tuproq intellekti platformasi",
    page_icon="🌱",
    layout="wide",
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
        "sensor": load_sensor_thresholds(),
        "devices": load_device_profiles(),
        "calibration": load_nutrient_calibration(),
        "rules": load_recommendation_rules(),
    }


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
    """Boshqa bo'limga o'tish (widget yaratilishidan oldin qo'llaniladi)."""

    st.session_state["_pending_nav"] = page
    st.rerun()


def apply_pending_navigation() -> None:
    pending = st.session_state.pop("_pending_nav", None)
    if pending in PAGES:
        st.session_state["nav"] = pending


def _crop_options(configs: dict[str, Any]) -> dict[str, str]:
    return {crop["key"]: crop["name_uz"] for crop in configs["crops"]["crops"]}


def _product_options(configs: dict[str, Any]) -> dict[str, str]:
    return {item["key"]: item["name_uz"] for item in configs["products"]["products"]}


def _stage_options(configs: dict[str, Any], crop_key: str) -> dict[str, str]:
    stages = configs["rules"].get("growth_stages", {}).get(crop_key, [])
    return {stage["key"]: stage["label_uz"] for stage in stages}


def _rule_options(configs: dict[str, Any], section: str) -> dict[str, str]:
    return {item["key"]: item["label_uz"] for item in configs["rules"].get(section, [])}


# ---------------------------------------------------------------------------
# Tahlilni bajarish
# ---------------------------------------------------------------------------


def run_analysis(
    analyzer_data: dict[str, Any],
    sensor_data: dict[str, Any] | None,
    field_data: dict[str, Any],
    source: DataSource,
    sample_label: str = "Namuna",
) -> bool:
    """Kirishni tekshiradi, birlashtiradi va natijani sessiyaga saqlaydi."""

    configs = get_configs()

    try:
        analyzer = AnalyzerReading(**analyzer_data, source=source)
    except Exception as exc:  # noqa: BLE001 - pydantic xatolari
        st.error("Analizator ma'lumotlarida xatolik bor. Qiymatlarni tekshiring.")
        with st.expander("Tafsilot"):
            st.write(str(exc)[:600])
        return False

    sensor: SensorReading | None = None
    if sensor_data:
        cleaned = {key: value for key, value in sensor_data.items() if value is not None}
        if cleaned:
            try:
                sensor = SensorReading(**cleaned, source=source)
            except Exception as exc:  # noqa: BLE001
                st.error("Sensor ma'lumotlarida xatolik bor. Qiymatlarni tekshiring.")
                with st.expander("Tafsilot"):
                    st.write(str(exc)[:600])
                return False

    try:
        artifact = get_artifact()
        fusion = fuse(
            analyzer=analyzer,
            sensor=sensor,
            artifact=artifact,
            manual_ph=field_data.get("manual_ph"),
            manual_ec=field_data.get("manual_ec"),
            manual_moisture=field_data.get("manual_moisture"),
            sensor_thresholds=configs["sensor"],
            phosphorus_thresholds=configs["thresholds"],
            rules=configs["rules"],
        )
    except ModelNotAvailableError as exc:
        st.error(str(exc))
        return False
    except ConfigError as exc:
        st.error(f"Konfiguratsiya xatosi: {exc}")
        return False

    if fusion.estimate is None:
        st.error("Fosforni baholab bo'lmadi — analizator ma'lumoti yetarli emas.")
        return False

    if fusion.ph is None or fusion.ec_ds_m is None or fusion.soil_moisture_percent is None:
        st.error(
            "Tavsiya uchun pH, EC va namlik qiymatlari kerak. Ularni sensor orqali yoki "
            "qo'lda kiriting."
        )
        return False

    try:
        recommendation = build_recommendation(
            olsen_p_mg_kg=fusion.estimate.olsen_p_mg_kg,
            status_key=fusion.estimate.status_key,
            status_label_uz=fusion.estimate.status_label_uz,
            ph=fusion.ph,
            ec_ds_m=fusion.ec_ds_m,
            moisture_pct=fusion.soil_moisture_percent,
            crop_key=field_data["crop_key"],
            field_area_ha=field_data["field_area_ha"],
            target_yield_t_ha=field_data["target_yield_t_ha"],
            fertilizer_key=field_data["fertilizer_key"],
            model_confidence=fusion.confidence,
            within_calibration=fusion.estimate.within_calibration,
            profiles=configs["crops"],
            products=configs["products"],
            rules=configs["rules"],
            growth_stage_key=field_data.get("growth_stage_key"),
            previous_fertilizer_key=field_data.get("previous_fertilizer_key"),
            soil_texture_key=field_data.get("soil_texture_key"),
            irrigation_method_key=field_data.get("irrigation_method_key"),
            soil_temperature_c=fusion.soil_temperature_c,
            nitrogen=fusion.nitrogen,
            potassium=fusion.potassium,
            fertilizer_price_uzs_per_kg=field_data.get("fertilizer_price"),
        )
        explanations = build_explanations(
            fusion.estimate,
            recommendation,
            fusion.ph,
            fusion.ec_ds_m,
            fusion.soil_moisture_percent,
        )
    except ConfigError as exc:
        st.error(f"Konfiguratsiya xatosi: {exc}")
        return False
    except (ValueError, KeyError) as exc:
        st.error(f"Tavsiyani hisoblab bo'lmadi: {exc}")
        return False

    st.session_state["result"] = {
        "fusion": fusion,
        "estimate": fusion.estimate,
        "recommendation": recommendation,
        "explanations": explanations,
        "analyzer": analyzer,
        "sensor": sensor,
        "field": field_data,
        "sample_label": sample_label,
        "analysis_id": make_analysis_id(),
    }
    return True


def _run_scenario(demo_id: str) -> bool:
    """Demo ssenariyni to'liq bajaradi."""

    scenario = build_scenario(demo_id)
    return run_analysis(
        analyzer_data=scenario["analyzer"],
        sensor_data=scenario["sensor"],
        field_data=scenario["field"],
        source=DataSource.DEMO,
        sample_label=scenario["title_uz"],
    )


# ---------------------------------------------------------------------------
# A. BOSH SAHIFA
# ---------------------------------------------------------------------------


def page_home(configs: dict[str, Any], model_info: dict[str, Any] | None) -> None:
    st.markdown(
        """
        <div class="agro-hero">
            <span class="eyebrow">AI · Ko'p sensorli diagnostika · AgriTech</span>
            <h1>AgroIQ — tuproq diagnostikasi va aniq o'g'itlash uchun
            integratsiyalashgan AI platformasi</h1>
            <p>Portativ fosfor analizatori, universal tuproq sensori va sun'iy intellekt
            yordamida dalangizning oziqa holatini baholang hamda ekinga mos o'g'itlash
            tavsiyasini oling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Qanday ishlaydi?", "To'rt bosqichli ish oqimi")
    steps = st.columns(4, gap="medium")
    workflow = [
        (
            "Universal sensor",
            "pH, EC, namlik, harorat va NPK skrining indikatorlari dala sharoitida o'lchanadi.",
        ),
        (
            "AgroIQ analizatori",
            "O'simlik o'zlashtira oladigan fosfor kolorimetrik usulda aniqlanadi.",
        ),
        (
            "AI birlashtirish",
            "Ma'lumotlar tekshiriladi, talqin qilinadi va ishonchlilik darajasi baholanadi.",
        ),
        (
            "Aqlli tavsiya",
            "O'g'it turi, miqdori, qo'llash muddati va usuli hisoblab beriladi.",
        ),
    ]
    for index, (column, (title, body)) in enumerate(zip(steps, workflow, strict=True), start=1):
        with column:
            step_card(index, title, body)

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        section_header("Muammo")
        st.markdown(
            "Fermerlar o'g'itlash bo'yicha qarorlarni ko'pincha **umumiy tavsiyalar** asosida, "
            "dala darajasidagi tezkor ma'lumotsiz qabul qiladi. Laboratoriya tahlili qimmat, "
            "sekin yoki uzoq bo'lishi mumkin. Umumiy tuproq sensorlari esa o'simlik o'zlashtira "
            "oladigan fosforni ishonchli o'lchay olmaydi."
        )
        section_header("Yechim")
        st.markdown(
            "AgroIQ **maxsus kolorimetrik fosfor analizatorini**, **ko'p parametrli tuproq "
            "sensori ma'lumotlarini** va **tushuntiriladigan tavsiya dvigatelini** birlashtiradi. "
            "Har bir tavsiya ortidagi sabab, ishonchlilik darajasi va cheklovlar ochiq ko'rsatiladi."
        )
    with right:
        section_header("Asosiy afzalliklar")
        for title, body in [
            ("Ko'p parametrli diagnostika", "Fosfor, pH, EC, namlik, harorat va NPK skrining — bitta ish oqimida."),
            ("Tezkor dala tahlili", "Laboratoriya kutishisiz, dala sharoitida bir necha daqiqada."),
            ("Aniq va tushuntiriladigan tavsiya", "Har bir raqam oshkora hisob-kitob qadamlari bilan asoslanadi."),
            ("Xarajat va ekologik yukni kamaytirish", "Ortiqcha o'g'it berishning oldini oladi."),
        ]:
            info_card(title, body)
            st.write("")

    st.write("")
    actions = st.columns([1, 1, 2])
    with actions[0]:
        if st.button("Tahlilni boshlash", type="primary", width="stretch"):
            goto(PAGE_ANALYSIS)
    with actions[1]:
        if st.button("Demo ssenariylar", width="stretch"):
            goto(PAGE_DEMO)

    st.write("")
    if model_info:
        metrics = model_info["metrics"]
        columns = st.columns(4, gap="medium")
        with columns[0]:
            metric_card("Fosfor modeli", model_info["model_name"], sub="Avtomatik tanlangan")
        with columns[1]:
            metric_card("Aniqlik (R²)", f"{metrics['r2']:.3f}", sub="Ajratilgan test to'plamida", accent=ACCENT)
        with columns[2]:
            metric_card("RMSE", f"{metrics['rmse']:.2f}", "mg/kg", sub="O'rtacha kvadratik xato", accent=ACCENT)
        with columns[3]:
            metric_card("Qurilmalar", "2", sub="Analizator + universal sensor", accent=ACCENT)

    st.write("")
    disclaimer_box(DISCLAIMER_UZ)


# ---------------------------------------------------------------------------
# B. YANGI TAHLIL
# ---------------------------------------------------------------------------


def page_analysis(configs: dict[str, Any]) -> None:
    st.markdown("## Yangi tahlil")
    st.caption(
        "Fosforni baholashda faqat AgroIQ kolorimetrik o'lchovi ishlatiladi. "
        "Universal sensor qiymatlari tuproq sharoitini baholash va tavsiyani "
        "aniqlashtirish uchun kerak."
    )

    crops = _crop_options(configs)
    products = _product_options(configs)

    # ---------------- 1-BO'LIM: MA'LUMOT MANBASI ----------------
    section_header("1. Ma'lumot manbasi", "Ma'lumotni qanday kiritasiz?")
    mode = st.radio(
        "Manba",
        ["Demo ssenariy", "Qo'lda kiritish", "Sensor API / gateway", "JSON yoki CSV yuklash"],
        horizontal=True,
        key="input_mode",
        label_visibility="collapsed",
    )

    defaults: dict[str, Any] = {
        "red": 120.0, "green": 170.0, "blue": 215.0,
        "reaction_time_sec": 600.0, "analyzer_temperature_c": 24.0,
        "analyzer_device_id": "AGROIQ-P-001", "cartridge_batch_id": "",
        "nitrogen_indicator": 45.0, "phosphorus_indicator": 18.0,
        "potassium_indicator": 170.0, "ph": 7.6, "ec_ds_m": 2.0,
        "soil_temperature_c": 22.0, "soil_moisture_percent": 22.0,
        "sensor_device_id": "SOIL-001",
    }
    sample_label = "Qo'lda kiritilgan namuna"
    source = DataSource.MANUAL

    if mode == "Demo ssenariy":
        source = DataSource.DEMO
        labels = {item["demo_id"]: item["title_uz"] for item in SCENARIOS}
        chosen = st.selectbox(
            "Ssenariy", list(labels), format_func=lambda key: labels[key], key="demo_pick"
        )
        scenario = build_scenario(chosen)
        st.info(scenario["summary_uz"])
        defaults.update(scenario["analyzer"])
        defaults.update(scenario["sensor"])
        defaults["cartridge_batch_id"] = scenario["analyzer"].get("cartridge_batch_id") or ""
        sample_label = scenario["title_uz"]
        st.session_state["prefill_field"] = scenario["field"]

    elif mode == "Sensor API / gateway":
        source = DataSource.API
        st.caption(
            "Gateway REST API orqali oxirgi o'lchovni yuklang. Apparat bo'lmasa, "
            "`mock://` manzilidan foydalaning — hech qanday tarmoq so'rovi yuborilmaydi."
        )
        api_columns = st.columns([3, 1])
        with api_columns[0]:
            endpoint = st.text_input(
                "API manzili",
                value=st.session_state.get(
                    "api_endpoint", api_defaults(configs["devices"])["mock_endpoint"]
                ),
                key="api_endpoint",
                help="Masalan: http://192.168.1.50:8000/api/v1/readings/latest",
            )
        with api_columns[1]:
            timeout = st.number_input("Timeout (sek)", 1.0, 30.0, 5.0, 1.0, key="api_timeout")
        token = st.text_input(
            "API token (ixtiyoriy)",
            type="password",
            key="api_token",
            help="Token faqat sessiya xotirasida saqlanadi va hech qachon faylga yozilmaydi.",
        )
        if st.button("Oxirgi ma'lumotni olish", type="primary"):
            result = fetch_from_gateway(endpoint, token=token or None, timeout=float(timeout))
            if result.ok and result.reading:
                st.session_state["api_reading"] = result.reading
                st.success(result.message_uz)
            else:
                st.session_state.pop("api_reading", None)
                st.error(result.message_uz)
                st.info(
                    "Ma'lumotni qo'lda kiritishingiz yoki JSON/CSV fayl yuklashingiz mumkin."
                )

        reading = st.session_state.get("api_reading")
        if reading is not None:
            for key in (
                "nitrogen_indicator", "phosphorus_indicator", "potassium_indicator",
                "ph", "ec_ds_m", "soil_temperature_c", "soil_moisture_percent",
            ):
                value = getattr(reading, key)
                if value is not None:
                    defaults[key] = value
            defaults["sensor_device_id"] = reading.sensor_device_id or "SOIL-001"
            st.caption(
                f"Yuklangan o'lchov: {reading.sensor_device_id} · "
                f"{str(reading.measured_at)[:19]}"
            )

    elif mode == "JSON yoki CSV yuklash":
        source = DataSource.FILE
        upload_columns = st.columns([3, 1])
        with upload_columns[0]:
            uploaded = st.file_uploader(
                "Sensor ma'lumot fayli", type=["json", "csv"], key="sensor_upload"
            )
        with upload_columns[1]:
            st.write("")
            st.download_button(
                "Namuna JSON",
                data=example_sensor_file("json"),
                file_name="agroiq_sensor_namuna.json",
                mime="application/json",
                width="stretch",
            )
        if uploaded is not None:
            result = parse_uploaded_sensor_file(uploaded.getvalue(), uploaded.name)
            if result.ok and result.reading:
                st.success(result.message_uz)
                for warning in result.warnings:
                    st.warning(warning)
                for key in (
                    "nitrogen_indicator", "phosphorus_indicator", "potassium_indicator",
                    "ph", "ec_ds_m", "soil_temperature_c", "soil_moisture_percent",
                ):
                    value = getattr(result.reading, key)
                    if value is not None:
                        defaults[key] = value
                defaults["sensor_device_id"] = result.reading.sensor_device_id or "CSV-UPLOAD"
            else:
                st.error(result.message_uz)

    prefill = st.session_state.get("prefill_field", {}) if mode == "Demo ssenariy" else {}

    # ---------------- FORMA ----------------
    with st.form("analysis_form"):
        # --- 2-BO'LIM: AGROIQ ANALIZATORI ---
        section_header(
            "2. AgroIQ fosfor analizatori",
            "Ushbu o'lchov o'simlik o'zlashtira oladigan fosforni baholash uchun ishlatiladi.",
        )
        with st.container(border=True):
            color_columns = st.columns(3)
            with color_columns[0]:
                red = st.number_input(
                    "Qizil (R)", 0.0, 255.0, float(defaults["red"]), 1.0,
                    help="Analizator sensori qayd etgan qizil kanal qiymati (0-255).",
                )
            with color_columns[1]:
                green = st.number_input(
                    "Yashil (G)", 0.0, 255.0, float(defaults["green"]), 1.0,
                    help="Yashil kanal qiymati (0-255).",
                )
            with color_columns[2]:
                blue = st.number_input(
                    "Ko'k (B)", 0.0, 255.0, float(defaults["blue"]), 1.0,
                    help="Ko'k kanal qiymati. Fosfor ko'p bo'lsa eritma ko'kroq bo'ladi.",
                )

            condition_columns = st.columns(2)
            with condition_columns[0]:
                reaction_time = st.number_input(
                    "Reaksiya vaqti (sek)", 10.0, 1800.0, float(defaults["reaction_time_sec"]), 10.0,
                    help="Reagent qo'shilgandan keyin o'tgan vaqt. Odatda 8-12 daqiqa.",
                )
            with condition_columns[1]:
                analyzer_temp = st.number_input(
                    "Namuna harorati (°C)", 0.0, 60.0, float(defaults["analyzer_temperature_c"]), 0.5,
                    help="Harorat rang hosil bo'lish tezligiga ta'sir qiladi.",
                )

            device_columns = st.columns(2)
            with device_columns[0]:
                analyzer_id = st.text_input(
                    "Analizator ID", value=str(defaults["analyzer_device_id"]),
                    help="Qurilma identifikatori — natijani aniq qurilmaga bog'lash uchun.",
                )
            with device_columns[1]:
                cartridge_id = st.text_input(
                    "Kartrij partiyasi (ixtiyoriy)", value=str(defaults["cartridge_batch_id"]),
                    help="Reagent kartriji partiyasi — kalibrlash kuzatuvi uchun.",
                )
            color_swatch(red, green, blue)

        # --- 3-BO'LIM: UNIVERSAL SENSOR ---
        section_header(
            "3. Universal tuproq sensori",
            "Tuproq sharoiti va NPK skrining indikatorlari.",
        )
        with st.container(border=True):
            npk_columns = st.columns(3)
            with npk_columns[0]:
                nitrogen = st.number_input(
                    "Azot indikatori (N)", 0.0, 2000.0, float(defaults["nitrogen_indicator"]), 1.0,
                    help="Sensorning azot skrining indikatori. Laboratoriya tahlilini almashtirmaydi.",
                )
            with npk_columns[1]:
                sensor_p = st.number_input(
                    "Fosfor indikatori (P)", 0.0, 2000.0, float(defaults["phosphorus_indicator"]), 1.0,
                    help=(
                        "Bu sensorning umumiy yoki empirik fosfor indikatori bo'lib, AgroIQ "
                        "kolorimetrik Olsen-P natijasini almashtirmaydi."
                    ),
                )
            with npk_columns[2]:
                potassium = st.number_input(
                    "Kaliy indikatori (K)", 0.0, 5000.0, float(defaults["potassium_indicator"]), 1.0,
                    help="Sensorning kaliy skrining indikatori. Laboratoriya tahlilini almashtirmaydi.",
                )

            soil_columns = st.columns(4)
            with soil_columns[0]:
                ph = st.number_input(
                    "pH", 3.0, 10.0, float(defaults["ph"]), 0.1,
                    help="Tuproq muhiti. 6.5-7.5 fosfor uchun eng qulay oraliq.",
                )
            with soil_columns[1]:
                ec = st.number_input(
                    "EC (dS/m)", 0.0, 30.0, float(defaults["ec_ds_m"]), 0.1,
                    help="Sho'rlanish ko'rsatkichi. 4 dS/m dan yuqori qiymat sho'rlanishdan darak beradi.",
                )
            with soil_columns[2]:
                soil_temp = st.number_input(
                    "Tuproq harorati (°C)", -20.0, 70.0, float(defaults["soil_temperature_c"]), 0.5,
                    help="Sovuq tuproqda oziq moddalar sekin o'zlashtiriladi.",
                )
            with soil_columns[3]:
                moisture = st.number_input(
                    "Namlik (%)", 0.0, 100.0, float(defaults["soil_moisture_percent"]), 1.0,
                    help="Tuproqdagi namlik ulushi. Quruq tuproqda o'g'it sekin o'zlashadi.",
                )

            meta_columns = st.columns(3)
            with meta_columns[0]:
                sensor_id = st.text_input(
                    "Sensor ID", value=str(defaults["sensor_device_id"]),
                    help="Universal sensor identifikatori.",
                )
            with meta_columns[1]:
                measured_date = st.date_input(
                    "O'lchov sanasi", value=datetime.now().date(),
                    help="O'lchov qachon olingan. 24 soatdan eski ma'lumot eskirgan deb belgilanadi.",
                )
            with meta_columns[2]:
                measured_time = st.time_input(
                    "O'lchov vaqti", value=datetime.now().time().replace(microsecond=0),
                    help="O'lchov vaqti (mahalliy).",
                )

        # --- 4-BO'LIM: EKIN VA DALA ---
        section_header("4. Ekin va dala ma'lumotlari", "Tavsiyani aniqlashtirish uchun.")
        with st.container(border=True):
            crop_keys = list(crops)
            default_crop = prefill.get("crop_key", crop_keys[0])
            crop_columns = st.columns(3)
            with crop_columns[0]:
                crop_key = st.selectbox(
                    "Ekin", crop_keys,
                    index=crop_keys.index(default_crop) if default_crop in crop_keys else 0,
                    format_func=lambda key: crops[key],
                    help="Ekin turi fosforga bo'lgan talabni belgilaydi.",
                )
            stages = _stage_options(configs, crop_key)
            stage_keys = list(stages)
            default_stage = prefill.get("growth_stage_key", stage_keys[0] if stage_keys else None)
            with crop_columns[1]:
                stage_key = st.selectbox(
                    "O'sish bosqichi", stage_keys,
                    index=stage_keys.index(default_stage) if default_stage in stage_keys else 0,
                    format_func=lambda key: stages[key],
                    help="Kech bosqichlarda fosfor qo'llash samaradorligi pasayadi.",
                ) if stage_keys else None
            with crop_columns[2]:
                region = st.text_input(
                    "Hudud", value=prefill.get("region", ""),
                    help="Viloyat yoki tuman nomi — hisobotda ko'rsatiladi.",
                )

            textures = _rule_options(configs, "soil_textures")
            texture_keys = list(textures)
            default_texture = prefill.get("soil_texture_key", "nomalum")
            field_columns = st.columns(3)
            with field_columns[0]:
                texture_key = st.selectbox(
                    "Tuproq mexanik tarkibi (ixtiyoriy)", texture_keys,
                    index=texture_keys.index(default_texture) if default_texture in texture_keys else 0,
                    format_func=lambda key: textures[key],
                    help="Og'ir tuproqda fosforning bog'lanishi kuchliroq.",
                )
            with field_columns[1]:
                area = st.number_input(
                    "Dala maydoni (ga)", 0.01, 100000.0,
                    float(prefill.get("field_area_ha", 10.0)), 0.5,
                    help="Umumiy o'g'it miqdorini hisoblash uchun kerak.",
                )
            with field_columns[2]:
                target_yield = st.number_input(
                    "Maqsadli hosildorlik (t/ga)", 0.1, 50.0,
                    float(prefill.get("target_yield_t_ha", 3.5)), 0.1,
                    help="Rejalashtirilgan hosil — oziq modda talabini belgilaydi.",
                )

            history = _rule_options(configs, "previous_fertilizer_credit")
            history_keys = list(history)
            default_history = prefill.get("previous_fertilizer_key", "nomalum")
            irrigation = _rule_options(configs, "irrigation_methods")
            irrigation_keys = list(irrigation)
            default_irrigation = prefill.get("irrigation_method_key", irrigation_keys[0])

            history_columns = st.columns(3)
            with history_columns[0]:
                previous_crop = st.text_input(
                    "Oldingi ekin", value=prefill.get("previous_crop", ""),
                    help="O'tgan mavsumda ekilgan ekin.",
                )
            with history_columns[1]:
                previous_fertilizer = st.selectbox(
                    "Oldingi fosforli o'g'itlash", history_keys,
                    index=history_keys.index(default_history) if default_history in history_keys else 0,
                    format_func=lambda key: history[key],
                    help="Tuproqdagi qoldiq fosforni hisobga olish uchun.",
                )
            with history_columns[2]:
                irrigation_method = st.selectbox(
                    "Sug'orish usuli", irrigation_keys,
                    index=irrigation_keys.index(default_irrigation) if default_irrigation in irrigation_keys else 0,
                    format_func=lambda key: irrigation[key],
                    help="Qo'llash usuli bo'yicha izoh beriladi.",
                )

            product_keys = list(products)
            default_product = prefill.get("fertilizer_key", product_keys[0])
            product_columns = st.columns(2)
            with product_columns[0]:
                fertilizer_key = st.selectbox(
                    "O'g'it mahsuloti", product_keys,
                    index=product_keys.index(default_product) if default_product in product_keys else 0,
                    format_func=lambda key: products[key],
                    help="Mahsulotdagi P2O5 ulushi asosida miqdor hisoblanadi.",
                )
            with product_columns[1]:
                price = st.number_input(
                    "O'g'it narxi (so'm/kg, ixtiyoriy)", 0.0, 1000000.0, 0.0, 100.0,
                    help="0 qoldirilsa, konfiguratsiyadagi indikativ narx ishlatiladi.",
                )

        submitted = st.form_submit_button(
            "AI tahlilini boshlash", type="primary", width="stretch"
        )

    if submitted:
        measured_at = datetime.combine(
            measured_date,
            measured_time if isinstance(measured_time, dtime) else dtime(12, 0),
        ).replace(tzinfo=timezone.utc)

        analyzer_data = {
            "red": red, "green": green, "blue": blue,
            "reaction_time_sec": reaction_time,
            "analyzer_temperature_c": analyzer_temp,
            "analyzer_device_id": analyzer_id or None,
            "cartridge_batch_id": cartridge_id or None,
        }
        sensor_data = {
            "nitrogen_indicator": nitrogen, "phosphorus_indicator": sensor_p,
            "potassium_indicator": potassium, "ph": ph, "ec_ds_m": ec,
            "soil_temperature_c": soil_temp, "soil_moisture_percent": moisture,
            "sensor_device_id": sensor_id or None, "measured_at": measured_at,
        }
        field_data = {
            "crop_key": crop_key, "growth_stage_key": stage_key, "region": region or None,
            "soil_texture_key": texture_key, "field_area_ha": area,
            "target_yield_t_ha": target_yield, "previous_crop": previous_crop or None,
            "previous_fertilizer_key": previous_fertilizer,
            "irrigation_method_key": irrigation_method, "fertilizer_key": fertilizer_key,
            "fertilizer_price": price if price > 0 else None,
        }
        with st.spinner("AI tahlili bajarilmoqda..."):
            if run_analysis(analyzer_data, sensor_data, field_data, source, sample_label):
                goto(PAGE_RESULTS)


# ---------------------------------------------------------------------------
# C. NATIJALAR
# ---------------------------------------------------------------------------


def page_results(configs: dict[str, Any], model_info: dict[str, Any] | None) -> None:
    result = st.session_state.get("result")
    if not result:
        st.info("Hozircha natija yo'q. Avval tahlil o'tkazing.")
        if st.button("Yangi tahlilga o'tish", type="primary"):
            goto(PAGE_ANALYSIS)
        return

    fusion = result["fusion"]
    estimate = result["estimate"]
    recommendation = result["recommendation"]
    analyzer = result["analyzer"]
    sensor = result["sensor"]

    st.markdown("## Tahlil natijasi")
    st.caption(
        f"Namuna: **{result['sample_label']}** · Tahlil ID: `{result['analysis_id']}`"
    )

    if model_info and model_info.get("dataset_kind") == "demo":
        banner(f"⚠️ {DEMO_MODEL_BANNER_UZ}")

    if fusion.phosphorus_comparison.agrees is False:
        alert_card(
            "attention",
            "Fosfor bo'yicha qurilmalar o'rtasida tafovut",
            fusion.phosphorus_comparison.explanation_uz,
        )

    # --- Asosiy ko'rsatkichlar ---
    top = st.columns(6, gap="small")
    with top[0]:
        metric_card(
            "Fosfor (Olsen-P)",
            f"{estimate.olsen_p_mg_kg:.1f}", "mg/kg",
            sub=f"±{estimate.uncertainty_mg_kg:.1f} · o'zlashtiriladigan", accent=PRIMARY,
        )
    with top[1]:
        metric_card("Fosfor holati", estimate.status_label_uz,
                    sub=f"{estimate.ci95_low:.0f}–{estimate.ci95_high:.0f} mg/kg (95%)",
                    accent=estimate.status_color)
    with top[2]:
        ph_info = fusion.conditions["ph"]
        metric_card("pH", f"{fusion.ph:.1f}" if fusion.ph is not None else "—",
                    sub=ph_info.band_label_uz, accent=ph_info.color)
    with top[3]:
        ec_info = fusion.conditions["ec_ds_m"]
        metric_card("EC", f"{fusion.ec_ds_m:.1f}" if fusion.ec_ds_m is not None else "—",
                    "dS/m", sub=ec_info.band_label_uz, accent=ec_info.color)
    with top[4]:
        moisture_info = fusion.conditions["soil_moisture_percent"]
        metric_card(
            "Namlik",
            f"{fusion.soil_moisture_percent:.0f}" if fusion.soil_moisture_percent is not None else "—",
            "%", sub=moisture_info.band_label_uz, accent=moisture_info.color,
        )
    with top[5]:
        metric_card("Tavsiya ishonchliligi", fusion.confidence_label_uz,
                    sub="Konservativ baholash", accent=ACCENT)

    st.write("")
    # --- Sensor kartalari ---
    section_header("Universal sensor ko'rsatkichlari", "Skrining indikatorlari")
    sensor_columns = st.columns(4, gap="medium")
    with sensor_columns[0]:
        nutrient_card(fusion.nitrogen)
    with sensor_columns[1]:
        sensor_p = fusion.phosphorus_comparison.sensor_indicator
        metric_card(
            "Universal sensor P indikatori",
            f"{sensor_p:.1f}" if sensor_p is not None else "—",
            sub="Bu Olsen-P EMAS — faqat qiyosiy ko'rsatkich",
            accent="#9AA69C",
        )
    with sensor_columns[2]:
        nutrient_card(fusion.potassium)
    with sensor_columns[3]:
        temperature_info = fusion.conditions["soil_temperature_c"]
        metric_card(
            "Tuproq harorati",
            f"{fusion.soil_temperature_c:.1f}" if fusion.soil_temperature_c is not None else "—",
            "°C", sub=temperature_info.band_label_uz, accent=temperature_info.color,
        )

    st.markdown("---")

    # --- A. Tuproq diagnostikasi ---
    section_header("A. Tuproq diagnostikasi")
    diagnostic_columns = st.columns([3, 2], gap="large")
    with diagnostic_columns[0]:
        rows = condition_chart_data(fusion.conditions)
        if rows:
            st.plotly_chart(
                soil_condition_chart(rows), width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.info("Tuproq sharoiti o'lchovlari kiritilmagan.")
    with diagnostic_columns[1]:
        if fusion.diagnostic_index:
            index = fusion.diagnostic_index
            st.markdown(f"**{index['title_uz']}**")
            st.plotly_chart(
                diagnostic_index_gauge(index), width="stretch",
                config={"displayModeBar": False},
            )
            st.caption(f"{index['label_uz']} — {index['disclaimer_uz']}")

    # --- B. Fosfor bo'yicha AI natijasi ---
    section_header("B. Fosfor bo'yicha AI natijasi", "Kolorimetrik model — asosiy manba")
    phosphorus_columns = st.columns([3, 2], gap="large")
    with phosphorus_columns[0]:
        st.plotly_chart(
            phosphorus_gauge(estimate.olsen_p_mg_kg, configs["thresholds"], estimate.uncertainty_mg_kg),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption(estimate.status_description_uz)
    with phosphorus_columns[1]:
        comparison = fusion.phosphorus_comparison
        if comparison.sensor_indicator is not None and comparison.colorimetric_olsen_p is not None:
            st.plotly_chart(
                phosphorus_comparison_chart(
                    comparison.colorimetric_olsen_p, comparison.sensor_indicator,
                    bool(comparison.agrees),
                ),
                width="stretch", config={"displayModeBar": False},
            )
            st.caption(comparison.explanation_uz)

    # --- C. O'g'itlash tavsiyasi ---
    section_header("C. O'g'itlash tavsiyasi")
    if recommendation.no_phosphorus_needed:
        alert_card(
            "info",
            "Ushbu mavsumda fosforli o'g'it tavsiya etilmaydi",
            recommendation.application_stage_uz + " " + recommendation.application_method_uz,
        )
    else:
        rec_columns = st.columns(4, gap="medium")
        with rec_columns[0]:
            metric_card("Kerakli oziq modda",
                        f"{recommendation.required_p2o5_low:.0f}–{recommendation.required_p2o5_high:.0f}",
                        "kg P₂O₅/ga")
        with rec_columns[1]:
            metric_card(f"{recommendation.product_name_uz} me'yori",
                        f"{recommendation.product_kg_ha_low:.0f}–{recommendation.product_kg_ha_high:.0f}",
                        "kg/ga", sub=f"Tarkibida {recommendation.p2o5_fraction * 100:.0f}% P₂O₅",
                        accent=ACCENT)
        with rec_columns[2]:
            metric_card(f"Dala uchun jami ({recommendation.field_area_ha:g} ga)",
                        f"{recommendation.total_product_kg_low:.0f}–{recommendation.total_product_kg_high:.0f}",
                        "kg",
                        sub=f"≈ {recommendation.total_bags_low:.0f}–{recommendation.total_bags_high:.0f} qop (50 kg)",
                        accent=ACCENT)
        with rec_columns[3]:
            if recommendation.estimated_cost_uzs_low is not None:
                low = recommendation.estimated_cost_uzs_low / 1_000_000
                high = recommendation.estimated_cost_uzs_high / 1_000_000
                metric_card("Taxminiy xarajat", f"{low:.1f}–{high:.1f}", "mln",
                            sub=f"so'm · {recommendation.field_area_ha:g} ga uchun")

    timing_columns = st.columns(2, gap="large")
    with timing_columns[0]:
        info_card("Qo'llash muddati", recommendation.application_stage_uz)
    with timing_columns[1]:
        info_card("Qo'llash usuli", recommendation.application_method_uz)

    st.write("")
    npk_columns = st.columns(2, gap="large")
    for column, assessment in zip(npk_columns, (fusion.nitrogen, fusion.potassium), strict=True):
        with column:
            info_card(
                f"{assessment.label_uz} — {assessment.status_label_uz}",
                f"{assessment.advice_uz}<br/><br/><i>Miqdoriy me'yor berilmadi: mahalliy "
                f"laboratoriya kalibrlashi mavjud emas.</i>",
            )

    # --- D. Ogohlantirishlar ---
    flag_details = fusion.flag_details()
    if recommendation.warnings or flag_details:
        section_header("D. Ogohlantirishlar")
        for warning in recommendation.warnings:
            alert_card(warning.level, warning.title_uz, warning.message_uz)
        shown = {warning.title_uz for warning in recommendation.warnings}
        for detail in flag_details:
            if detail["title"] not in shown:
                alert_card(detail["level"], detail["title"], detail["message"])

    # --- E. Nima uchun bu tavsiya berildi? ---
    section_header("E. Nima uchun bu tavsiya berildi?")
    for index, reason in enumerate(result["explanations"], start=1):
        st.markdown(
            f"""<div style="display:flex;gap:.7rem;margin-bottom:.55rem;align-items:flex-start;">
            <div style="min-width:26px;height:26px;border-radius:8px;background:{PRIMARY};color:#fff;
                        display:flex;align-items:center;justify-content:center;font-size:.8rem;
                        font-weight:700;">{index}</div>
            <div style="color:{INK};font-size:.95rem;line-height:1.55;padding-top:.15rem;">{reason}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # --- F. Texnik ma'lumotlar ---
    section_header("F. Agronom uchun texnik ma'lumotlar")
    with st.expander("Texnik tafsilotlarni ochish"):
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
            width="stretch", hide_index=True,
        )
        if recommendation.growth_stage_note_uz:
            st.markdown(f"**O'sish bosqichi:** {recommendation.growth_stage_note_uz}")
        if recommendation.previous_fertilizer_note_uz:
            st.markdown(f"**Oldingi o'g'itlash:** {recommendation.previous_fertilizer_note_uz}")
        if recommendation.irrigation_note_uz:
            st.markdown(f"**Sug'orish usuli:** {recommendation.irrigation_note_uz}")
        if recommendation.soil_texture_note_uz:
            st.markdown(f"**Tuproq tarkibi:** {recommendation.soil_texture_note_uz}")

    # --- G. Ma'lumot manbalari va ishonchlilik ---
    section_header("G. Ma'lumot manbalari va ishonchlilik")
    provenance_columns = st.columns([2, 1], gap="large")
    with provenance_columns[0]:
        st.markdown("**Ma'lumot manbalari**")
        source_badges(fusion.data_sources, DATA_SOURCE_LABELS_UZ)
        device_rows = []
        if analyzer.analyzer_device_id:
            device_rows.append(f"Analizator: `{analyzer.analyzer_device_id}`")
        if analyzer.cartridge_batch_id:
            device_rows.append(f"Kartrij: `{analyzer.cartridge_batch_id}`")
        if sensor and sensor.sensor_device_id:
            device_rows.append(f"Sensor: `{sensor.sensor_device_id}`")
        if sensor and sensor.measured_at:
            device_rows.append(f"Sensor o'lchov vaqti: `{str(sensor.measured_at)[:19]}`")
        if device_rows:
            st.markdown(" · ".join(device_rows))
        if fusion.quality_flags:
            st.markdown("**Sifat bayroqlari**")
            st.markdown(
                " ".join(f"`{flag}`" for flag in fusion.quality_flags)
            )
    with provenance_columns[1]:
        st.markdown(
            f"Umumiy ishonchlilik: {confidence_badge(fusion.confidence, fusion.confidence_label_uz)}",
            unsafe_allow_html=True,
        )
        st.caption(
            "Ishonchlilik o'lchov sifati, kalibrlash oralig'i va qurilmalar mosligiga qarab "
            "konservativ tarzda pasaytiriladi."
        )

    # --- PDF va harakatlar ---
    st.markdown("---")
    action_columns = st.columns([2, 1, 1])
    with action_columns[0]:
        try:
            pdf_bytes = build_pdf_report(
                estimate=estimate,
                recommendation=recommendation,
                measurements={
                    "red": analyzer.red, "green": analyzer.green, "blue": analyzer.blue,
                    "reaction_time_sec": analyzer.reaction_time_sec,
                    "sample_temperature_c": analyzer.analyzer_temperature_c,
                    "ph": fusion.ph or 0.0, "ec_ds_m": fusion.ec_ds_m or 0.0,
                    "moisture_pct": fusion.soil_moisture_percent or 0.0,
                    "soil_temperature_c": fusion.soil_temperature_c,
                },
                explanations=result["explanations"],
                model_info=model_info,
                analysis_id=result["analysis_id"],
                sample_label=result["sample_label"],
                fusion=fusion,
                device_info={
                    "analyzer_device_id": analyzer.analyzer_device_id,
                    "cartridge_batch_id": analyzer.cartridge_batch_id,
                    "sensor_device_id": sensor.sensor_device_id if sensor else None,
                    "sensor_measured_at": str(sensor.measured_at)[:19] if sensor and sensor.measured_at else None,
                    "region": result["field"].get("region"),
                    "previous_crop": result["field"].get("previous_crop"),
                },
            )
            st.download_button(
                "PDF hisobotni yuklab olish", data=pdf_bytes,
                file_name=f"AgroIQ_hisobot_{result['analysis_id']}.pdf",
                mime="application/pdf", width="stretch",
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


# ---------------------------------------------------------------------------
# D. QURILMALAR
# ---------------------------------------------------------------------------


def page_devices(configs: dict[str, Any]) -> None:
    st.markdown("## Qurilmalar")
    st.caption(
        "AgroIQ ikkita mustaqil o'lchov manbaidan foydalanadi. Ular turli vazifalarni "
        "bajaradi va natijalari hech qachon aralashtirilmaydi."
    )

    profiles = configs["devices"]
    analyzer_profile = get_device_profile("agroiq_analyzer", profiles)
    sensor_profile = get_device_profile("universal_soil_sensor", profiles)

    result = st.session_state.get("result")
    api_reading = st.session_state.get("api_reading")

    columns = st.columns(2, gap="large")
    with columns[0]:
        last_measurement = "—"
        analyzer_state, analyzer_state_label = "manual", "Qo'lda / demo"
        device_id = analyzer_profile["default_device_id"]
        if result:
            analyzer = result["analyzer"]
            device_id = analyzer.analyzer_device_id or device_id
            last_measurement = f"{result['analysis_id']} · Olsen-P {result['estimate'].olsen_p_mg_kg:.1f} mg/kg"
            analyzer_state, analyzer_state_label = "demo", "Ma'lumot mavjud"
        device_card(
            name=analyzer_profile["name_uz"],
            role=analyzer_profile["role_uz"],
            state=analyzer_state,
            state_label=analyzer_state_label,
            rows=[
                ("Qurilma ID", device_id),
                ("O'lchash tamoyili", analyzer_profile["measurement_principle_uz"]),
                ("Ulanish", analyzer_profile["transport_uz"]),
                ("Chiqish", analyzer_profile["output_uz"]),
                ("Oxirgi o'lchov", last_measurement),
                ("Kalibrlash", "Demo (sintetik) — real kalibrlash zarur"),
                ("Fosfor uchun rol", "ASOSIY manba"),
            ],
        )
    with columns[1]:
        sensor_state, sensor_state_label = "offline", "Ulanmagan"
        sensor_id = sensor_profile["default_device_id"]
        last_sensor = "—"
        if api_reading is not None:
            sensor_state, sensor_state_label = "online", "API orqali ulangan"
            sensor_id = api_reading.sensor_device_id or sensor_id
            last_sensor = f"pH {api_reading.ph} · EC {api_reading.ec_ds_m} · {str(api_reading.measured_at)[:16]}"
        elif result and result.get("sensor"):
            sensor = result["sensor"]
            sensor_state, sensor_state_label = "demo", "Ma'lumot mavjud"
            sensor_id = sensor.sensor_device_id or sensor_id
            last_sensor = f"pH {sensor.ph} · EC {sensor.ec_ds_m} · {str(sensor.measured_at)[:16]}"
        device_card(
            name=sensor_profile["name_uz"],
            role=sensor_profile["role_uz"],
            state=sensor_state,
            state_label=sensor_state_label,
            rows=[
                ("Sensor ID", sensor_id),
                ("Aloqa turi", "RS485 / Modbus RTU → Gateway API"),
                ("O'lchanadigan parametrlar", "pH, EC, harorat, namlik, N, P, K"),
                ("Oxirgi o'lchov", last_sensor),
                ("Kalibrlash", "NPK: kalibrlanmagan (skrining)"),
                ("Fosfor uchun rol", "Faqat qiyosiy indikator"),
            ],
        )

    st.write("")
    section_header("Integratsiya arxitekturasi")
    st.markdown("**Universal tuproq sensori**")
    architecture_row(profiles["architecture_uz"]["sensor_chain"])
    st.write("")
    st.markdown("**Portativ fosfor analizatori**")
    architecture_row(profiles["architecture_uz"]["analyzer_chain"])
    st.caption(
        "Streamlit Community Cloud foydalanuvchining lokal RS485 portiga to'g'ridan-to'g'ri "
        "ulana olmaydi. Shu sababli apparat bilan aloqa lokal Edge Gateway orqali, REST API "
        "yordamida amalga oshiriladi. Demo, qo'lda va fayl rejimlari apparatsiz ham ishlaydi."
    )

    st.markdown("---")
    section_header("Ulanishni sozlash", "Demo · Qo'lda · API")
    defaults = api_defaults(profiles)
    connection_mode = st.radio(
        "Ulanish rejimi", ["Demo", "Qo'lda", "API"], horizontal=True, key="conn_mode"
    )

    if connection_mode == "Demo":
        st.info(
            "Demo rejimida ma'lumot tayyorlangan ssenariylardan olinadi. Apparat talab "
            "qilinmaydi — namoyish uchun eng ishonchli variant."
        )
        if st.button("Demo ssenariylarga o'tish", type="primary"):
            goto(PAGE_DEMO)
    elif connection_mode == "Qo'lda":
        st.info(
            "Qo'lda rejimida barcha qiymatlar 'Yangi tahlil' sahifasidagi shakl orqali "
            "kiritiladi. Bu internet yoki apparat bo'lmaganda ishlaydi."
        )
        if st.button("Yangi tahlilga o'tish", type="primary"):
            goto(PAGE_ANALYSIS)
    else:
        st.caption(
            "API tokeni faqat sessiya xotirasida saqlanadi — u manba kodiga, "
            "konfiguratsiyaga yoki GitHub'ga hech qachon yozilmaydi."
        )
        endpoint_columns = st.columns([3, 1])
        with endpoint_columns[0]:
            endpoint = st.text_input(
                "Endpoint URL",
                value=st.session_state.get("api_endpoint", defaults["mock_endpoint"]),
                key="api_endpoint",
                help=f"Namuna: {defaults['example_endpoint']}",
            )
        with endpoint_columns[1]:
            timeout = st.number_input(
                "Timeout (sek)", 1.0, float(defaults.get("max_timeout_seconds", 30)), 5.0, 1.0,
                key="api_timeout",
            )
        token = st.text_input("API token (ixtiyoriy)", type="password", key="api_token")

        button_columns = st.columns(2)
        with button_columns[0]:
            if st.button("Ulanishni tekshirish", width="stretch"):
                outcome = check_connection(endpoint, token or None, float(timeout))
                (st.success if outcome.ok else st.error)(outcome.message_uz)
        with button_columns[1]:
            if st.button("Oxirgi ma'lumotni olish", type="primary", width="stretch"):
                outcome = fetch_from_gateway(endpoint, token or None, float(timeout))
                if outcome.ok and outcome.reading:
                    st.session_state["api_reading"] = outcome.reading
                    st.success(outcome.message_uz)
                    st.rerun()
                else:
                    st.error(outcome.message_uz)
                    st.info("Qo'lda kiritish yoki fayl yuklash rejimidan foydalanishingiz mumkin.")

        if api_reading is not None:
            st.markdown("**Oxirgi qabul qilingan o'lchov**")
            st.dataframe(
                pd.DataFrame([{
                    "Sensor ID": api_reading.sensor_device_id,
                    "pH": api_reading.ph,
                    "EC (dS/m)": api_reading.ec_ds_m,
                    "Harorat (°C)": api_reading.soil_temperature_c,
                    "Namlik (%)": api_reading.soil_moisture_percent,
                    "N": api_reading.nitrogen_indicator,
                    "P (indikator)": api_reading.phosphorus_indicator,
                    "K": api_reading.potassium_indicator,
                }]),
                width="stretch", hide_index=True,
            )
            if st.button("Ushbu ma'lumot bilan tahlil qilish", type="primary"):
                goto(PAGE_ANALYSIS)

    st.write("")
    with st.expander("Lokal Edge Gateway'ni ishga tushirish"):
        st.markdown(
            "Haqiqiy sensor bilan ishlash uchun gateway sensor yonidagi kompyuterda "
            "ishga tushiriladi:"
        )
        st.code(
            "pip install -r edge_gateway/requirements-edge.txt\n"
            "python -m edge_gateway.gateway --mode mock --port 8000\n\n"
            "# Haqiqiy Modbus sensori bilan:\n"
            "python -m edge_gateway.gateway --mode modbus --serial-port /dev/ttyUSB0",
            language="bash",
        )
        st.markdown(
            "So'ngra yuqoridagi API manziliga `http://<gateway-ip>:8000/api/v1/readings/latest` "
            "kiriting. Batafsil: `edge_gateway/README.md`."
        )


# ---------------------------------------------------------------------------
# E. MODEL VA VALIDATSIYA
# ---------------------------------------------------------------------------


def page_model(model_info: dict[str, Any] | None, configs: dict[str, Any]) -> None:
    st.markdown("## Model va validatsiya")
    if model_info is None:
        st.error(
            "Model yuklanmagan. `python scripts/train_model.py` buyrug'ini bajaring va sahifani yangilang."
        )
        return

    st.caption(
        "Ushbu sahifa mustaqil ekspertlar uchun: arxitektura, o'qitish metodologiyasi, "
        "metrikalar va cheklovlar ochiq ko'rsatilgan."
    )
    if model_info["dataset_kind"] == "demo":
        banner(f"⚠️ {DEMO_MODEL_BANNER_UZ}")

    # --- Arxitektura ---
    section_header("Ma'lumotlarni birlashtirish arxitekturasi")
    st.markdown("**1-yo'nalish — fosfor (miqdoriy)**")
    architecture_row(["Kolorimetrik ma'lumot", "Olsen-P AI modeli"])
    st.write("")
    st.markdown("**2-yo'nalish — tuproq sharoiti (kontekst)**")
    architecture_row(["Universal sensor", "Tuproq sharoiti klassifikatori"])
    st.write("")
    st.markdown("**Birlashtirish**")
    architecture_row(
        ["Ikkala yo'nalish", "Tushuntiriladigan birlashtirish", "Tavsiya dvigateli"]
    )

    st.write("")
    architecture_columns = st.columns(2, gap="large")
    with architecture_columns[0]:
        alert_card(
            "info", "Fosfor: kolorimetrik model ASOSIY manba",
            "AI modeli o'simlik o'zlashtira oladigan fosforni (Olsen-P ekvivalenti) FAQAT "
            "optik xususiyatlardan baholaydi. Universal sensorning P indikatori bu qiymatni "
            "almashtirmaydi va u bilan o'rtachalanmaydi — u faqat moslik tekshiruvi uchun ishlatiladi.",
        )
    with architecture_columns[1]:
        alert_card(
            "caution", "Azot va kaliy: skrining indikatorlari",
            "N va K qiymatlari mahalliy laboratoriya kalibrlashi mavjud bo'lmaguncha faqat "
            "sifatiy baho sifatida ko'rsatiladi. Miqdoriy me'yor "
            "<code>config/nutrient_calibration.json</code> faylida kalibrlash tasdiqlanmaguncha "
            "berilmaydi.",
        )

    st.markdown("---")
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
        section_header("Ma'lumotlar to'plami")
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
                f"- **GroupKFold CV:** R² = {cv['r2']:.3f}, RMSE = {cv['rmse']:.2f} mg/kg "
                f"({int(cv['n_splits'])} fold)"
            )
    with info_columns[1]:
        section_header("Modelning kirish o'zgaruvchilari")
        pills = "".join(
            f'<span class="agro-pill">{name}</span>'
            for name in ["RGB kanallar", "Normallashgan RGB", "RGB nisbatlari",
                         "Hue / Saturation / Value", "Absorbsiya indekslari",
                         "Rang indekslari", "Reaksiya vaqti", "Harorat"]
        )
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown(f"Jami **{len(model_info['feature_columns'])}** ta xususiyat.")
        st.markdown(
            "pH, EC va namlik fosfor konsentratsiyasini bashorat qilishda **ishlatilmaydi** — "
            "ular faqat tavsiya bosqichida qo'llaniladi."
        )

    st.markdown("---")
    chart_columns = st.columns(2, gap="large")
    with chart_columns[0]:
        section_header("Model nomzodlarini taqqoslash")
        st.plotly_chart(
            model_comparison_chart(model_info["candidates"], model_info["model_name"]),
            width="stretch", config={"displayModeBar": False},
        )
        st.dataframe(
            pd.DataFrame(model_info["candidates"]).rename(
                columns={"name": "Model", "r2": "R²", "rmse": "RMSE", "mae": "MAE"}
            ).round(4),
            width="stretch", hide_index=True,
        )
    with chart_columns[1]:
        importances = get_feature_importances()
        if importances is not None:
            section_header("Xususiyatlarning hissasi")
            st.plotly_chart(
                feature_importance_chart(importances), width="stretch",
                config={"displayModeBar": False},
            )
        elif model_info.get("test_actual"):
            section_header("Laboratoriya va model qiymatlari")
            st.plotly_chart(
                prediction_scatter(
                    pd.Series(model_info["test_actual"]), pd.Series(model_info["test_predicted"])
                ),
                width="stretch", config={"displayModeBar": False},
            )

    st.markdown("---")
    range_columns = st.columns(2, gap="large")
    with range_columns[0]:
        section_header("Kalibrlash oralig'i")
        st.dataframe(
            pd.DataFrame([
                {"Parametr": name, "Minimal": f"{bounds['min']:.1f}", "Maksimal": f"{bounds['max']:.1f}"}
                for name, bounds in model_info["calibration"].items()
            ]),
            width="stretch", hide_index=True,
        )
        target = model_info["target_range"]
        st.markdown(
            f"**Maqsad (Olsen-P) oralig'i:** {target['min']:.1f} – {target['max']:.1f} mg/kg"
        )
    with range_columns[1]:
        section_header("Noaniqlik va ishonchlilik")
        st.markdown(
            f"""
            Standart noaniqlik ikki manbadan birlashtiriladi:

            1. **Model kelishmovchiligi** — Random Forest tanlanganda daraxtlar tarqoqligi.
            2. **Qoldiq tarqoqlik** — geteroskedastik baho:
               σ ≈ {model_info['residual_intercept']:.2f} + {model_info['residual_slope']:.3f} × bashorat.

            Yakuniy: σ = √(σ_model² + σ_qoldiq²).

            Tavsiya ishonchliligi esa qo'shimcha ravishda sifat bayroqlari
            (kalibrlash oralig'i, ma'lumot yangiligi, qurilmalar mosligi) bo'yicha
            konservativ pasaytiriladi.
            """
        )

    st.markdown("---")
    section_header("Ochiq cheklovlar", "Yashirilmaydi")
    limit_columns = st.columns(2, gap="large")
    limitations = [
        ("Sintetik ma'lumotda o'qitilgan",
         "Joriy model sintetik demo datasetda kalibrlangan. Ko'rsatilgan metrikalar real "
         "dala aniqligini kafolatlamaydi."),
        ("Qurilma-laboratoriya juftliklari kerak",
         "Real kalibrlash uchun kamida 150-300 juft (qurilma o'lchovi + akkreditatsiyalangan "
         "laboratoriya natijasi) namuna talab qilinadi."),
        ("Dala validatsiyasi yakunlanmagan",
         "O'g'itlash me'yorlari hosildorlik bo'yicha dala tajribalari bilan hali tasdiqlanmagan."),
        ("N va K sertifikatlangan o'lchov emas",
         "Universal sensorning azot va kaliy qiymatlari laboratoriya o'lchovi sifatida "
         "qaralmaydi va miqdoriy me'yor uchun ishlatilmaydi."),
        ("Laboratoriya o'rnini bosmaydi",
         "AgroIQ tezkor dala bahosini beradi. Rasmiy agrokimyoviy xulosa uchun sertifikatlangan "
         "laboratoriya tahlili zarur."),
        ("Hududiy kalibrlash zarur",
         "Tuproq turi, karbonatlilik, reagent partiyasi va qurilma optikasi natijaga ta'sir qiladi."),
    ]
    for index, (title, body) in enumerate(limitations):
        with limit_columns[index % 2]:
            info_card(title, body)
            st.write("")


# ---------------------------------------------------------------------------
# F. DEMO REJIMI
# ---------------------------------------------------------------------------


def page_demo(configs: dict[str, Any]) -> None:
    st.markdown("## Demo rejimi")
    st.caption(
        "To'rtta tayyorlangan ssenariy platformaning turli sharoitlarda qanday javob "
        "berishini ko'rsatadi. Har biri ikkala qurilma ma'lumotini ham o'z ichiga oladi. "
        "Hech qanday fayl yuklash yoki apparat talab etilmaydi."
    )
    banner(f"⚠️ {DEMO_MODEL_BANNER_UZ}")

    if "demo_index" not in st.session_state:
        st.session_state["demo_index"] = 0

    for row_start in (0, 2):
        columns = st.columns(2, gap="large")
        for offset, column in enumerate(columns):
            index = row_start + offset
            if index >= len(SCENARIOS):
                continue
            scenario = SCENARIOS[index]
            with column:
                with st.container(border=True):
                    st.markdown(f"**{scenario['title_uz']}**")
                    st.caption(scenario["summary_uz"])
                    st.markdown(
                        f'<span class="agro-pill">{scenario["expected_uz"]}</span>',
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "Ishga tushirish", key=f"run_{scenario['demo_id']}",
                        width="stretch", type="primary" if index == 0 else "secondary",
                    ):
                        st.session_state["demo_index"] = index
                        with st.spinner("AI tahlili bajarilmoqda..."):
                            if _run_scenario(scenario["demo_id"]):
                                goto(PAGE_RESULTS)
        st.write("")

    next_columns = st.columns([1, 3])
    with next_columns[0]:
        if st.button("Keyingi demo namuna →", width="stretch"):
            next_index = (st.session_state["demo_index"] + 1) % len(SCENARIOS)
            st.session_state["demo_index"] = next_index
            with st.spinner("AI tahlili bajarilmoqda..."):
                if _run_scenario(SCENARIOS[next_index]["demo_id"]):
                    goto(PAGE_RESULTS)

    st.markdown("---")
    section_header("Ssenariylar parametrlari")
    rows = []
    for scenario in SCENARIOS:
        built = build_scenario(scenario["demo_id"])
        rows.append({
            "Ssenariy": scenario["title_uz"],
            "R/G/B": f"{built['analyzer']['red']:.0f}/{built['analyzer']['green']:.0f}/{built['analyzer']['blue']:.0f}",
            "Kutilgan P": f"{scenario['target_p']:.1f}",
            "Sensor P": built["sensor"]["phosphorus_indicator"],
            "pH": built["sensor"]["ph"],
            "EC": built["sensor"]["ec_ds_m"],
            "Namlik": built["sensor"]["soil_moisture_percent"],
            "Ekin": built["field"]["crop_key"],
            "Analizator": built["analyzer"]["analyzer_device_id"],
            "Sensor ID": built["sensor"]["sensor_device_id"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# G. LOYIHA HAQIDA
# ---------------------------------------------------------------------------


def page_about() -> None:
    st.markdown("## Loyiha haqida")

    columns = st.columns(2, gap="large")
    with columns[0]:
        section_header("Muammo")
        st.markdown(
            "Fermerlarda dala darajasidagi bir nechta tuproq parametriga va ekinga mos "
            "o'g'itlash bo'yicha yo'riqnomaga **tez, integratsiyalashgan va arzon** kirish "
            "imkoniyati yo'q."
        )
        section_header("Yechim")
        st.markdown(
            "AgroIQ maxsus **o'zlashtiriladigan fosfor analizatorini**, **universal tuproq "
            "sensorini** va **tushuntiriladigan AI tavsiya platformasini** birlashtiradi."
        )
        section_header("Maqsadli mijozlar")
        for item in [
            "Fermer xo'jaliklari", "Agroklasterlar", "Agronomlar",
            "Sayyor tuproq tahlili xizmatlari", "Agrolaboratoriyalar",
            "O'g'it distribyutorlari", "Agrokonsalting tashkilotlari",
        ]:
            st.markdown(f"- {item}")
    with columns[1]:
        section_header("Biznes modeli")
        for title, body in [
            ("AgroIQ o'qish qurilmasi savdosi", "Portativ kolorimetrik analizator."),
            ("Universal sensor to'plami", "Sensor sotuvi yoki mavjud sensorlarni integratsiya qilish."),
            ("Fosfor kartrijlari", "Takroriy sotiladigan reagent kartrijlari — barqaror daromad."),
            ("Xizmat sifatida tahlil", "Sayyor tuproq tahlili (har bir test uchun to'lov)."),
            ("AI platforma obunasi", "Yillik obuna asosida tahlil platformasi."),
            ("B2B klaster paneli", "Agroklasterlar uchun ko'p dalali boshqaruv paneli."),
            ("Kalibrlash va texnik xizmat", "Qurilmalarni davriy kalibrlash xizmati."),
            ("Agronomik API xizmatlari", "Uchinchi tomon tizimlari uchun API."),
        ]:
            info_card(title, body)
            st.write("")

    st.markdown("---")
    section_header("Kelgusi yo'l xaritasi")
    roadmap = st.columns(3, gap="medium")
    modules = [
        ("Validatsiyalangan N va K kartrijlari", "Laboratoriya kalibrlashi bilan miqdoriy azot va kaliy tavsiyalari."),
        ("Sug'orish suvi diagnostikasi", "Suvning sho'rligi va tarkibini baholash."),
        ("Mobil ilova", "Oflayn rejimda ishlaydigan fermer ilovasi."),
        ("Dron va sun'iy yo'ldosh", "Masofaviy zondlash ma'lumotlari bilan integratsiya."),
        ("O'zgaruvchan me'yorli xaritalar", "Dala ichidagi farqlarga mos o'g'itlash xaritalari."),
        ("Markaziy Osiyo kalibrlash datasetlari", "Hududiy tuproqlar uchun ochiq kalibrlash bazasi."),
    ]
    for index, (title, body) in enumerate(modules):
        with roadmap[index % 3]:
            info_card(title, body)
            st.write("")

    st.markdown("---")
    section_header("Jamoa")
    team = st.columns(4, gap="medium")
    roles = [
        ("[Ism Familiya]", "Loyiha rahbari", "Mahsulot strategiyasi va biznes rivojlantirish"),
        ("[Ism Familiya]", "AI / ML muhandisi", "Model, kalibrlash va ma'lumotlar tahlili"),
        ("[Ism Familiya]", "Agronom-maslahatchi", "Agronomik qoidalar va dala validatsiyasi"),
        ("[Ism Familiya]", "Apparat muhandisi", "Analizator optikasi va sensor integratsiyasi"),
    ]
    for column, (name, role, description) in zip(team, roles, strict=True):
        with column:
            info_card(name, f"<b>{role}</b><br/>{description}")

    st.write("")
    disclaimer_box(DISCLAIMER_UZ)


# ---------------------------------------------------------------------------
# Asosiy oqim
# ---------------------------------------------------------------------------


def main() -> None:
    inject_css()
    apply_pending_navigation()

    try:
        configs = get_configs()
    except ConfigError as exc:
        st.error(f"Konfiguratsiya xatosi: {exc}")
        st.stop()
        return

    model_info: dict[str, Any] | None = None
    model_error: str | None = None
    try:
        model_info = get_model_info()
    except ModelNotAvailableError as exc:
        model_error = str(exc)
    except Exception:  # noqa: BLE001 - traceback ko'rsatilmaydi
        model_error = (
            "Model faylini yuklashda kutilmagan xatolik. "
            "`python scripts/train_model.py` buyrug'ini bajaring."
        )

    # --- Yon panel: FAQAT brend, ogohlantirish va versiya (navigatsiya YO'Q) ---
    with st.sidebar:
        render_logo(compact=True)
        st.markdown(
            f"<div style='color:{MUTED};font-size:.82rem;line-height:1.5;margin-bottom:1rem;'>"
            "Integratsiyalashgan tuproq intellekti va aniq o'g'itlash platformasi"
            "</div>",
            unsafe_allow_html=True,
        )
        if model_info and model_info["dataset_kind"] == "demo":
            st.markdown(
                "<div style='background:#FFF8E1;border:1px solid #EBD08A;border-radius:10px;"
                "padding:.6rem .7rem;font-size:.76rem;color:#6B4B00;line-height:1.45;'>"
                "<b>Demo model</b><br/>Sintetik ma'lumotda o'qitilgan. Dala uchun kalibrlash zarur."
                "</div>",
                unsafe_allow_html=True,
            )
        st.write("")
        # Holat qurilmalar sahifasidagi kartalar bilan izchil bo'lishi kerak.
        api_reading = st.session_state.get("api_reading")
        active_result = st.session_state.get("result")
        if api_reading is not None:
            sensor_state, sensor_label = "online", "Sensor: API orqali ulangan"
        elif active_result and active_result.get("sensor") is not None:
            sensor_state, sensor_label = "demo", "Sensor: ma'lumot mavjud"
        else:
            sensor_state, sensor_label = "offline", "Sensor: ulanmagan"
        analyzer_pill = status_pill("demo", "Analizator: demo / qo'lda")
        sensor_pill = status_pill(sensor_state, sensor_label)
        st.markdown(
            f"<div style='font-size:.78rem;color:{MUTED};margin-bottom:.35rem;'>Ulanish holati</div>"
            f"{analyzer_pill}<div style='height:.35rem;'></div>{sensor_pill}",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            f"<div style='color:{MUTED};font-size:.72rem;line-height:1.5;'>"
            f"{__app_name__} MVP v{__version__}<br/>President AI Award prototipi</div>",
            unsafe_allow_html=True,
        )

    if model_error:
        st.error(model_error)
        st.info(
            "Modelni yaratish uchun terminalda quyidagi buyruqni bajaring:\n\n"
            "```\npython scripts/train_model.py\n```"
        )

    # --- Yuqori gorizontal navigatsiya ---
    page = top_navigation(PAGES, key="nav")

    if page == PAGE_HOME:
        page_home(configs, model_info)
    elif page == PAGE_ANALYSIS:
        if model_error:
            st.warning("Model tayyor bo'lmaguncha tahlil o'tkazib bo'lmaydi.")
        else:
            page_analysis(configs)
    elif page == PAGE_RESULTS:
        page_results(configs, model_info)
    elif page == PAGE_DEVICES:
        page_devices(configs)
    elif page == PAGE_MODEL:
        page_model(model_info, configs)
    elif page == PAGE_DEMO:
        if model_error:
            st.warning("Model tayyor bo'lmaguncha demo rejimini ishga tushirib bo'lmaydi.")
        else:
            page_demo(configs)
    elif page == PAGE_ABOUT:
        page_about()


if __name__ == "__main__":
    main()
