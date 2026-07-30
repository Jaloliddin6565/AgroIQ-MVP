"""Integratsiya testlari: navigatsiya, PDF, tavsiya va bulut mosligi (v0.2.0)."""

from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest
from pypdf import PdfReader

from src.data_fusion import fuse
from src.demo_scenarios import all_scenarios, build_scenario
from src.explanations import build_explanations
from src.model_inference import model_summary
from src.recommendation_engine import build_recommendation
from src.report_generator import build_pdf_report
from src.sensor_schemas import AnalyzerReading, DataSource, SensorReading
from src.soil_interpretation import assess_nutrient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Yuqori navigatsiya holati
# ---------------------------------------------------------------------------


def test_navigation_has_seven_pages_in_required_order():
    from app import PAGES

    assert PAGES == [
        "Bosh sahifa",
        "Yangi tahlil",
        "Natijalar",
        "Qurilmalar",
        "Model va validatsiya",
        "Demo rejimi",
        "Loyiha haqida",
    ]


def test_navigation_uses_horizontal_radio():
    """Navigatsiya native `st.radio(horizontal=True)` orqali qurilgan bo'lishi kerak."""

    from src import ui_components

    source = Path(ui_components.__file__).read_text(encoding="utf-8")
    top_nav = source[source.index("def top_navigation") : source.index("def section_header")]
    assert "horizontal=True" in top_nav
    assert "st.radio" in top_nav


def test_sidebar_contains_no_page_navigation():
    """Yon panelda takroriy navigatsiya BO'LMASLIGI kerak."""

    sidebar_block = APP_SOURCE[
        APP_SOURCE.index("with st.sidebar:") : APP_SOURCE.index("if model_error:")
    ]
    assert "top_navigation" not in sidebar_block
    assert "PAGES" not in sidebar_block
    for page in ("Yangi tahlil", "Demo rejimi", "Model va validatsiya"):
        assert page not in sidebar_block


def test_top_navigation_is_rendered_in_main_area():
    main_flow = APP_SOURCE[APP_SOURCE.index("def main()") :]
    assert "top_navigation(PAGES" in main_flow


def test_pending_navigation_pattern_avoids_widget_conflict():
    """Widget yaratilgandan keyin session_state o'zgartirilmasligi kerak."""

    assert "_pending_nav" in APP_SOURCE
    assert "apply_pending_navigation()" in APP_SOURCE
    main_flow = APP_SOURCE[APP_SOURCE.index("def main()") :]
    assert main_flow.index("apply_pending_navigation()") < main_flow.index("top_navigation(PAGES")


# ---------------------------------------------------------------------------
# Tavsiya: N va K faqat sifatiy
# ---------------------------------------------------------------------------


def _recommendation_with_nutrients(**overrides):
    kwargs = {
        "olsen_p_mg_kg": 7.0, "status_key": "low", "status_label_uz": "Past",
        "ph": 7.4, "ec_ds_m": 1.5, "moisture_pct": 24.0, "crop_key": "paxta",
        "field_area_ha": 10.0, "target_yield_t_ha": 3.5, "fertilizer_key": "ammofos",
        "nitrogen": assess_nutrient("nitrogen", 30.0),
        "potassium": assess_nutrient("potassium", 100.0),
    }
    kwargs.update(overrides)
    return build_recommendation(**kwargs)


def test_recommendation_reports_nk_qualitatively_only():
    recommendation = _recommendation_with_nutrients()
    assert recommendation.nitrogen_quantitative is False
    assert recommendation.potassium_quantitative is False
    assert recommendation.nitrogen_status_uz == "Ehtimol past"
    assert recommendation.potassium_status_uz == "Ehtimol past"
    assert "laboratoriya" in recommendation.nitrogen_advice_uz.lower()


def test_recommendation_has_no_quantitative_nk_fields():
    """Tavsiya obyektida azot/kaliy uchun kg/ga maydonlari BO'LMASLIGI kerak."""

    recommendation = _recommendation_with_nutrients()
    payload = recommendation.to_dict()
    for key in payload:
        lowered = key.lower()
        if any(token in lowered for token in ("nitrogen", "potassium")):
            assert "kg" not in lowered
            assert "rate" not in lowered


def test_phosphorus_recommendation_remains_quantitative():
    recommendation = _recommendation_with_nutrients()
    assert recommendation.required_p2o5_high > 0
    assert recommendation.product_kg_ha_high > 0
    assert recommendation.total_product_kg_high > 0


def test_recommendation_includes_type_amount_timing_method():
    recommendation = _recommendation_with_nutrients()
    assert recommendation.product_name_uz
    assert recommendation.product_kg_ha_low >= 0
    assert recommendation.application_stage_uz
    assert recommendation.application_method_uz


def test_growth_stage_reduces_late_season_dose():
    early = _recommendation_with_nutrients(growth_stage_key="ekish_oldi")
    late = _recommendation_with_nutrients(growth_stage_key="gullash")
    assert late.required_p2o5_high < early.required_p2o5_high


def test_previous_fertilizer_credit_reduces_dose():
    none_applied = _recommendation_with_nutrients(previous_fertilizer_key="yoq")
    heavy = _recommendation_with_nutrients(previous_fertilizer_key="yuqori")
    assert heavy.required_p2o5_high < none_applied.required_p2o5_high


# ---------------------------------------------------------------------------
# PDF: ikkala qurilma ma'lumoti
# ---------------------------------------------------------------------------


def _build_pdf(demo_id: str, artifact) -> str:
    scenario = build_scenario(demo_id)
    analyzer = AnalyzerReading(**scenario["analyzer"], source=DataSource.DEMO)
    sensor = SensorReading(**scenario["sensor"], source=DataSource.DEMO)
    fusion = fuse(analyzer, sensor, artifact=artifact)
    recommendation = build_recommendation(
        olsen_p_mg_kg=fusion.estimate.olsen_p_mg_kg,
        status_key=fusion.estimate.status_key,
        status_label_uz=fusion.estimate.status_label_uz,
        ph=fusion.ph, ec_ds_m=fusion.ec_ds_m, moisture_pct=fusion.soil_moisture_percent,
        crop_key=scenario["field"]["crop_key"],
        field_area_ha=scenario["field"]["field_area_ha"],
        target_yield_t_ha=scenario["field"]["target_yield_t_ha"],
        fertilizer_key=scenario["field"]["fertilizer_key"],
        model_confidence=fusion.confidence,
        within_calibration=fusion.estimate.within_calibration,
        growth_stage_key=scenario["field"]["growth_stage_key"],
        nitrogen=fusion.nitrogen, potassium=fusion.potassium,
    )
    explanations = build_explanations(
        fusion.estimate, recommendation, fusion.ph, fusion.ec_ds_m, fusion.soil_moisture_percent
    )
    pdf = build_pdf_report(
        estimate=fusion.estimate,
        recommendation=recommendation,
        measurements={
            "red": analyzer.red, "green": analyzer.green, "blue": analyzer.blue,
            "reaction_time_sec": analyzer.reaction_time_sec,
            "sample_temperature_c": analyzer.analyzer_temperature_c,
            "ph": fusion.ph, "ec_ds_m": fusion.ec_ds_m,
            "moisture_pct": fusion.soil_moisture_percent,
            "soil_temperature_c": fusion.soil_temperature_c,
        },
        explanations=explanations,
        model_info=model_summary(artifact),
        fusion=fusion,
        device_info={
            "analyzer_device_id": analyzer.analyzer_device_id,
            "cartridge_batch_id": analyzer.cartridge_batch_id,
            "sensor_device_id": sensor.sensor_device_id,
            "sensor_measured_at": str(sensor.measured_at)[:19],
            "region": scenario["field"]["region"],
            "previous_crop": scenario["field"]["previous_crop"],
        },
    )
    assert pdf.startswith(b"%PDF")
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)


def test_pdf_contains_both_device_datasets(artifact):
    text = _build_pdf("demo_low", artifact)
    assert "AgroIQ portativ fosfor analizatori" in text
    assert "Universal tuproq sensori" in text
    assert "AGROIQ-P-001" in text
    assert "SOIL-001" in text


def test_pdf_labels_sensor_p_as_not_olsen(artifact):
    text = _build_pdf("demo_low", artifact)
    assert "Sensor P indikatori" in text
    assert "Olsen-P EMAS" in text


def test_pdf_contains_nk_screening_disclaimer(artifact):
    text = _build_pdf("demo_low", artifact)
    assert "Azot" in text
    assert "Kaliy" in text
    assert "SKRINING" in text


def test_pdf_reports_disagreement_scenario(artifact):
    text = _build_pdf("demo_disagreement", artifact)
    assert "tafovut" in text
    assert "asosiy qiymat sifatida ishlatildi" in text
    assert "P_INDICATORS_DISAGREE" in text


def test_pdf_contains_provenance_and_confidence(artifact):
    text = _build_pdf("demo_medium", artifact)
    assert "Ma'lumot manbalari" in text
    assert "ishonchliligi" in text


def test_pdf_generated_for_every_scenario(artifact):
    for scenario in all_scenarios():
        text = _build_pdf(scenario["demo_id"], artifact)
        assert "AgroIQ" in text and len(text) > 500


# ---------------------------------------------------------------------------
# Streamlit Cloud mosligi
# ---------------------------------------------------------------------------


def _requirement_names(path: Path) -> list[str]:
    """Faylni o'qib, faqat haqiqiy paket nomlarini qaytaradi (izohlarsiz)."""

    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for separator in (">=", "==", "<=", "~=", ">", "<", "["):
            if separator in line:
                line = line.split(separator, 1)[0]
        names.append(line.strip().lower())
    return names


def test_main_requirements_have_no_hardware_dependencies():
    """Asosiy talablarda apparat paketlari BO'LMASLIGI kerak (izohlar hisobga olinmaydi)."""

    installed = _requirement_names(PROJECT_ROOT / "requirements.txt")
    for forbidden in ("pyserial", "pymodbus", "serial", "bluetooth", "pybluez", "pywin32"):
        assert forbidden not in installed, f"'{forbidden}' bulut deploy'ini buzadi"


def test_edge_requirements_are_separate():
    """Gateway bog'liqliklari alohida faylda va asosiy ro'yxatga o'rnatilmaydi."""

    edge = PROJECT_ROOT / "edge_gateway" / "requirements-edge.txt"
    assert edge.exists()
    edge_names = _requirement_names(edge)
    assert "pymodbus" in edge_names
    assert "fastapi" in edge_names

    main_names = _requirement_names(PROJECT_ROOT / "requirements.txt")
    # Apparat paketlari asosiy ro'yxatga o'tib ketmaganini tekshiramiz.
    assert not (set(edge_names) & {"pymodbus", "pyserial"}) & set(main_names)


def test_app_does_not_import_hardware_libraries():
    tree = ast.parse(APP_SOURCE)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("serial", "pymodbus", "bluetooth", "usb", "edge_gateway"):
        assert forbidden not in imported


def test_src_modules_do_not_import_hardware_libraries():
    for path in (PROJECT_ROOT / "src").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("import serial", "import pymodbus", "from serial", "from pymodbus"):
            assert forbidden not in source, f"{path.name}: '{forbidden}'"


def test_no_absolute_paths_in_source():
    """Absolyut yo'llar bulut muhitida ishlamaydi."""

    for path in list((PROJECT_ROOT / "src").glob("*.py")) + [PROJECT_ROOT / "app.py"]:
        source = path.read_text(encoding="utf-8")
        for forbidden in ("/home/", "C:\\\\", "/Users/"):
            assert forbidden not in source, f"{path.name}: absolyut yo'l '{forbidden}'"


def test_no_hardcoded_secrets():
    for path in list((PROJECT_ROOT / "src").glob("*.py")) + [PROJECT_ROOT / "app.py"]:
        source = path.read_text(encoding="utf-8").lower()
        for forbidden in ("api_key =", "apikey =", "password =", "secret =", "bearer sk-"):
            assert forbidden not in source, f"{path.name}: maxfiy ma'lumot '{forbidden}'"


def test_api_token_is_only_in_session_state():
    """Token faqat sessiya xotirasida — faylga yozilmaydi."""

    assert 'key="api_token"' in APP_SOURCE
    assert "type=\"password\"" in APP_SOURCE
    for forbidden in ("open(token", "write(token", "json.dump(token"):
        assert forbidden not in APP_SOURCE


def test_edge_gateway_mock_works_without_hardware():
    """Mock sensor hech qanday apparat yoki qo'shimcha kutubxonasiz ishlashi kerak."""

    from edge_gateway.mock_sensor import read_mock

    payload = read_mock(seed=1)
    assert payload.device_id
    assert payload.ph is not None
    assert 3.0 <= payload.ph <= 10.0


def test_modbus_reader_imports_without_hardware_library():
    """pymodbus yo'q bo'lsa ham modul import qilinishi va xato bermasligi kerak."""

    from edge_gateway import modbus_reader

    info = modbus_reader.describe()
    assert info["write_supported"] is False
    assert isinstance(info["modbus_available"], bool)


def test_modbus_read_raises_friendly_error_when_unavailable():
    from edge_gateway.modbus_reader import (
        MODBUS_AVAILABLE,
        ModbusUnavailableError,
        read_modbus,
    )

    if MODBUS_AVAILABLE:
        pytest.skip("pymodbus o'rnatilgan — bu test faqat kutubxonasiz muhit uchun.")
    with pytest.raises(ModbusUnavailableError) as excinfo:
        read_modbus()
    assert "requirements-edge" in str(excinfo.value)


def test_gateway_payload_matches_platform_schema():
    """Gateway sxemasi platforma sxemasi bilan mos bo'lishi kerak."""

    from edge_gateway.mock_sensor import read_mock
    from src.device_integration import parse_gateway_payload

    payload = read_mock(seed=3).model_dump(mode="json")
    result = parse_gateway_payload(payload)
    assert result.ok is True, result.message_uz


def test_config_files_exist_and_are_valid_json():
    import json

    for name in (
        "crop_profiles.json", "phosphorus_thresholds.json", "fertilizer_products.json",
        "sensor_thresholds.json", "device_profiles.json", "nutrient_calibration.json",
        "recommendation_rules.json",
    ):
        path = PROJECT_ROOT / "config" / name
        assert path.exists(), f"{name} yo'q"
        json.loads(path.read_text(encoding="utf-8"))


def test_version_is_bumped():
    from src import __version__

    assert __version__ == "0.2.1"


# ---------------------------------------------------------------------------
# v0.2.1 — UI tuzilmasi (yon panel, yuqori chiziq, footer)
# ---------------------------------------------------------------------------


def test_sidebar_contains_only_branding():
    """Yon panelda faqat logotip va qisqa brend matni qolishi kerak."""

    sidebar_block = APP_SOURCE[
        APP_SOURCE.index("with st.sidebar:") : APP_SOURCE.index("if model_error:")
    ]
    assert "render_logo" in sidebar_block
    # Ko'chirilgan bloklar yon panelda BO'LMASLIGI kerak.
    for removed in ("Ulanish holati", "status_pill", "__version__", "President AI Award",
                    "DEMO_MODEL_BANNER", "st.markdown(\"---\")"):
        assert removed not in sidebar_block, f"yon panelda qoldi: {removed}"


def test_demo_disclosure_moved_to_top_strip_not_removed():
    """Demo ma'lumot haqidagi ochiq ma'lumot yo'qolmasligi kerak — u ko'chirildi."""

    assert "TOP_STRIP_DEMO_TEXT" in APP_SOURCE
    assert "top_info_strip(" in APP_SOURCE
    assert "sintetik ma'lumotlar asosida prototip" in APP_SOURCE


def test_footer_is_rendered_with_version():
    main_flow = APP_SOURCE[APP_SOURCE.index("def main()") :]
    assert "footer(__app_name__, __version__)" in main_flow


def test_phosphorus_chart_has_no_plotly_legend_or_axis_title():
    """Legenda va o'q sarlavhasi HTML ga ko'chirilgan — to'qnashuv bo'lishi mumkin emas."""

    from src.data_validation import load_thresholds
    from src.ui_components import phosphorus_gauge

    figure = phosphorus_gauge(12.0, load_thresholds(), 2.0)
    assert figure.layout.showlegend is False
    assert figure.layout.xaxis.title.text is None
    for trace in figure.data:
        assert trace.showlegend is False


def test_phosphorus_chart_bottom_margin_fits_ticks():
    from src.data_validation import load_thresholds
    from src.ui_components import phosphorus_gauge

    figure = phosphorus_gauge(12.0, load_thresholds(), 2.0)
    # Faqat tik yozuvlari qolgani uchun pastki chekka yetarli.
    assert figure.layout.margin.b >= 30
    assert figure.layout.height <= 170


def test_ui_exposes_new_layout_components():
    from src import ui_components

    for name in ("top_info_strip", "footer", "page_header",
                 "phosphorus_result_panel", "phosphorus_scale_legend"):
        assert hasattr(ui_components, name), f"{name} yo'q"
