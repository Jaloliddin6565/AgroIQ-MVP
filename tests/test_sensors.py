"""Universal sensor, qurilma integratsiyasi va sxemalar testlari (v0.2.0)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.data_validation import (
    ConfigError,
    is_quantitative_enabled,
    load_device_profiles,
    load_nutrient_calibration,
    load_sensor_thresholds,
)
from src.device_integration import (
    example_sensor_file,
    fetch_from_gateway,
    generate_mock_payload,
    get_device_profile,
    parse_gateway_payload,
    parse_uploaded_sensor_file,
    sensor_reading_to_payload,
    check_connection,
)
from src.sensor_schemas import (
    AnalyzerReading,
    DataSource,
    GatewayResponse,
    QualityFlag,
    SensorReading,
    ensure_aware,
    is_in_future,
    is_stale,
    utc_now,
)


# ---------------------------------------------------------------------------
# Sensor sxemasi validatsiyasi
# ---------------------------------------------------------------------------


def test_sensor_reading_accepts_valid_values():
    reading = SensorReading(
        nitrogen_indicator=45.0, phosphorus_indicator=18.0, potassium_indicator=170.0,
        ph=7.6, ec_ds_m=2.0, soil_temperature_c=22.0, soil_moisture_percent=24.0,
        sensor_device_id="SOIL-001",
    )
    assert reading.ph == 7.6
    assert reading.has_soil_context() is True
    assert reading.has_npk() is True


def test_sensor_reading_allows_partial_data():
    """Sensor barcha parametrlarni bermasligi mumkin — bu xato emas."""

    reading = SensorReading(ph=7.2)
    assert reading.has_soil_context() is True
    assert reading.has_npk() is False
    assert reading.nitrogen_indicator is None


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("ph", 15.0), ("ph", 1.0),
        ("ec_ds_m", -1.0), ("ec_ds_m", 50.0),
        ("soil_moisture_percent", -5.0), ("soil_moisture_percent", 150.0),
        ("soil_temperature_c", -60.0), ("soil_temperature_c", 120.0),
        ("nitrogen_indicator", -1.0), ("potassium_indicator", 99999.0),
    ],
)
def test_sensor_reading_rejects_out_of_range(field, bad_value):
    with pytest.raises(ValidationError):
        SensorReading(**{field: bad_value})


def test_sensor_device_id_is_trimmed_and_blank_becomes_none():
    assert SensorReading(sensor_device_id="  SOIL-9  ").sensor_device_id == "SOIL-9"
    assert SensorReading(sensor_device_id="   ").sensor_device_id is None


def test_analyzer_reading_requires_valid_rgb():
    with pytest.raises(ValidationError):
        AnalyzerReading(red=300, green=100, blue=100, reaction_time_sec=600,
                        analyzer_temperature_c=24)


# ---------------------------------------------------------------------------
# Gateway javobi validatsiyasi
# ---------------------------------------------------------------------------


def test_gateway_response_parses_documented_example():
    """Hujjatlashtirilgan API javobi namunasi qabul qilinishi kerak."""

    payload = {
        "device_id": "SOIL-001",
        "timestamp": "2026-07-30T10:00:00+05:00",
        "source": "modbus_gateway",
        "nitrogen_indicator": 42.0,
        "phosphorus_indicator": 18.0,
        "potassium_indicator": 165.0,
        "ph": 7.8,
        "ec_ds_m": 1.9,
        "soil_temperature_c": 29.4,
        "soil_moisture_percent": 21.7,
        "quality_flags": [],
    }
    result = parse_gateway_payload(payload)
    assert result.ok is True
    assert result.reading is not None
    assert result.reading.sensor_device_id == "SOIL-001"
    assert result.reading.ph == 7.8
    assert result.reading.source == DataSource.API


def test_gateway_response_rejects_invalid_payload():
    result = parse_gateway_payload({"device_id": "X", "timestamp": "emas", "ph": 99})
    assert result.ok is False
    assert "sxemaga mos kelmadi" in result.message_uz
    assert result.reading is None


def test_gateway_response_ignores_unknown_fields():
    """Kelajakdagi qo'shimcha maydonlar javobni buzmasligi kerak."""

    payload = generate_mock_payload(seed=1)
    payload["future_field"] = "kutilmagan"
    assert parse_gateway_payload(payload).ok is True


def test_gateway_response_missing_device_id_rejected():
    payload = generate_mock_payload(seed=1)
    del payload["device_id"]
    assert parse_gateway_payload(payload).ok is False


def test_gateway_to_sensor_reading_conversion():
    response = GatewayResponse(
        device_id="SOIL-7", timestamp=utc_now(), ph=7.0, ec_ds_m=1.0
    )
    reading = response.to_sensor_reading()
    assert isinstance(reading, SensorReading)
    assert reading.sensor_device_id == "SOIL-7"
    assert reading.source == DataSource.API


# ---------------------------------------------------------------------------
# Vaqt belgilari
# ---------------------------------------------------------------------------


def test_stale_timestamp_detected():
    old = utc_now() - timedelta(hours=30)
    assert is_stale(old, 24) is True


def test_fresh_timestamp_not_stale():
    recent = utc_now() - timedelta(minutes=30)
    assert is_stale(recent, 24) is False


def test_missing_timestamp_is_not_stale():
    assert is_stale(None, 24) is False


def test_future_timestamp_detected():
    future = utc_now() + timedelta(hours=2)
    assert is_in_future(future, 10) is True
    assert is_in_future(utc_now(), 10) is False


def test_naive_datetime_treated_as_utc():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware = ensure_aware(naive)
    assert aware is not None and aware.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# Mock API va ulanish
# ---------------------------------------------------------------------------


def test_mock_endpoint_returns_reading_without_network():
    result = fetch_from_gateway("mock://agroiq-edge-gateway/api/v1/readings/latest", mock_seed=5)
    assert result.ok is True
    assert result.reading is not None
    assert result.reading.ph is not None


def test_mock_payload_is_reproducible_with_seed():
    first = generate_mock_payload(seed=99)
    second = generate_mock_payload(seed=99)
    for key in ("ph", "ec_ds_m", "nitrogen_indicator", "potassium_indicator"):
        assert first[key] == second[key]


def test_mock_profiles_differ():
    normal = generate_mock_payload(seed=3, profile="normal")
    saline = generate_mock_payload(seed=3, profile="alkaline_saline")
    assert saline["ec_ds_m"] > normal["ec_ds_m"]
    assert saline["ph"] > normal["ph"]


def test_check_connection_reports_success_for_mock():
    outcome = check_connection("mock://test")
    assert outcome.ok is True
    assert "muvaffaqiyatli" in outcome.message_uz


def test_empty_endpoint_rejected():
    assert fetch_from_gateway("").ok is False


def test_invalid_scheme_rejected():
    result = fetch_from_gateway("ftp://sensor.local/data")
    assert result.ok is False
    assert "boshlanishi kerak" in result.message_uz


def test_api_unavailable_falls_back_gracefully():
    """API ishlamasa, ilova traceback ko'rsatmasdan tushunarli xabar berishi kerak."""

    result = fetch_from_gateway("http://127.0.0.1:9/none", timeout=1.0)
    assert result.ok is False
    assert result.reading is None
    assert "Traceback" not in result.message_uz
    assert any(word in result.message_uz for word in ("ulanib", "javob bermadi", "xatolik"))


# ---------------------------------------------------------------------------
# Fayl yuklash
# ---------------------------------------------------------------------------


def test_json_upload_parsed():
    content = json.dumps(generate_mock_payload(seed=11)).encode("utf-8")
    result = parse_uploaded_sensor_file(content, "sensor.json")
    assert result.ok is True
    assert result.reading is not None
    assert result.reading.source == DataSource.FILE


def test_json_list_uses_last_entry():
    first = generate_mock_payload(seed=1)
    last = generate_mock_payload(seed=2)
    last["ph"] = 6.4
    content = json.dumps([first, last]).encode("utf-8")
    result = parse_uploaded_sensor_file(content, "readings.json")
    assert result.ok is True
    assert result.reading.ph == 6.4


def test_csv_upload_parsed():
    content = example_sensor_file("csv")
    result = parse_uploaded_sensor_file(content, "sensor.csv")
    assert result.ok is True
    assert result.reading is not None


def test_broken_json_gives_friendly_error():
    result = parse_uploaded_sensor_file(b"{bu json emas", "sensor.json")
    assert result.ok is False
    assert "JSON" in result.message_uz
    assert "Traceback" not in result.message_uz


def test_empty_file_rejected():
    assert parse_uploaded_sensor_file(b"", "sensor.json").ok is False


def test_unsupported_extension_rejected():
    result = parse_uploaded_sensor_file(b"data", "sensor.xlsx")
    assert result.ok is False
    assert ".json" in result.message_uz


def test_sensor_reading_roundtrip_to_payload():
    reading = SensorReading(ph=7.1, ec_ds_m=1.4, sensor_device_id="SOIL-5", measured_at=utc_now())
    payload = sensor_reading_to_payload(reading)
    assert payload["device_id"] == "SOIL-5"
    assert parse_gateway_payload(payload).ok is True


# ---------------------------------------------------------------------------
# Konfiguratsiya
# ---------------------------------------------------------------------------


def test_sensor_thresholds_load_and_have_bands():
    thresholds = load_sensor_thresholds()
    for parameter in ("ph", "ec_ds_m", "soil_moisture_percent", "soil_temperature_c"):
        assert thresholds[parameter]["bands"]


def test_npk_thresholds_marked_screening_only():
    thresholds = load_sensor_thresholds()
    for parameter in ("nitrogen_indicator", "phosphorus_indicator", "potassium_indicator"):
        assert thresholds[parameter]["screening_only"] is True


def test_device_profiles_contain_both_devices():
    profiles = load_device_profiles()
    keys = {device["key"] for device in profiles["devices"]}
    assert {"agroiq_analyzer", "universal_soil_sensor"} <= keys


def test_only_analyzer_is_primary_phosphorus_source():
    profiles = load_device_profiles()
    primary = [
        device["key"]
        for device in profiles["devices"]
        if device.get("is_primary_phosphorus_source")
    ]
    assert primary == ["agroiq_analyzer"]


def test_unknown_device_profile_raises():
    with pytest.raises(ConfigError):
        get_device_profile("mavjud_emas")


def test_invalid_sensor_config_raises(tmp_path):
    (tmp_path / "sensor_thresholds.json").write_text("{ buzilgan", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_sensor_thresholds(str(tmp_path))


def test_sensor_config_missing_section_raises(tmp_path):
    (tmp_path / "sensor_thresholds.json").write_text(json.dumps({"ph": {}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_sensor_thresholds(str(tmp_path))


# ---------------------------------------------------------------------------
# N/K kalibrlash bayrog'i — eng muhim ilmiy cheklov
# ---------------------------------------------------------------------------


def test_nitrogen_and_potassium_quantitative_disabled_by_default():
    """Kalibrlashsiz N va K uchun miqdoriy tavsiya BERILMASLIGI kerak."""

    assert is_quantitative_enabled("nitrogen") is False
    assert is_quantitative_enabled("potassium") is False


def test_phosphorus_quantitative_enabled():
    assert is_quantitative_enabled("phosphorus") is True


def test_calibration_flag_can_be_enabled_via_config(tmp_path):
    """Kelajakda kalibrlash tasdiqlansa, bayroq konfiguratsiya orqali yoqilishi mumkin."""

    payload = {
        "phosphorus": {"quantitative_enabled": True},
        "nitrogen": {"quantitative_enabled": True, "requires_lab_confirmation": False},
        "potassium": {"quantitative_enabled": False},
    }
    (tmp_path / "nutrient_calibration.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    calibration = load_nutrient_calibration(str(tmp_path))
    assert is_quantitative_enabled("nitrogen", calibration) is True
    assert is_quantitative_enabled("potassium", calibration) is False


def test_calibration_config_missing_flag_raises(tmp_path):
    payload = {"phosphorus": {}, "nitrogen": {}, "potassium": {}}
    (tmp_path / "nutrient_calibration.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_nutrient_calibration(str(tmp_path))


def test_nk_calibration_is_not_validated_in_current_config():
    calibration = load_nutrient_calibration()
    for nutrient in ("nitrogen", "potassium"):
        assert calibration[nutrient]["calibration"]["validated"] is False
        assert calibration[nutrient]["requires_lab_confirmation"] is True


# ---------------------------------------------------------------------------
# Sifat bayroqlari ro'yxati
# ---------------------------------------------------------------------------


def test_all_required_quality_flags_exist():
    required = {
        "COLOR_OUTSIDE_CALIBRATION_RANGE", "PH_HIGH", "EC_HIGH", "MOISTURE_LOW",
        "SENSOR_DATA_STALE", "DEVICE_ID_MISSING", "P_INDICATORS_DISAGREE",
        "N_REQUIRES_LAB_CONFIRMATION", "K_REQUIRES_LAB_CONFIRMATION",
        "DEMO_DATA", "SYNTHETIC_MODEL",
    }
    available = {flag.value for flag in QualityFlag}
    assert required <= available


def test_every_flag_has_uzbek_description():
    from src.sensor_schemas import QUALITY_FLAG_LABELS_UZ

    for flag in QualityFlag:
        assert flag.value in QUALITY_FLAG_LABELS_UZ, f"{flag.value} uchun matn yo'q"
        meta = QUALITY_FLAG_LABELS_UZ[flag.value]
        assert meta["title"] and meta["message"]
        assert meta["level"] in {"info", "caution", "attention"}
