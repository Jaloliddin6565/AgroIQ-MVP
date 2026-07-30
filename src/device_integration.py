"""Qurilma integratsiyasi: demo, qo'lda, API (gateway) va fayl rejimlari.

MUHIM ARXITEKTURA QARORI
------------------------
Streamlit Community Cloud'da ishlayotgan ilova foydalanuvchining lokal
kompyuteridagi RS485/ketma-ket portga TO'G'RIDAN-TO'G'RI kira olmaydi.
Shu sababli bu yerda hech qanday `pyserial` yoki Modbus bog'liqligi YO'Q.

Buning o'rniga gateway arxitekturasi qo'llaniladi:

    Universal sensor → RS485/Modbus RTU → lokal Edge Gateway → REST API → AgroIQ

Ilova faqat HTTP orqali gateway bilan gaplashadi. Apparat ulanishi ixtiyoriy:
demo, qo'lda va fayl rejimlari har doim ishlaydi.

API token faqat sessiya xotirasida saqlanadi — u hech qachon faylga yozilmaydi.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.data_validation import ConfigError, load_device_profiles
from src.sensor_schemas import (
    DataSource,
    GatewayResponse,
    SensorReading,
)

MOCK_SCHEME = "mock://"
DEFAULT_TIMEOUT = 5.0


class DeviceConnectionError(RuntimeError):
    """Qurilma yoki gateway bilan bog'lanishda xatolik."""


@dataclass
class ConnectionResult:
    """Ulanishni tekshirish yoki ma'lumot olish natijasi."""

    ok: bool
    message_uz: str
    reading: SensorReading | None = None
    raw_payload: dict[str, Any] | None = None
    latency_ms: float | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Qurilma profillari
# ---------------------------------------------------------------------------


def get_device_profile(key: str, profiles: dict[str, Any] | None = None) -> dict[str, Any]:
    """Qurilma profilini kalit bo'yicha qaytaradi."""

    profiles = profiles or load_device_profiles()
    for device in profiles["devices"]:
        if device["key"] == key:
            return device
    raise ConfigError(f"Qurilma profili topilmadi: '{key}'.")


def api_defaults(profiles: dict[str, Any] | None = None) -> dict[str, Any]:
    profiles = profiles or load_device_profiles()
    return dict(profiles.get("api_defaults", {}))


# ---------------------------------------------------------------------------
# Mock (soxta) sensor generatori — apparatsiz namoyish uchun
# ---------------------------------------------------------------------------


def generate_mock_payload(
    device_id: str = "SOIL-001",
    seed: int | None = None,
    profile: str = "normal",
) -> dict[str, Any]:
    """Gateway javobiga o'xshash realistik soxta ma'lumot yaratadi.

    `profile` qiymatlari: "normal", "alkaline_saline", "dry_low_fertility".
    Seed berilganda natija takrorlanuvchi bo'ladi.
    """

    rng = random.Random(seed)

    presets = {
        "normal": {
            "ph": (7.0, 0.25),
            "ec": (1.6, 0.3),
            "moisture": (24.0, 3.0),
            "temperature": (22.0, 2.0),
            "n": (65.0, 10.0),
            "p": (28.0, 5.0),
            "k": (190.0, 25.0),
        },
        "alkaline_saline": {
            "ph": (8.2, 0.15),
            "ec": (5.6, 0.6),
            "moisture": (18.0, 2.5),
            "temperature": (27.0, 2.0),
            "n": (38.0, 8.0),
            "p": (14.0, 4.0),
            "k": (150.0, 20.0),
        },
        "dry_low_fertility": {
            "ph": (7.9, 0.2),
            "ec": (1.2, 0.25),
            "moisture": (10.5, 1.5),
            "temperature": (30.0, 2.5),
            "n": (26.0, 6.0),
            "p": (9.0, 3.0),
            "k": (105.0, 18.0),
        },
    }
    preset = presets.get(profile, presets["normal"])

    def draw(key: str, low: float, high: float) -> float:
        mean, spread = preset[key]
        return round(min(max(rng.gauss(mean, spread), low), high), 2)

    return {
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc)
        .astimezone(timezone(timedelta(hours=5)))
        .isoformat(timespec="seconds"),
        "source": "mock_gateway",
        "nitrogen_indicator": draw("n", 0, 2000),
        "phosphorus_indicator": draw("p", 0, 2000),
        "potassium_indicator": draw("k", 0, 5000),
        "ph": draw("ph", 3.0, 10.0),
        "ec_ds_m": draw("ec", 0.0, 30.0),
        "soil_temperature_c": draw("temperature", -20.0, 70.0),
        "soil_moisture_percent": draw("moisture", 0.0, 100.0),
        "quality_flags": [],
    }


# ---------------------------------------------------------------------------
# Javobni tekshirish
# ---------------------------------------------------------------------------


def parse_gateway_payload(
    payload: dict[str, Any], source: DataSource = DataSource.API
) -> ConnectionResult:
    """Gateway javobini sxema bo'yicha tekshiradi va sensor o'lchoviga aylantiradi."""

    try:
        response = GatewayResponse.model_validate(payload)
    except ValidationError as exc:
        problems = []
        for item in exc.errors():
            location = ".".join(str(part) for part in item.get("loc", ()))
            problems.append(location or "javob")
        unique = ", ".join(dict.fromkeys(problems))
        return ConnectionResult(
            ok=False,
            message_uz=f"Gateway javobi sxemaga mos kelmadi. Muammoli maydonlar: {unique}.",
        )

    reading = response.to_sensor_reading(source=source)
    warnings: list[str] = []
    if not reading.has_soil_context():
        warnings.append("Javobda pH, EC va namlik qiymatlari yo'q.")
    if not reading.has_npk():
        warnings.append("Javobda NPK indikatorlari yo'q.")

    return ConnectionResult(
        ok=True,
        message_uz=f"Ma'lumot qabul qilindi (qurilma: {response.device_id}).",
        reading=reading,
        raw_payload=payload,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# API mijozi
# ---------------------------------------------------------------------------


def _is_mock_endpoint(endpoint: str) -> bool:
    return endpoint.strip().lower().startswith(MOCK_SCHEME)


def fetch_from_gateway(
    endpoint: str,
    token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    mock_seed: int | None = None,
    mock_profile: str = "normal",
) -> ConnectionResult:
    """Gateway'dan oxirgi o'lchovni oladi.

    `mock://` bilan boshlanuvchi manzil apparatsiz namoyish uchun ishlatiladi va
    hech qanday tarmoq so'rovi yuborilmaydi.

    Tarmoq xatolari foydalanuvchiga tushunarli o'zbekcha xabarga aylantiriladi —
    traceback hech qachon ko'rsatilmaydi.
    """

    endpoint = (endpoint or "").strip()
    if not endpoint:
        return ConnectionResult(ok=False, message_uz="API manzili kiritilmagan.")

    if _is_mock_endpoint(endpoint):
        payload = generate_mock_payload(seed=mock_seed, profile=mock_profile)
        result = parse_gateway_payload(payload, source=DataSource.API)
        result.message_uz = "Mock gateway javob berdi (apparatsiz namoyish rejimi)."
        result.latency_ms = 12.0
        return result

    if not endpoint.lower().startswith(("http://", "https://")):
        return ConnectionResult(
            ok=False,
            message_uz="API manzili 'http://', 'https://' yoki 'mock://' bilan boshlanishi kerak.",
        )

    # `requests` asosiy talablar ro'yxatida bo'lmasligi mumkin — import xatosi
    # ilovani ishdan chiqarmasligi kerak.
    try:
        import requests  # noqa: PLC0415 - ixtiyoriy bog'liqlik
    except ImportError:
        return ConnectionResult(
            ok=False,
            message_uz=(
                "HTTP kutubxonasi mavjud emas. Mock rejimidan yoki fayl yuklashdan "
                "foydalaning."
            ),
        )

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    started = datetime.now(timezone.utc)
    try:
        response = requests.get(endpoint, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - barcha tarmoq xatolari
        name = type(exc).__name__
        if "Timeout" in name:
            message = f"Gateway {timeout:.0f} soniyada javob bermadi. Tarmoqni tekshiring."
        elif "ConnectionError" in name:
            message = "Gateway'ga ulanib bo'lmadi. Manzil va tarmoqni tekshiring."
        else:
            message = "Gateway bilan bog'lanishda xatolik yuz berdi."
        return ConnectionResult(ok=False, message_uz=message)

    latency = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0

    if response.status_code == 401 or response.status_code == 403:
        return ConnectionResult(
            ok=False,
            message_uz="Kirish rad etildi (401/403). API tokenini tekshiring.",
            latency_ms=latency,
        )
    if response.status_code >= 400:
        return ConnectionResult(
            ok=False,
            message_uz=f"Gateway xatolik qaytardi (HTTP {response.status_code}).",
            latency_ms=latency,
        )

    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - JSON bo'lmagan javob
        return ConnectionResult(
            ok=False,
            message_uz="Gateway javobi JSON formatida emas.",
            latency_ms=latency,
        )

    if not isinstance(payload, dict):
        return ConnectionResult(
            ok=False,
            message_uz="Gateway javobi kutilgan JSON obyekti emas.",
            latency_ms=latency,
        )

    result = parse_gateway_payload(payload, source=DataSource.API)
    result.latency_ms = latency
    return result


def check_connection(
    endpoint: str, token: str | None = None, timeout: float = DEFAULT_TIMEOUT
) -> ConnectionResult:
    """Ulanishni tekshiradi (ma'lumotni ham oladi, lekin uni saqlamaydi)."""

    result = fetch_from_gateway(endpoint, token=token, timeout=timeout)
    if result.ok:
        latency = f" ({result.latency_ms:.0f} ms)" if result.latency_ms else ""
        result.message_uz = f"Ulanish muvaffaqiyatli{latency}. {result.message_uz}"
    return result


# ---------------------------------------------------------------------------
# Fayldan yuklash (JSON / CSV)
# ---------------------------------------------------------------------------


def parse_uploaded_sensor_file(
    content: bytes | str, filename: str = "sensor.json"
) -> ConnectionResult:
    """Yuklangan JSON yoki CSV fayldan sensor o'lchovini o'qiydi.

    CSV bo'lsa oxirgi qator ishlatiladi (eng yangi o'lchov deb qabul qilinadi).
    """

    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return ConnectionResult(
                ok=False, message_uz="Faylni o'qib bo'lmadi — u UTF-8 formatida emas."
            )
    else:
        text = content

    if not text.strip():
        return ConnectionResult(ok=False, message_uz="Yuklangan fayl bo'sh.")

    lowered = filename.lower()

    if lowered.endswith(".json"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return ConnectionResult(
                ok=False,
                message_uz=f"JSON formati buzilgan ({exc.lineno}-qator). Faylni tekshiring.",
            )
        if isinstance(payload, list):
            if not payload:
                return ConnectionResult(ok=False, message_uz="JSON ro'yxati bo'sh.")
            payload = payload[-1]
        if not isinstance(payload, dict):
            return ConnectionResult(
                ok=False, message_uz="JSON fayl obyekt yoki obyektlar ro'yxati bo'lishi kerak."
            )
        result = parse_gateway_payload(payload, source=DataSource.FILE)
        if result.ok:
            result.message_uz = f"'{filename}' faylidan ma'lumot yuklandi."
        return result

    if lowered.endswith((".csv", ".txt")):
        try:
            frame = pd.read_csv(StringIO(text))
        except Exception:  # noqa: BLE001 - CSV formatining har xil xatolari
            return ConnectionResult(
                ok=False, message_uz=f"'{filename}' faylini CSV sifatida o'qib bo'lmadi."
            )
        if frame.empty:
            return ConnectionResult(ok=False, message_uz="CSV faylda ma'lumot yo'q.")

        row = frame.iloc[-1]
        payload: dict[str, Any] = {}
        for column in frame.columns:
            value = row[column]
            if pd.isna(value):
                continue
            name = str(column).strip()
            value = value.item() if hasattr(value, "item") else value
            # CSV da ro'yxatlar matn ko'rinishida saqlanadi ("[]" yoki "a;b").
            if name == "quality_flags" and isinstance(value, str):
                cleaned = value.strip().strip("[]").replace("'", "").replace('"', "")
                value = [part.strip() for part in cleaned.split(";") if part.strip()] if cleaned else []
            payload[name] = value
        payload.setdefault("device_id", "CSV-UPLOAD")
        payload.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        payload.setdefault("source", "uploaded_file")

        result = parse_gateway_payload(payload, source=DataSource.FILE)
        if result.ok:
            result.message_uz = (
                f"'{filename}' faylidan oxirgi qator yuklandi ({len(frame)} ta qatordan)."
            )
        return result

    return ConnectionResult(
        ok=False,
        message_uz="Faqat .json va .csv fayllar qo'llab-quvvatlanadi.",
    )


def sensor_reading_to_payload(reading: SensorReading) -> dict[str, Any]:
    """Sensor o'lchovini gateway formatidagi lug'atga aylantiradi (eksport uchun)."""

    return {
        "device_id": reading.sensor_device_id or "UNKNOWN",
        "timestamp": (reading.measured_at or datetime.now(timezone.utc)).isoformat(),
        "source": reading.source.value,
        "nitrogen_indicator": reading.nitrogen_indicator,
        "phosphorus_indicator": reading.phosphorus_indicator,
        "potassium_indicator": reading.potassium_indicator,
        "ph": reading.ph,
        "ec_ds_m": reading.ec_ds_m,
        "soil_temperature_c": reading.soil_temperature_c,
        "soil_moisture_percent": reading.soil_moisture_percent,
        "quality_flags": list(reading.quality_flags),
    }


def example_sensor_file(fmt: str = "json") -> bytes:
    """Foydalanuvchi yuklab olishi mumkin bo'lgan namuna fayl."""

    payload = generate_mock_payload(seed=42)
    if fmt == "csv":
        buffer = BytesIO()
        pd.DataFrame([payload]).to_csv(buffer, index=False)
        return buffer.getvalue()
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
