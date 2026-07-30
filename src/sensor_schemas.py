"""Universal tuproq sensori va birlashtirilgan tahlil yozuvi uchun sxemalar.

Ushbu modul v0.2.0 da qo'shildi. U ikkita ma'lumot manbasini QAT'IY ajratadi:

1. `AnalyzerReading`  — AgroIQ kolorimetrik analizatori (Olsen-P uchun ASOSIY manba).
2. `SensorReading`    — universal ko'p parametrli tuproq sensori (kontekst + NPK skrining).

Eng muhim ilmiy qoida: sensorning `phosphorus_indicator` qiymati Olsen-P EMAS.
U hech qachon kolorimetrik bahoning o'rnini bosmaydi va u bilan o'rtachalanmaydi.
Shu sababli ular alohida maydonlarda, alohida nomlar bilan saqlanadi.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Ma'lumot manbasi (provenance)
# ---------------------------------------------------------------------------


class DataSource(str, Enum):
    """Ma'lumot qayerdan kelganini belgilaydi — natijada ochiq ko'rsatiladi."""

    DEMO = "demo"
    MANUAL = "manual"
    ANALYZER = "analyzer"
    SENSOR = "sensor"
    API = "api"
    FILE = "file"


DATA_SOURCE_LABELS_UZ: dict[str, str] = {
    DataSource.DEMO.value: "Demo ssenariy",
    DataSource.MANUAL.value: "Qo'lda kiritilgan",
    DataSource.ANALYZER.value: "AgroIQ analizatori",
    DataSource.SENSOR.value: "Universal sensor",
    DataSource.API.value: "Gateway API",
    DataSource.FILE.value: "Yuklangan fayl",
}


class QualityFlag(str, Enum):
    """Ma'lumot sifati va cheklovlarini belgilovchi bayroqlar."""

    COLOR_OUTSIDE_CALIBRATION_RANGE = "COLOR_OUTSIDE_CALIBRATION_RANGE"
    PH_HIGH = "PH_HIGH"
    PH_LOW = "PH_LOW"
    EC_HIGH = "EC_HIGH"
    MOISTURE_LOW = "MOISTURE_LOW"
    MOISTURE_HIGH = "MOISTURE_HIGH"
    SOIL_TEMPERATURE_EXTREME = "SOIL_TEMPERATURE_EXTREME"
    SENSOR_DATA_STALE = "SENSOR_DATA_STALE"
    SENSOR_DATA_MISSING = "SENSOR_DATA_MISSING"
    DEVICE_ID_MISSING = "DEVICE_ID_MISSING"
    P_INDICATORS_DISAGREE = "P_INDICATORS_DISAGREE"
    N_REQUIRES_LAB_CONFIRMATION = "N_REQUIRES_LAB_CONFIRMATION"
    K_REQUIRES_LAB_CONFIRMATION = "K_REQUIRES_LAB_CONFIRMATION"
    DEMO_DATA = "DEMO_DATA"
    SYNTHETIC_MODEL = "SYNTHETIC_MODEL"
    TIMESTAMP_IN_FUTURE = "TIMESTAMP_IN_FUTURE"


QUALITY_FLAG_LABELS_UZ: dict[str, dict[str, str]] = {
    QualityFlag.COLOR_OUTSIDE_CALIBRATION_RANGE.value: {
        "title": "Kalibrlash oralig'idan tashqarida",
        "message": "Kolorimetrik o'lchov model validatsiya qilingan oraliqdan chetda. Laboratoriya tasdig'i tavsiya etiladi.",
        "level": "attention",
    },
    QualityFlag.PH_HIGH.value: {
        "title": "Yuqori pH",
        "message": "Ishqoriy muhitda fosforning o'simlik tomonidan o'zlashtirilishi kamayishi mumkin.",
        "level": "caution",
    },
    QualityFlag.PH_LOW.value: {
        "title": "Past pH",
        "message": "Kislotali muhitda fosfor temir va alyuminiy birikmalari bilan bog'lanishi mumkin.",
        "level": "caution",
    },
    QualityFlag.EC_HIGH.value: {
        "title": "Yuqori EC",
        "message": "Sho'rlanish oziq moddalarning o'zlashtirilishini cheklashi mumkin.",
        "level": "attention",
    },
    QualityFlag.MOISTURE_LOW.value: {
        "title": "Past namlik",
        "message": "Quruq tuproqda o'g'itning erishi va o'zlashtirilishi cheklanadi.",
        "level": "caution",
    },
    QualityFlag.MOISTURE_HIGH.value: {
        "title": "Yuqori namlik",
        "message": "Ortiqcha nam sharoitda oziq moddalarning yuvilib ketishi ehtimoli ortadi.",
        "level": "info",
    },
    QualityFlag.SOIL_TEMPERATURE_EXTREME.value: {
        "title": "Harorat maqbul oraliqdan tashqarida",
        "message": "Tuproq harorati oziq moddalar o'zlashtirilishiga ta'sir qilishi mumkin.",
        "level": "info",
    },
    QualityFlag.SENSOR_DATA_STALE.value: {
        "title": "Sensor ma'lumoti eskirgan",
        "message": "O'lchov 24 soatdan oldin olingan. Tuproq holati o'zgargan bo'lishi mumkin.",
        "level": "caution",
    },
    QualityFlag.SENSOR_DATA_MISSING.value: {
        "title": "Sensor ma'lumoti yo'q",
        "message": "Universal sensor o'lchovlari kiritilmagan. Tahlil faqat kolorimetrik va qo'lda kiritilgan qiymatlarga asoslandi.",
        "level": "info",
    },
    QualityFlag.DEVICE_ID_MISSING.value: {
        "title": "Qurilma ID kiritilmagan",
        "message": "O'lchovni aniq qurilmaga bog'lash imkoni yo'q — kuzatuvchanlik cheklangan.",
        "level": "info",
    },
    QualityFlag.P_INDICATORS_DISAGREE.value: {
        "title": "Fosfor bo'yicha qurilmalar o'rtasida tafovut",
        "message": (
            "Fosfor bo'yicha qurilmalar o'rtasida tafovut aniqlandi. AgroIQ kolorimetrik natijasi "
            "tavsiyada asosiy qiymat sifatida ishlatildi. Laboratoriya tasdig'i tavsiya etiladi."
        ),
        "level": "attention",
    },
    QualityFlag.N_REQUIRES_LAB_CONFIRMATION.value: {
        "title": "Azot: laboratoriya tasdig'i kerak",
        "message": "Sensorning azot qiymati skrining indikatori. Miqdoriy me'yor uchun laboratoriya tahlili zarur.",
        "level": "info",
    },
    QualityFlag.K_REQUIRES_LAB_CONFIRMATION.value: {
        "title": "Kaliy: laboratoriya tasdig'i kerak",
        "message": "Sensorning kaliy qiymati skrining indikatori. Miqdoriy me'yor uchun laboratoriya tahlili zarur.",
        "level": "info",
    },
    QualityFlag.DEMO_DATA.value: {
        "title": "Demo ma'lumot",
        "message": "Ushbu tahlil namoyish uchun tayyorlangan ma'lumotga asoslangan.",
        "level": "info",
    },
    QualityFlag.SYNTHETIC_MODEL.value: {
        "title": "Sintetik ma'lumotda o'qitilgan model",
        "message": "Fosfor modeli sintetik demo ma'lumotda kalibrlangan. Dala uchun real kalibrlash zarur.",
        "level": "caution",
    },
    QualityFlag.TIMESTAMP_IN_FUTURE.value: {
        "title": "Vaqt belgisi kelajakda",
        "message": "O'lchov vaqti joriy vaqtdan keyin ko'rsatilgan — qurilma soatini tekshiring.",
        "level": "caution",
    },
}


# ---------------------------------------------------------------------------
# Qurilma o'lchovlari
# ---------------------------------------------------------------------------


class AnalyzerReading(BaseModel):
    """AgroIQ kolorimetrik analizatori o'lchovi — Olsen-P uchun ASOSIY manba."""

    model_config = ConfigDict(extra="forbid")

    red: float = Field(..., ge=0, le=255)
    green: float = Field(..., ge=0, le=255)
    blue: float = Field(..., ge=0, le=255)
    reaction_time_sec: float = Field(..., ge=10, le=1800)
    analyzer_temperature_c: float = Field(..., ge=0, le=60)
    analyzer_device_id: str | None = Field(default=None, max_length=64)
    cartridge_batch_id: str | None = Field(default=None, max_length=64)
    measured_at: datetime | None = None
    source: DataSource = DataSource.MANUAL

    @field_validator("analyzer_device_id", "cartridge_batch_id")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SensorReading(BaseModel):
    """Universal ko'p parametrli tuproq sensori o'lchovi.

    DIQQAT: `phosphorus_indicator` — bu Olsen-P EMAS. U sensorning umumiy
    yoki empirik fosfor indikatori bo'lib, kolorimetrik natijani almashtirmaydi.
    """

    model_config = ConfigDict(extra="forbid")

    nitrogen_indicator: float | None = Field(default=None, ge=0, le=2000)
    phosphorus_indicator: float | None = Field(default=None, ge=0, le=2000)
    potassium_indicator: float | None = Field(default=None, ge=0, le=5000)
    ph: float | None = Field(default=None, ge=3.0, le=10.0)
    ec_ds_m: float | None = Field(default=None, ge=0.0, le=30.0)
    soil_temperature_c: float | None = Field(default=None, ge=-20.0, le=70.0)
    soil_moisture_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    sensor_device_id: str | None = Field(default=None, max_length=64)
    measured_at: datetime | None = None
    source: DataSource = DataSource.MANUAL
    quality_flags: list[str] = Field(default_factory=list)

    @field_validator("sensor_device_id")
    @classmethod
    def _clean_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def has_soil_context(self) -> bool:
        """pH, EC va namlikning kamida bittasi mavjudmi."""

        return any(
            value is not None
            for value in (self.ph, self.ec_ds_m, self.soil_moisture_percent)
        )

    def has_npk(self) -> bool:
        return any(
            value is not None
            for value in (
                self.nitrogen_indicator,
                self.phosphorus_indicator,
                self.potassium_indicator,
            )
        )


class GatewayResponse(BaseModel):
    """Edge Gateway REST javobining sxemasi.

    Tashqi manbadan kelgan JSON shu sxema orqali tekshiriladi — noto'g'ri yoki
    to'liq bo'lmagan javob ilovani ishdan chiqarmaydi.
    """

    model_config = ConfigDict(extra="ignore")

    device_id: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime
    source: str = Field(default="modbus_gateway", max_length=64)
    nitrogen_indicator: float | None = Field(default=None, ge=0, le=2000)
    phosphorus_indicator: float | None = Field(default=None, ge=0, le=2000)
    potassium_indicator: float | None = Field(default=None, ge=0, le=5000)
    ph: float | None = Field(default=None, ge=3.0, le=10.0)
    ec_ds_m: float | None = Field(default=None, ge=0.0, le=30.0)
    soil_temperature_c: float | None = Field(default=None, ge=-20.0, le=70.0)
    soil_moisture_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    quality_flags: list[str] = Field(default_factory=list)

    def to_sensor_reading(self, source: DataSource = DataSource.API) -> SensorReading:
        """Gateway javobini ichki sensor o'lchoviga aylantiradi."""

        return SensorReading(
            nitrogen_indicator=self.nitrogen_indicator,
            phosphorus_indicator=self.phosphorus_indicator,
            potassium_indicator=self.potassium_indicator,
            ph=self.ph,
            ec_ds_m=self.ec_ds_m,
            soil_temperature_c=self.soil_temperature_c,
            soil_moisture_percent=self.soil_moisture_percent,
            sensor_device_id=self.device_id,
            measured_at=self.timestamp,
            source=source,
            quality_flags=list(self.quality_flags),
        )


class FieldContext(BaseModel):
    """Ekin va dala ma'lumotlari — tavsiya bosqichi uchun."""

    model_config = ConfigDict(extra="forbid")

    crop_key: str = Field(..., min_length=1)
    growth_stage_key: str | None = None
    region: str | None = Field(default=None, max_length=80)
    soil_texture_key: str | None = None
    field_area_ha: float = Field(..., gt=0, le=100000)
    target_yield_t_ha: float = Field(..., gt=0, le=50)
    previous_crop: str | None = Field(default=None, max_length=80)
    previous_fertilizer_key: str | None = None
    irrigation_method_key: str | None = None
    fertilizer_key: str = Field(..., min_length=1)
    fertilizer_price_uzs_per_kg: float | None = Field(default=None, ge=0, le=1_000_000)


# ---------------------------------------------------------------------------
# Birlashtirilgan tahlil yozuvi
# ---------------------------------------------------------------------------


class AnalysisRecord(BaseModel):
    """Bitta tahlilning to'liq, tekshirilgan yozuvi (provenance bilan)."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    timestamp: datetime
    data_sources: list[str] = Field(default_factory=list)

    analyzer_device_id: str | None = None
    sensor_device_id: str | None = None
    cartridge_batch_id: str | None = None

    red: float | None = None
    green: float | None = None
    blue: float | None = None
    reaction_time_sec: float | None = None
    analyzer_temperature_c: float | None = None

    estimated_olsen_p_mg_kg: float | None = None
    olsen_p_uncertainty: float | None = None

    nitrogen_indicator: float | None = None
    sensor_phosphorus_indicator: float | None = None
    potassium_indicator: float | None = None
    ph: float | None = None
    ec_ds_m: float | None = None
    soil_temperature_c: float | None = None
    soil_moisture_percent: float | None = None
    sensor_measured_at: datetime | None = None

    crop: str | None = None
    growth_stage: str | None = None
    field_area_ha: float | None = None
    target_yield: float | None = None

    recommendation_confidence: str | None = None
    quality_flags: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Yordamchi funksiyalar
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware(moment: datetime | None) -> datetime | None:
    """Vaqt zonasi bo'lmagan datetime'ni UTC deb qabul qiladi."""

    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def is_stale(
    measured_at: datetime | None,
    stale_after_hours: float,
    reference: datetime | None = None,
) -> bool:
    """O'lchov eskirganini aniqlaydi."""

    measured_at = ensure_aware(measured_at)
    if measured_at is None:
        return False
    reference = ensure_aware(reference) or utc_now()
    return (reference - measured_at) > timedelta(hours=stale_after_hours)


def is_in_future(
    measured_at: datetime | None,
    tolerance_minutes: float,
    reference: datetime | None = None,
) -> bool:
    """O'lchov vaqti kelajakda ekanini aniqlaydi (qurilma soati xatosi)."""

    measured_at = ensure_aware(measured_at)
    if measured_at is None:
        return False
    reference = ensure_aware(reference) or utc_now()
    return (measured_at - reference) > timedelta(minutes=tolerance_minutes)
