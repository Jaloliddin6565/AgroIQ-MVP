"""Edge Gateway ma'lumot sxemalari.

Ushbu modul asosiy Streamlit ilovasidan MUSTAQIL — u faqat lokal gateway
kompyuterida ishlaydi va bulutga joylashtirilmaydi.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field, field_validator

TASHKENT_TZ = timezone(timedelta(hours=5))


class SensorPayload(BaseModel):
    """Gateway qaytaradigan validatsiyalangan o'lchov.

    Bu sxema AgroIQ platformasidagi `GatewayResponse` bilan bir xil bo'lishi kerak.
    Qiymat oralig'idan chiqsa, xato qaytariladi — noto'g'ri ma'lumot uzatilmaydi.
    """

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

    @field_validator("timestamp")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=TASHKENT_TZ)
        return value


class HealthResponse(BaseModel):
    """Gateway holati."""

    status: str = "ok"
    mode: str = "mock"
    device_id: str = "SOIL-001"
    gateway_version: str = "0.2.0"
    server_time: datetime = Field(default_factory=lambda: datetime.now(TASHKENT_TZ))
    modbus_available: bool = False
