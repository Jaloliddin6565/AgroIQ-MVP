"""Apparatsiz ishlaydigan soxta sensor — namoyish va sinov uchun.

Bu modul hech qanday tashqi bog'liqlikni talab qilmaydi (faqat standart kutubxona
va pydantic). Uni ishlatib, butun ma'lumot zanjirini haqiqiy sensorsiz sinash mumkin.
"""

from __future__ import annotations

import math
import random
from datetime import datetime

from edge_gateway.schemas import TASHKENT_TZ, SensorPayload

#: Kun davomidagi tabiiy o'zgarishlarni taqlid qilish uchun bazaviy qiymatlar.
_BASE = {
    "nitrogen_indicator": 45.0,
    "phosphorus_indicator": 19.0,
    "potassium_indicator": 168.0,
    "ph": 7.7,
    "ec_ds_m": 1.9,
    "soil_temperature_c": 24.0,
    "soil_moisture_percent": 22.0,
}


def read_mock(device_id: str = "SOIL-001", seed: int | None = None) -> SensorPayload:
    """Realistik soxta o'lchov qaytaradi.

    Harorat va namlik sutkalik siklga ega — bu grafiklarni jonli ko'rsatadi.
    `seed` berilganda natija takrorlanuvchi bo'ladi (testlar uchun).
    """

    rng = random.Random(seed)
    now = datetime.now(TASHKENT_TZ)

    # Sutkalik sikl: harorat kunduzi yuqori, namlik esa aksincha.
    hour_angle = (now.hour + now.minute / 60.0) / 24.0 * 2 * math.pi
    temperature_swing = 4.0 * math.sin(hour_angle - math.pi / 2)
    moisture_swing = -2.0 * math.sin(hour_angle - math.pi / 2)

    def jitter(key: str, spread: float, offset: float = 0.0) -> float:
        return round(_BASE[key] + offset + rng.gauss(0.0, spread), 2)

    return SensorPayload(
        device_id=device_id,
        timestamp=now,
        source="mock_sensor",
        nitrogen_indicator=max(0.0, jitter("nitrogen_indicator", 4.0)),
        phosphorus_indicator=max(0.0, jitter("phosphorus_indicator", 2.0)),
        potassium_indicator=max(0.0, jitter("potassium_indicator", 12.0)),
        ph=min(10.0, max(3.0, jitter("ph", 0.08))),
        ec_ds_m=max(0.0, jitter("ec_ds_m", 0.15)),
        soil_temperature_c=jitter("soil_temperature_c", 0.5, temperature_swing),
        soil_moisture_percent=min(
            100.0, max(0.0, jitter("soil_moisture_percent", 0.8, moisture_swing))
        ),
        quality_flags=[],
    )
