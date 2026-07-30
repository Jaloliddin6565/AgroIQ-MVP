"""Namoyish ssenariylari — v0.2.0 (ikkala qurilma ma'lumoti bilan).

Har bir ssenariy to'liq to'plamni beradi:
  - AgroIQ kolorimetrik analizatori o'lchovi (RGB, vaqt, harorat, qurilma ID);
  - universal tuproq sensori o'lchovi (pH, EC, namlik, harorat, NPK, ID, vaqt);
  - ekin va dala ma'lumotlari.

RGB qiymatlari `model_training.simulate_color` fizik modeli orqali maqsadli
Olsen-P dan hisoblanadi — shuning uchun ular modelning kutilgan natijasiga mos.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.model_training import simulate_color

# O'zbekiston vaqti (UTC+5)
TASHKENT_TZ = timezone(timedelta(hours=5))


def _recent(hours_ago: float = 0.5) -> datetime:
    """Yaqinda olingan o'lchov vaqti (eskirgan deb belgilanmasligi uchun)."""

    return datetime.now(timezone.utc).astimezone(TASHKENT_TZ) - timedelta(hours=hours_ago)


SCENARIOS: list[dict[str, Any]] = [
    {
        "demo_id": "demo_low",
        "title_uz": "1-ssenariy — Paxta, past fosfor",
        "summary_uz": (
            "Ishqoriy tuproqda fosfor darajasi past, sho'rlanish normal. "
            "Model to'liq fosforli o'g'itlash me'yorini tavsiya qiladi."
        ),
        "expected_uz": "Past fosfor → yuqori me'yor",
        "target_p": 6.5,
        "analyzer": {
            "reaction_time_sec": 600.0,
            "analyzer_temperature_c": 24.0,
            "analyzer_device_id": "AGROIQ-P-001",
            "cartridge_batch_id": "CRT-2026-014",
        },
        "sensor": {
            "nitrogen_indicator": 44.0,
            "phosphorus_indicator": 9.0,
            "potassium_indicator": 168.0,
            "ph": 8.1,
            "ec_ds_m": 1.8,
            "soil_temperature_c": 22.5,
            "soil_moisture_percent": 21.0,
            "sensor_device_id": "SOIL-001",
        },
        "field": {
            "crop_key": "paxta",
            "growth_stage_key": "ekish_oldi",
            "region": "Sirdaryo",
            "soil_texture_key": "orta_qumoq",
            "field_area_ha": 12.0,
            "target_yield_t_ha": 3.5,
            "previous_crop": "Bug'doy",
            "previous_fertilizer_key": "past",
            "irrigation_method_key": "egat",
            "fertilizer_key": "ammofos",
        },
    },
    {
        "demo_id": "demo_medium",
        "title_uz": "2-ssenariy — Bug'doy, o'rtacha fosfor",
        "summary_uz": (
            "Fosfor o'rtacha, ammo elektr o'tkazuvchanlik yuqoriroq — "
            "sho'rlanish ogohlantirishi va ishonchlilikning pasayishi ko'rsatiladi."
        ),
        "expected_uz": "O'rtacha fosfor + sho'rlanish ogohlantirishi",
        "target_p": 14.0,
        "analyzer": {
            "reaction_time_sec": 540.0,
            "analyzer_temperature_c": 22.5,
            "analyzer_device_id": "AGROIQ-P-002",
            "cartridge_batch_id": "CRT-2026-021",
        },
        "sensor": {
            "nitrogen_indicator": 58.0,
            "phosphorus_indicator": 17.5,
            "potassium_indicator": 205.0,
            "ph": 7.6,
            "ec_ds_m": 5.4,
            "soil_temperature_c": 19.0,
            "soil_moisture_percent": 24.0,
            "sensor_device_id": "SOIL-002",
        },
        "field": {
            "crop_key": "bugdoy",
            "growth_stage_key": "ekish_oldi",
            "region": "Jizzax",
            "soil_texture_key": "ogir_qumoq",
            "field_area_ha": 25.0,
            "target_yield_t_ha": 5.5,
            "previous_crop": "Paxta",
            "previous_fertilizer_key": "ortacha",
            "irrigation_method_key": "yomgirlatib",
            "fertilizer_key": "ammofos",
        },
    },
    {
        "demo_id": "demo_high",
        "title_uz": "3-ssenariy — Yuqori fosfor",
        "summary_uz": (
            "Fosfor zaxirasi yuqori — qo'shimcha fosforli o'g'it tavsiya etilmaydi. "
            "Ortiqcha xarajat va ekologik yukning oldi olinadi."
        ),
        "expected_uz": "Yuqori fosfor → o'g'it tavsiya etilmaydi",
        "target_p": 38.0,
        "analyzer": {
            "reaction_time_sec": 660.0,
            "analyzer_temperature_c": 26.0,
            "analyzer_device_id": "AGROIQ-P-001",
            "cartridge_batch_id": "CRT-2026-014",
        },
        "sensor": {
            "nitrogen_indicator": 72.0,
            "phosphorus_indicator": 44.0,
            "potassium_indicator": 260.0,
            "ph": 7.3,
            "ec_ds_m": 1.4,
            "soil_temperature_c": 24.0,
            "soil_moisture_percent": 26.0,
            "sensor_device_id": "SOIL-001",
        },
        "field": {
            "crop_key": "paxta",
            "growth_stage_key": "ekish_oldi",
            "region": "Farg'ona",
            "soil_texture_key": "orta_qumoq",
            "field_area_ha": 8.0,
            "target_yield_t_ha": 4.0,
            "previous_crop": "Makkajo'xori",
            "previous_fertilizer_key": "yuqori",
            "irrigation_method_key": "tomchilatib",
            "fertilizer_key": "superfosfat",
        },
    },
    {
        "demo_id": "demo_disagreement",
        "title_uz": "4-ssenariy — Qurilmalar o'rtasida tafovut",
        "summary_uz": (
            "Universal sensor fosforni yuqori ko'rsatmoqda, AgroIQ kolorimetrik tahlili esa "
            "past. Platforma qiymatlarni O'RTACHALAMAYDI — kolorimetrik natija asosiy bo'lib "
            "qoladi va laboratoriya tasdig'i so'raladi."
        ),
        "expected_uz": "Tafovut → kolorimetrik natija asosiy",
        "target_p": 7.0,
        "analyzer": {
            "reaction_time_sec": 600.0,
            "analyzer_temperature_c": 25.0,
            "analyzer_device_id": "AGROIQ-P-003",
            "cartridge_batch_id": "CRT-2026-030",
        },
        "sensor": {
            # Sensor P indikatori kolorimetrik natijadan keskin yuqori —
            # bu aynan tafovut bayrog'ini ko'rsatish uchun tanlangan.
            "nitrogen_indicator": 52.0,
            "phosphorus_indicator": 46.0,
            "potassium_indicator": 180.0,
            "ph": 7.9,
            "ec_ds_m": 2.6,
            "soil_temperature_c": 23.0,
            "soil_moisture_percent": 19.5,
            "sensor_device_id": "SOIL-003",
        },
        "field": {
            "crop_key": "paxta",
            "growth_stage_key": "ekish_oldi",
            "region": "Qashqadaryo",
            "soil_texture_key": "qumoq",
            "field_area_ha": 15.0,
            "target_yield_t_ha": 3.8,
            "previous_crop": "Paxta",
            "previous_fertilizer_key": "nomalum",
            "irrigation_method_key": "egat",
            "fertilizer_key": "ammofos",
        },
    },
]


def build_scenario(demo_id: str) -> dict[str, Any]:
    """Ssenariyni to'liq (RGB hisoblangan, vaqt belgilari yangilangan) qaytaradi."""

    for scenario in SCENARIOS:
        if scenario["demo_id"] == demo_id:
            break
    else:
        raise KeyError(f"Demo ssenariy topilmadi: '{demo_id}'")

    analyzer = dict(scenario["analyzer"])
    red, green, blue = simulate_color(
        scenario["target_p"],
        analyzer["reaction_time_sec"],
        analyzer["analyzer_temperature_c"],
    )
    analyzer.update(
        {"red": round(red, 1), "green": round(green, 1), "blue": round(blue, 1)}
    )

    sensor = dict(scenario["sensor"])
    sensor["measured_at"] = _recent(0.5)

    return {
        "demo_id": scenario["demo_id"],
        "title_uz": scenario["title_uz"],
        "summary_uz": scenario["summary_uz"],
        "expected_uz": scenario["expected_uz"],
        "target_p": scenario["target_p"],
        "analyzer": analyzer,
        "sensor": sensor,
        "field": dict(scenario["field"]),
    }


def all_scenarios() -> list[dict[str, Any]]:
    """Barcha ssenariylarni to'liq shaklda qaytaradi."""

    return [build_scenario(item["demo_id"]) for item in SCENARIOS]


def scenario_ids() -> list[str]:
    return [item["demo_id"] for item in SCENARIOS]
