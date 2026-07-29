"""Kirish ma'lumotlarini tekshirish va konfiguratsiya fayllarini yuklash.

Ushbu modul ikkita vazifani bajaradi:

1. Foydalanuvchi kiritgan o'lchov qiymatlarini (pydantic orqali) qat'iy validatsiya qiladi.
2. `config/` katalogidagi JSON konfiguratsiyalarni yuklaydi va sxemasini tekshiradi.

Barcha xato xabarlari o'zbek tilida (lotin yozuvida) qaytariladi, chunki ular
to'g'ridan-to'g'ri foydalanuvchi interfeysida ko'rsatiladi.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

REQUIRED_TRAINING_COLUMNS: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "reaction_time_sec",
    "sample_temperature_c",
    "lab_olsen_p_mg_kg",
)

MIN_VALID_TRAINING_ROWS = 30

# Sintetik demo datasetni belgilovchi ustun va qiymat. Shu marker tufayli
# generatsiya qilingan ma'lumot hech qachon real laboratoriya ma'lumoti sifatida
# ko'rsatilmaydi.
SYNTHETIC_MARKER_COLUMN = "data_source"
SYNTHETIC_MARKER_VALUE = "SYNTHETIC_DEMO"


def is_synthetic_frame(frame: pd.DataFrame) -> bool:
    """Dataset sintetik demo marker bilan belgilanganini tekshiradi."""

    if SYNTHETIC_MARKER_COLUMN not in frame.columns or frame.empty:
        return False
    values = frame[SYNTHETIC_MARKER_COLUMN].astype(str).str.strip().str.upper()
    return bool((values == SYNTHETIC_MARKER_VALUE).any())


class ConfigError(RuntimeError):
    """Konfiguratsiya fayli yo'q yoki noto'g'ri formatda bo'lganda ko'tariladi."""


class DataError(RuntimeError):
    """Ma'lumotlar to'plami bilan bog'liq muammolarda ko'tariladi."""


# ---------------------------------------------------------------------------
# Kirish modellari (pydantic)
# ---------------------------------------------------------------------------


class ColorimetricReading(BaseModel):
    """Kolorimetrik o'lchov natijasi — fosforni baholash modelining yagona kirishi."""

    model_config = ConfigDict(extra="forbid")

    red: float = Field(..., ge=0, le=255, description="Qizil kanal qiymati (0-255)")
    green: float = Field(..., ge=0, le=255, description="Yashil kanal qiymati (0-255)")
    blue: float = Field(..., ge=0, le=255, description="Ko'k kanal qiymati (0-255)")
    reaction_time_sec: float = Field(
        ..., ge=10, le=1800, description="Reaksiya vaqti, sekund (10-1800)"
    )
    sample_temperature_c: float = Field(
        ..., ge=0, le=60, description="Namuna harorati, °C (0-60)"
    )


class SoilContext(BaseModel):
    """Tavsiya bosqichida ishlatiladigan tuproq va dala parametrlari.

    DIQQAT: bu qiymatlar fosfor konsentratsiyasini bashorat qilishda ISHLATILMAYDI.
    Ular faqat agronomik tavsiya va ogohlantirishlar uchun xizmat qiladi.
    """

    model_config = ConfigDict(extra="forbid")

    ph: float = Field(..., ge=3.0, le=10.0, description="Tuproq pH (3.0-10.0)")
    ec_ds_m: float = Field(..., ge=0.0, le=30.0, description="Elektr o'tkazuvchanlik, dS/m")
    moisture_pct: float = Field(..., ge=0.0, le=100.0, description="Namlik, %")
    crop_key: str = Field(..., min_length=1, description="Ekin kaliti")
    field_area_ha: float = Field(..., gt=0, le=100000, description="Dala maydoni, ga")
    target_yield_t_ha: float = Field(..., gt=0, le=50, description="Maqsadli hosildorlik, t/ga")
    fertilizer_key: str = Field(..., min_length=1, description="O'g'it mahsuloti kaliti")

    @field_validator("crop_key", "fertilizer_key")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("qiymat bo'sh bo'lishi mumkin emas")
        return cleaned


class AnalysisRequest(BaseModel):
    """To'liq tahlil so'rovi: kolorimetriya + tuproq/ekin konteksti."""

    model_config = ConfigDict(extra="forbid")

    colorimetry: ColorimetricReading
    soil: SoilContext
    sample_label: str = Field(default="Namuna", max_length=120)


# Foydalanuvchiga tushunarli maydon nomlari (xato xabarlarida ishlatiladi).
FIELD_LABELS_UZ: dict[str, str] = {
    "red": "Qizil (R)",
    "green": "Yashil (G)",
    "blue": "Ko'k (B)",
    "reaction_time_sec": "Reaksiya vaqti",
    "sample_temperature_c": "Namuna harorati",
    "ph": "pH",
    "ec_ds_m": "Elektr o'tkazuvchanlik (EC)",
    "moisture_pct": "Namlik",
    "crop_key": "Ekin turi",
    "field_area_ha": "Dala maydoni",
    "target_yield_t_ha": "Maqsadli hosildorlik",
    "fertilizer_key": "O'g'it mahsuloti",
    "sample_label": "Namuna nomi",
}

_RANGE_HINTS_UZ: dict[str, str] = {
    "red": "0 dan 255 gacha bo'lishi kerak",
    "green": "0 dan 255 gacha bo'lishi kerak",
    "blue": "0 dan 255 gacha bo'lishi kerak",
    "reaction_time_sec": "10 dan 1800 sekundgacha bo'lishi kerak",
    "sample_temperature_c": "0 dan 60 °C gacha bo'lishi kerak",
    "ph": "3.0 dan 10.0 gacha bo'lishi kerak",
    "ec_ds_m": "manfiy bo'lishi mumkin emas (0-30 dS/m)",
    "moisture_pct": "0 dan 100 % gacha bo'lishi kerak",
    "field_area_ha": "musbat son bo'lishi kerak",
    "target_yield_t_ha": "musbat son bo'lishi kerak",
}


def humanize_validation_error(error: ValidationError) -> list[str]:
    """Pydantic xatolarini o'zbekcha, foydalanuvchiga tushunarli matnga aylantiradi."""

    messages: list[str] = []
    for item in error.errors():
        location = [str(part) for part in item.get("loc", ()) if isinstance(part, str)]
        field = location[-1] if location else "kirish"
        label = FIELD_LABELS_UZ.get(field, field)
        hint = _RANGE_HINTS_UZ.get(field)
        if hint:
            messages.append(f"{label}: {hint}.")
        elif item.get("type") == "missing":
            messages.append(f"{label}: qiymat kiritilmagan.")
        else:
            messages.append(f"{label}: qiymat noto'g'ri.")
    # Takrorlanuvchi xabarlarni tartibni saqlagan holda olib tashlash.
    return list(dict.fromkeys(messages))


def validate_analysis_request(payload: dict[str, Any]) -> tuple[AnalysisRequest | None, list[str]]:
    """Xom lug'atni tekshiradi va (model, xatolar) juftligini qaytaradi."""

    try:
        return AnalysisRequest.model_validate(payload), []
    except ValidationError as exc:
        return None, humanize_validation_error(exc)


# ---------------------------------------------------------------------------
# Konfiguratsiya yuklash
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Konfiguratsiya fayli topilmadi: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"'{path.name}' faylida JSON formati buzilgan ({exc.lineno}-qator). "
            "Faylni tuzatib, dasturni qayta ishga tushiring."
        ) from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"'{path.name}' fayli obyekt (JSON dictionary) bo'lishi kerak.")
    return payload


def _require_keys(payload: dict[str, Any], keys: Iterable[str], filename: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ConfigError(
            f"'{filename}' faylida majburiy bo'lim(lar) yo'q: {', '.join(missing)}."
        )


@lru_cache(maxsize=1)
def load_thresholds(config_dir: str | None = None) -> dict[str, Any]:
    """Fosfor sinflari chegaralarini yuklaydi va tekshiradi."""

    base = Path(config_dir) if config_dir else CONFIG_DIR
    payload = _read_json(base / "phosphorus_thresholds.json")
    _require_keys(payload, ["classes"], "phosphorus_thresholds.json")

    classes = payload["classes"]
    if not isinstance(classes, list) or not classes:
        raise ConfigError("'phosphorus_thresholds.json': 'classes' bo'sh bo'lmagan ro'yxat bo'lishi kerak.")

    for item in classes:
        _require_keys(item, ["key", "label_uz", "min", "max"], "phosphorus_thresholds.json")
        if float(item["min"]) >= float(item["max"]):
            raise ConfigError(
                f"'phosphorus_thresholds.json': '{item['key']}' sinfida min >= max."
            )
    payload["classes"] = sorted(classes, key=lambda item: float(item["min"]))
    return payload


@lru_cache(maxsize=1)
def load_crop_profiles(config_dir: str | None = None) -> dict[str, Any]:
    """Ekin profillarini yuklaydi va tekshiradi."""

    base = Path(config_dir) if config_dir else CONFIG_DIR
    payload = _read_json(base / "crop_profiles.json")
    _require_keys(
        payload,
        ["crops", "status_multipliers", "soil_condition_rules", "global_ph_adjustment"],
        "crop_profiles.json",
    )
    if not isinstance(payload["crops"], list) or not payload["crops"]:
        raise ConfigError("'crop_profiles.json': 'crops' bo'sh bo'lmagan ro'yxat bo'lishi kerak.")
    for crop in payload["crops"]:
        _require_keys(
            crop,
            ["key", "name_uz", "p2o5_removal_kg_per_ton", "default_target_yield"],
            "crop_profiles.json",
        )
        if float(crop["p2o5_removal_kg_per_ton"]) <= 0:
            raise ConfigError(
                f"'crop_profiles.json': '{crop['key']}' uchun P2O5 olib chiqish qiymati musbat bo'lishi kerak."
            )
    return payload


@lru_cache(maxsize=1)
def load_fertilizer_products(config_dir: str | None = None) -> dict[str, Any]:
    """O'g'it mahsulotlari tarkibini yuklaydi va tekshiradi."""

    base = Path(config_dir) if config_dir else CONFIG_DIR
    payload = _read_json(base / "fertilizer_products.json")
    _require_keys(payload, ["products"], "fertilizer_products.json")
    if not isinstance(payload["products"], list) or not payload["products"]:
        raise ConfigError("'fertilizer_products.json': 'products' bo'sh bo'lmagan ro'yxat bo'lishi kerak.")
    for product in payload["products"]:
        _require_keys(product, ["key", "name_uz", "p2o5_fraction"], "fertilizer_products.json")
        fraction = float(product["p2o5_fraction"])
        if not 0 < fraction <= 1:
            raise ConfigError(
                f"'fertilizer_products.json': '{product['key']}' uchun P2O5 ulushi 0 va 1 orasida bo'lishi kerak."
            )
    return payload


def clear_config_cache() -> None:
    """Konfiguratsiya keshini tozalaydi (testlar va qayta yuklash uchun)."""

    load_thresholds.cache_clear()
    load_crop_profiles.cache_clear()
    load_fertilizer_products.cache_clear()


def get_crop(crop_key: str, profiles: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ekin profilini kalit bo'yicha qaytaradi."""

    profiles = profiles or load_crop_profiles()
    for crop in profiles["crops"]:
        if crop["key"] == crop_key:
            return crop
    raise ConfigError(f"Ekin topilmadi: '{crop_key}'. Konfiguratsiyani tekshiring.")


def get_product(product_key: str, products: dict[str, Any] | None = None) -> dict[str, Any]:
    """O'g'it mahsulotini kalit bo'yicha qaytaradi."""

    products = products or load_fertilizer_products()
    for product in products["products"]:
        if product["key"] == product_key:
            return product
    raise ConfigError(f"O'g'it mahsuloti topilmadi: '{product_key}'. Konfiguratsiyani tekshiring.")


# ---------------------------------------------------------------------------
# Ma'lumotlar to'plamini tekshirish
# ---------------------------------------------------------------------------


def validate_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """O'quv ma'lumotlarini tozalaydi va ogohlantirishlar ro'yxatini qaytaradi.

    Qaytariladi: (yaroqli qatorlar, ogohlantirishlar).
    """

    warnings: list[str] = []
    missing = [column for column in REQUIRED_TRAINING_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(
            "Ma'lumotlar to'plamida majburiy ustunlar yo'q: " + ", ".join(missing)
        )

    numeric = frame.copy()
    for column in REQUIRED_TRAINING_COLUMNS:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")  # type: ignore[index]

    before = len(numeric)
    numeric = numeric.dropna(subset=list(REQUIRED_TRAINING_COLUMNS))
    if len(numeric) < before:
        warnings.append(f"{before - len(numeric)} ta qator to'liq bo'lmagani uchun chiqarib tashlandi.")

    bounds = {
        "red": (0, 255),
        "green": (0, 255),
        "blue": (0, 255),
        "reaction_time_sec": (1, 3600),
        "sample_temperature_c": (-5, 70),
        "lab_olsen_p_mg_kg": (0, 500),
    }
    mask = pd.Series(True, index=numeric.index)
    for column, (low, high) in bounds.items():
        mask &= numeric[column].between(low, high)
    dropped = int((~mask).sum())
    if dropped:
        warnings.append(f"{dropped} ta qator qiymatlari ruxsat etilgan oraliqdan chiqqani uchun olib tashlandi.")
    numeric = numeric.loc[mask]

    return numeric.reset_index(drop=True), warnings


def load_training_data(path: Path | None = None) -> tuple[pd.DataFrame, list[str], bool]:
    """O'quv datasetini yuklaydi.

    Qaytariladi: (dataframe, ogohlantirishlar, yetarli_ma'lumot_bormi).
    Fayl bo'lmasa yoki yaroqli qatorlar 30 tadan kam bo'lsa, uchinchi qiymat False bo'ladi.
    """

    path = path or (DATA_DIR / "soil_samples.csv")
    if not path.exists():
        return pd.DataFrame(), [f"'{path.name}' topilmadi — demo (sintetik) ma'lumot ishlatiladi."], False

    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - CSV xatolarining barcha turlari
        return pd.DataFrame(), [f"'{path.name}' o'qib bo'lmadi: {exc}"], False

    try:
        cleaned, warnings = validate_training_frame(raw)
    except DataError as exc:
        return pd.DataFrame(), [str(exc)], False

    sufficient = len(cleaned) >= MIN_VALID_TRAINING_ROWS
    if not sufficient:
        warnings.append(
            f"Yaroqli qatorlar soni {len(cleaned)} ta — minimal talab {MIN_VALID_TRAINING_ROWS} ta. "
            "Demo (sintetik) ma'lumot ishlatiladi."
        )
    return cleaned, warnings, sufficient
