"""Kolorimetrik xususiyatlarni hosil qilish (feature engineering).

Fosforni baholash modeli FAQAT optik (rang) va o'lchov sharoiti xususiyatlaridan
foydalanadi. pH, EC, namlik, ekin turi va maqsadli hosildorlik bu yerda
QATNASHMAYDI — ular faqat tavsiya bosqichida ishlatiladi.

Fizik asos: molibden-ko'k usulida fosfat ionlari ko'k rangli kompleks hosil qiladi.
Konsentratsiya oshgani sari namuna qizil yorug'likni ko'proq yutadi, ya'ni qizil
kanal qiymati pasayadi. Shu sababli asosiy prediktor sifatida qizil kanaldan
hisoblangan absorbsiya indeksi ishlatiladi (Beer-Lambert qonuniga o'xshash).
"""

from __future__ import annotations

import colorsys

import numpy as np
import pandas as pd

# Model kirishida ishlatiladigan xom ustunlar.
RAW_INPUT_COLUMNS: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "reaction_time_sec",
    "sample_temperature_c",
)

# Hosil qilinadigan xususiyatlar ro'yxati (model pipeline ustunlari).
FEATURE_COLUMNS: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "red_norm",
    "green_norm",
    "blue_norm",
    "ratio_r_g",
    "ratio_r_b",
    "ratio_g_b",
    "hue",
    "saturation",
    "value",
    "brightness",
    "absorbance_red",
    "absorbance_green",
    "absorbance_blue",
    "blue_red_index",
    "green_red_index",
    "chroma",
    "reaction_time_sec",
    "log_reaction_time",
    "sample_temperature_c",
    "time_temp_index",
)

# Foydalanuvchiga tushunarli xususiyat nomlari (grafiklar uchun).
FEATURE_LABELS_UZ: dict[str, str] = {
    "red": "Qizil kanal (R)",
    "green": "Yashil kanal (G)",
    "blue": "Ko'k kanal (B)",
    "red_norm": "Normallashgan qizil",
    "green_norm": "Normallashgan yashil",
    "blue_norm": "Normallashgan ko'k",
    "ratio_r_g": "R/G nisbati",
    "ratio_r_b": "R/B nisbati",
    "ratio_g_b": "G/B nisbati",
    "hue": "Rang toni (Hue)",
    "saturation": "To'yinganlik",
    "value": "Yorqinlik (V)",
    "brightness": "O'rtacha yorqinlik",
    "absorbance_red": "Qizil kanal absorbsiyasi",
    "absorbance_green": "Yashil kanal absorbsiyasi",
    "absorbance_blue": "Ko'k kanal absorbsiyasi",
    "blue_red_index": "Ko'k-qizil indeksi",
    "green_red_index": "Yashil-qizil indeksi",
    "chroma": "Rang intensivligi (chroma)",
    "reaction_time_sec": "Reaksiya vaqti (sek)",
    "log_reaction_time": "Reaksiya vaqti (log)",
    "sample_temperature_c": "Harorat (°C)",
    "time_temp_index": "Vaqt-harorat indeksi",
}

_EPS = 1e-6


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, _EPS)


def _absorbance(channel: np.ndarray) -> np.ndarray:
    """Beer-Lambert uslubidagi absorbsiya indeksi: A = -log10(I / I0)."""

    intensity = np.clip(channel, 0.0, 255.0)
    return -np.log10(np.maximum((intensity + 1.0) / 256.0, _EPS))


def _hsv(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hues, saturations, values = [], [], []
    for r, g, b in zip(red, green, blue, strict=True):
        h, s, v = colorsys.rgb_to_hsv(
            float(np.clip(r, 0, 255)) / 255.0,
            float(np.clip(g, 0, 255)) / 255.0,
            float(np.clip(b, 0, 255)) / 255.0,
        )
        hues.append(h * 360.0)
        saturations.append(s)
        values.append(v)
    return np.asarray(hues), np.asarray(saturations), np.asarray(values)


def build_color_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Xom o'lchovlardan model uchun xususiyatlar jadvalini quradi.

    Args:
        frame: kamida `RAW_INPUT_COLUMNS` ustunlariga ega DataFrame.

    Returns:
        `FEATURE_COLUMNS` tartibidagi yangi DataFrame.
    """

    missing = [column for column in RAW_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Xususiyat hisoblash uchun ustunlar yetishmayapti: " + ", ".join(missing))

    red = np.asarray(frame["red"], dtype=float)
    green = np.asarray(frame["green"], dtype=float)
    blue = np.asarray(frame["blue"], dtype=float)
    reaction_time = np.asarray(frame["reaction_time_sec"], dtype=float)
    temperature = np.asarray(frame["sample_temperature_c"], dtype=float)

    total = red + green + blue
    hue, saturation, value = _hsv(red, green, blue)

    features = pd.DataFrame(index=frame.index)
    features["red"] = red
    features["green"] = green
    features["blue"] = blue
    features["red_norm"] = _safe_divide(red, total)
    features["green_norm"] = _safe_divide(green, total)
    features["blue_norm"] = _safe_divide(blue, total)
    features["ratio_r_g"] = _safe_divide(red, green)
    features["ratio_r_b"] = _safe_divide(red, blue)
    features["ratio_g_b"] = _safe_divide(green, blue)
    features["hue"] = hue
    features["saturation"] = saturation
    features["value"] = value
    features["brightness"] = total / 3.0
    features["absorbance_red"] = _absorbance(red)
    features["absorbance_green"] = _absorbance(green)
    features["absorbance_blue"] = _absorbance(blue)
    features["blue_red_index"] = _safe_divide(blue - red, blue + red)
    features["green_red_index"] = _safe_divide(green - red, green + red)
    features["chroma"] = (np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])) / 255.0
    features["reaction_time_sec"] = reaction_time
    features["log_reaction_time"] = np.log1p(np.maximum(reaction_time, 0.0))
    features["sample_temperature_c"] = temperature
    # Reaksiya rivojlanishining sodda kinetik ko'rsatkichi: vaqt x harorat ta'siri.
    features["time_temp_index"] = np.log1p(np.maximum(reaction_time, 0.0)) * (temperature / 25.0)

    return features[list(FEATURE_COLUMNS)]


def features_from_reading(
    red: float,
    green: float,
    blue: float,
    reaction_time_sec: float,
    sample_temperature_c: float,
) -> pd.DataFrame:
    """Bitta o'lchov uchun bir qatorli xususiyatlar jadvalini qaytaradi."""

    raw = pd.DataFrame(
        [
            {
                "red": red,
                "green": green,
                "blue": blue,
                "reaction_time_sec": reaction_time_sec,
                "sample_temperature_c": sample_temperature_c,
            }
        ]
    )
    return build_color_features(raw)


def calibration_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """O'quv to'plami bo'yicha kalibrlash oralig'ini (min/max) hisoblaydi."""

    ranges: dict[str, dict[str, float]] = {}
    for column in RAW_INPUT_COLUMNS:
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(series):
                ranges[column] = {"min": float(series.min()), "max": float(series.max())}
    return ranges


def check_within_calibration(
    reading: dict[str, float], ranges: dict[str, dict[str, float]], tolerance: float = 0.05
) -> list[str]:
    """Kirish qiymatlari kalibrlash oralig'ida ekanini tekshiradi.

    Returns:
        Oraliqdan chiqqan parametrlarning o'zbekcha nomlari ro'yxati (bo'sh bo'lsa — hammasi joyida).
    """

    from src.data_validation import FIELD_LABELS_UZ

    outside: list[str] = []
    for column, bounds in ranges.items():
        if column not in reading:
            continue
        span = max(bounds["max"] - bounds["min"], _EPS)
        margin = span * tolerance
        value = float(reading[column])
        if value < bounds["min"] - margin or value > bounds["max"] + margin:
            outside.append(FIELD_LABELS_UZ.get(column, column))
    return outside
