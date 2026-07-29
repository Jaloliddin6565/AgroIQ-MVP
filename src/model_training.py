"""Fosforni baholash modelini o'qitish va saqlash.

Modul uchta vazifani bajaradi:

1. Real dataset bo'lmasa — takrorlanuvchi (seed=42) sintetik demo dataset generatsiyasi.
2. Bir nechta modelni (chiziqli, polinomial, Random Forest) taqqoslash.
3. Eng yaxshi modelni metrikalar va kalibrlash oralig'i bilan birga saqlash.

Muhim: model FAQAT kolorimetrik xususiyatlardan foydalanadi. pH, EC, namlik,
ekin turi yoki maqsadli hosildorlik bu bosqichda ISHLATILMAYDI.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.data_validation import DATA_DIR, MODELS_DIR, is_synthetic_frame, load_training_data
from src.feature_engineering import (
    FEATURE_COLUMNS,
    RAW_INPUT_COLUMNS,
    build_color_features,
    calibration_ranges,
)

RANDOM_STATE = 42
MODEL_PATH = MODELS_DIR / "phosphorus_model.joblib"

# --- Sintetik generatorning fizik parametrlari (molibden-ko'k usuli) -------
_ABS_COEF_RED = 0.0300  # qizil kanal yutilish koeffitsienti (mg/kg^-1)
_ABS_COEF_GREEN = 0.0135
_ABS_COEF_BLUE = 0.0040
_BASELINE_ABS = 0.02
_RATE_25C = 1.0 / 240.0  # reaksiya tezlik konstantasi 25 °C da (1/sek)
_Q10 = 1.9  # har 10 °C da tezlikning ortishi


@dataclass
class ModelReport:
    """O'qitish natijasi haqidagi to'liq hisobot."""

    model_name: str
    metrics: dict[str, float]
    candidates: list[dict[str, Any]]
    dataset_kind: str  # "real" yoki "demo"
    n_samples: int
    n_train: int
    n_test: int
    split_strategy: str
    feature_columns: list[str]
    calibration: dict[str, dict[str, float]]
    target_range: dict[str, float]
    residual_std: float
    warnings: list[str] = field(default_factory=list)
    trained_at: str = ""
    cv_metrics: dict[str, float] = field(default_factory=dict)
    # Heteroskedastik noaniqlik modeli: sigma(pred) ~ intercept + slope * pred
    residual_intercept: float = 0.0
    residual_slope: float = 0.0
    test_actual: list[float] = field(default_factory=list)
    test_predicted: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sintetik demo dataset
# ---------------------------------------------------------------------------


def simulate_color(
    true_p_mg_kg: float,
    reaction_time_sec: float,
    temperature_c: float,
) -> tuple[float, float, float]:
    """Fosfor konsentratsiyasidan kutilayotgan RGB qiymatlarini hisoblaydi (shovqinsiz).

    Beer-Lambert qonuni + sodda reaksiya kinetikasi. Sintetik dataset va demo
    namunalar shu bitta fizik modeldan foydalanadi.
    """

    rate = _RATE_25C * (_Q10 ** ((temperature_c - 25.0) / 10.0))
    development = 1.0 - np.exp(-rate * reaction_time_sec)
    effective_p = true_p_mg_kg * development

    channels = []
    for coefficient in (_ABS_COEF_RED, _ABS_COEF_GREEN, _ABS_COEF_BLUE):
        absorbance = _BASELINE_ABS + coefficient * effective_p
        channels.append(float(np.clip(255.0 * (10.0**-absorbance), 0.0, 255.0)))
    return channels[0], channels[1], channels[2]


# Namoyish uchun tayyorlangan uchta ssenariy (12-bo'lim: "Demo rejimi").
DEMO_SCENARIOS: list[dict[str, Any]] = [
    {
        "demo_id": "demo_low",
        "scenario_uz": "1-ssenariy — Paxta, past fosfor",
        "description_uz": (
            "Ishqoriy tuproqda fosfor darajasi past. Model to'liq o'g'itlash me'yorini tavsiya qiladi."
        ),
        "target_p": 6.5,
        "reaction_time_sec": 600.0,
        "sample_temperature_c": 24.0,
        "ph": 8.1,
        "ec_ds_m": 1.8,
        "moisture_pct": 21.0,
        "crop_key": "paxta",
        "field_area_ha": 12.0,
        "target_yield_t_ha": 3.5,
        "fertilizer_key": "ammofos",
    },
    {
        "demo_id": "demo_medium",
        "scenario_uz": "2-ssenariy — Bug'doy, o'rtacha fosfor",
        "description_uz": (
            "Fosfor o'rtacha, ammo elektr o'tkazuvchanlik yuqoriroq — sho'rlanish ogohlantirishi ishlaydi."
        ),
        "target_p": 14.0,
        "reaction_time_sec": 540.0,
        "sample_temperature_c": 22.5,
        "ph": 7.6,
        "ec_ds_m": 5.4,
        "moisture_pct": 17.0,
        "crop_key": "bugdoy",
        "field_area_ha": 25.0,
        "target_yield_t_ha": 5.5,
        "fertilizer_key": "ammofos",
    },
    {
        "demo_id": "demo_high",
        "scenario_uz": "3-ssenariy — Yuqori fosfor",
        "description_uz": (
            "Fosfor zaxirasi yuqori — qo'shimcha fosforli o'g'it tavsiya etilmaydi."
        ),
        "target_p": 38.0,
        "reaction_time_sec": 660.0,
        "sample_temperature_c": 26.0,
        "ph": 7.3,
        "ec_ds_m": 1.4,
        "moisture_pct": 26.0,
        "crop_key": "paxta",
        "field_area_ha": 8.0,
        "target_yield_t_ha": 4.0,
        "fertilizer_key": "superfosfat",
    },
]


def build_demo_samples() -> pd.DataFrame:
    """Demo ssenariylarini to'liq jadval ko'rinishida (RGB bilan) qaytaradi."""

    rows: list[dict[str, Any]] = []
    for scenario in DEMO_SCENARIOS:
        red, green, blue = simulate_color(
            scenario["target_p"],
            scenario["reaction_time_sec"],
            scenario["sample_temperature_c"],
        )
        row = {key: value for key, value in scenario.items() if key != "target_p"}
        row.update(
            {
                "red": round(red, 1),
                "green": round(green, 1),
                "blue": round(blue, 1),
                "expected_p_mg_kg": scenario["target_p"],
                "data_source": "SYNTHETIC_DEMO",
            }
        )
        rows.append(row)
    columns = [
        "demo_id",
        "scenario_uz",
        "description_uz",
        "red",
        "green",
        "blue",
        "reaction_time_sec",
        "sample_temperature_c",
        "ph",
        "ec_ds_m",
        "moisture_pct",
        "crop_key",
        "field_area_ha",
        "target_yield_t_ha",
        "fertilizer_key",
        "expected_p_mg_kg",
        "data_source",
    ]
    return pd.DataFrame(rows)[columns]


def generate_demo_dataset(
    n_fields: int = 48,
    samples_per_field: tuple[int, int] = (4, 7),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Takrorlanuvchi sintetik dataset yaratadi.

    Rang qiymatlari Beer-Lambert qonuni va sodda reaksiya kinetikasi asosida
    modellashtiriladi: fosfor konsentratsiyasi ortishi bilan namuna ko'kroq
    bo'ladi va qizil kanal qiymati pasayadi.

    DIQQAT: bu REAL laboratoriya ma'lumoti EMAS. Faqat prototipni namoyish
    qilish uchun ishlatiladi.
    """

    rng = np.random.default_rng(random_state)
    regions = ["Sirdaryo", "Jizzax", "Buxoro", "Farg'ona", "Qashqadaryo"]
    soil_types = ["Bo'z tuproq", "O'tloqi-bo'z", "Sur-qo'ng'ir", "Allyuvial"]

    rows: list[dict[str, Any]] = []
    sample_counter = 1

    for field_index in range(n_fields):
        field_id = f"F{field_index + 1:03d}"
        region = regions[field_index % len(regions)]
        soil_type = soil_types[field_index % len(soil_types)]
        # Dala darajasidagi o'rtacha fosfor (lognormal taqsimot).
        # Qurilma o'lchash oralig'i (2-50 mg/kg); undan yuqorida suyultirish talab qilinadi.
        field_mean_p = float(np.clip(rng.lognormal(mean=2.35, sigma=0.55), 2.0, 48.0))
        # Har bir dalada o'lchov sharoitining (yoritilish, sensor siljishi) o'ziga xosligi.
        field_illumination = float(rng.normal(0.0, 4.0))
        # Tuproq matritsasi ta'siri (temir, kremniy birikmalari) — dala darajasida barqaror.
        field_matrix_bias = float(rng.normal(1.0, 0.06))

        n_samples = int(rng.integers(samples_per_field[0], samples_per_field[1] + 1))
        for _ in range(n_samples):
            true_p = float(np.clip(field_mean_p * rng.lognormal(0.0, 0.18), 1.0, 60.0))
            reaction_time = float(rng.uniform(120.0, 900.0))
            temperature = float(np.clip(rng.normal(24.0, 5.0), 8.0, 42.0))

            # Namunaga xos matritsa interferensiyasi rang rivojlanishini o'zgartiradi.
            matrix_factor = field_matrix_bias * float(rng.normal(1.0, 0.05))
            red, green, blue = simulate_color(
                true_p * matrix_factor, reaction_time, temperature
            )

            noise = rng.normal(0.0, 3.0, size=3)
            red = float(np.clip(red + noise[0] + field_illumination, 0, 255))
            green = float(np.clip(green + noise[1] + field_illumination, 0, 255))
            blue = float(np.clip(blue + noise[2] + field_illumination, 0, 255))

            # Laboratoriya Olsen-P o'lchovining o'z takrorlanuvchanlik xatoligi (~8%).
            lab_p = float(np.clip(true_p * rng.normal(1.0, 0.08), 0.5, 80.0))

            rows.append(
                {
                    "sample_id": f"S{sample_counter:04d}",
                    "field_id": field_id,
                    "red": round(red, 1),
                    "green": round(green, 1),
                    "blue": round(blue, 1),
                    "reaction_time_sec": round(reaction_time, 0),
                    "sample_temperature_c": round(temperature, 1),
                    "lab_olsen_p_mg_kg": round(lab_p, 2),
                    "region": region,
                    "soil_type": soil_type,
                    "measurement_date": (
                        pd.Timestamp("2025-03-01") + pd.Timedelta(days=int(rng.integers(0, 120)))
                    ).strftime("%Y-%m-%d"),
                    "data_source": "SYNTHETIC_DEMO",
                }
            )
            sample_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model nomzodlari
# ---------------------------------------------------------------------------


def build_candidates() -> dict[str, Pipeline]:
    """Taqqoslanadigan model pipeline'larini qaytaradi."""

    return {
        "Chiziqli regressiya": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Polinomial regressiya (2-daraja)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        max_depth=None,
                        min_samples_leaf=2,
                        max_features=0.6,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }


def fit_residual_scale(
    predictions: np.ndarray, residuals: np.ndarray
) -> tuple[float, float]:
    """Noaniqlikning bashorat kattaligiga bog'liqligini baholaydi.

    Kolorimetrik o'lchovda xatolikning katta qismi ko'paytiruvchi (nisbiy) xarakterga
    ega: matritsa interferensiyasi va laboratoriya takrorlanuvchanligi konsentratsiya
    bilan birga o'sadi. Shuning uchun |qoldiq| ~ a + b * bashorat chizig'i moslanadi va
    normal taqsimot uchun sigma = 1.2533 * E|qoldiq| deb olinadi.

    Returns:
        (intercept, slope) — ikkalasi ham manfiy bo'lmagan.
    """

    absolute = np.abs(residuals)
    if len(absolute) < 5:
        return float(np.std(residuals) if len(residuals) > 1 else 1.0), 0.0

    design = np.column_stack([np.ones_like(predictions), predictions])
    coefficients, *_ = np.linalg.lstsq(design, absolute, rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])

    scale = np.sqrt(np.pi / 2.0)  # E|X| -> sigma o'tkazish koeffitsienti
    intercept *= scale
    slope *= scale

    # Konservativ chegaralar: manfiy bo'lmasin va minimal pol saqlansin.
    slope = max(slope, 0.0)
    floor = 0.25 * float(np.std(residuals))
    intercept = max(intercept, floor, 0.1)
    return intercept, slope


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _split(
    features: pd.DataFrame, target: pd.Series, groups: pd.Series | None
) -> tuple[np.ndarray, np.ndarray, str]:
    """Train/test indekslarini qaytaradi. Iloji bo'lsa dala bo'yicha guruhlaydi."""

    if groups is not None and groups.nunique() >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(features, target, groups=groups))
        return train_idx, test_idx, "GroupShuffleSplit (field_id bo'yicha, test 25%)"

    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.25, random_state=RANDOM_STATE
    )
    return train_idx, test_idx, "Oddiy tasodifiy train/test bo'linishi (test 25%)"


def _group_cv_metrics(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series, groups: pd.Series | None
) -> dict[str, float]:
    """GroupKFold cross-validation metrikalari (mavjud bo'lsa)."""

    if groups is None or groups.nunique() < 5:
        return {}
    n_splits = int(min(5, groups.nunique()))
    cv = GroupKFold(n_splits=n_splits)
    predictions = np.zeros(len(target), dtype=float)
    for train_idx, test_idx in cv.split(features, target, groups=groups):
        fold_model = clone(pipeline)
        fold_model.fit(features.iloc[train_idx], target.iloc[train_idx])
        predictions[test_idx] = fold_model.predict(features.iloc[test_idx])
    metrics = _score(target.to_numpy(), predictions)
    metrics["n_splits"] = float(n_splits)
    return metrics


def train_model(
    frame: pd.DataFrame,
    dataset_kind: str,
    warnings: list[str] | None = None,
) -> tuple[Pipeline, ModelReport]:
    """Modellarni o'qitadi, taqqoslaydi va eng mosini tanlaydi."""

    warnings = list(warnings or [])
    features = build_color_features(frame)
    target = pd.to_numeric(frame["lab_olsen_p_mg_kg"], errors="coerce").astype(float)
    groups = frame["field_id"].astype(str) if "field_id" in frame.columns else None

    train_idx, test_idx, split_strategy = _split(features, target, groups)
    x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

    candidates: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in build_candidates().items():
        pipeline.fit(x_train, y_train)
        metrics = _score(y_test.to_numpy(), pipeline.predict(x_test))
        candidates.append({"name": name, **metrics})
        fitted[name] = pipeline

    best = max(candidates, key=lambda item: item["r2"])
    rf = next(item for item in candidates if item["name"] == "Random Forest")

    # Random Forest noaniqlikni baholashni qo'llab-quvvatlagani uchun,
    # natijasi eng yaxshisiga yaqin bo'lsa (0.02 R² ichida) unga ustunlik beriladi.
    if rf["r2"] >= best["r2"] - 0.02:
        selected_name = "Random Forest"
        if rf["name"] != best["name"]:
            warnings.append(
                "Random Forest tanlandi: natijasi eng yaxshi modelga yaqin va noaniqlikni "
                "baholash imkonini beradi."
            )
    else:
        selected_name = best["name"]

    selected_metrics = next(item for item in candidates if item["name"] == selected_name)
    selected_pipeline = fitted[selected_name]

    cv_metrics = _group_cv_metrics(build_candidates()[selected_name], features, target, groups)

    # Yakuniy model butun datasetda qayta o'qitiladi (metrikalar test to'plamidan olingan).
    final_pipeline = build_candidates()[selected_name]
    final_pipeline.fit(features, target)

    test_predictions = np.asarray(selected_pipeline.predict(x_test), dtype=float)
    residuals = y_test.to_numpy() - test_predictions
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else float(selected_metrics["rmse"])
    residual_intercept, residual_slope = fit_residual_scale(test_predictions, residuals)

    report = ModelReport(
        model_name=selected_name,
        metrics={
            "r2": selected_metrics["r2"],
            "rmse": selected_metrics["rmse"],
            "mae": selected_metrics["mae"],
        },
        candidates=candidates,
        dataset_kind=dataset_kind,
        n_samples=int(len(frame)),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        split_strategy=split_strategy,
        feature_columns=list(FEATURE_COLUMNS),
        calibration=calibration_ranges(frame),
        target_range={
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
        },
        residual_std=residual_std,
        warnings=warnings,
        trained_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        cv_metrics=cv_metrics,
        residual_intercept=residual_intercept,
        residual_slope=residual_slope,
        test_actual=[float(value) for value in y_test.to_numpy()],
        test_predicted=[float(value) for value in test_predictions],
    )
    return final_pipeline, report


def feature_importances(pipeline: Pipeline) -> pd.DataFrame | None:
    """Model qo'llab-quvvatlasa, xususiyat muhimligini qaytaradi."""

    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(importances) != len(FEATURE_COLUMNS):
        return None
    return (
        pd.DataFrame({"feature": list(FEATURE_COLUMNS), "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_model(pipeline: Pipeline, report: ModelReport, path: Path | None = None) -> Path:
    """Pipeline va hisobotni bitta joblib artefaktiga saqlaydi."""

    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "report": report,
        "feature_columns": list(FEATURE_COLUMNS),
        "raw_columns": list(RAW_INPUT_COLUMNS),
        "random_state": RANDOM_STATE,
        "sklearn_python": platform.python_version(),
        "format_version": 1,
    }
    joblib.dump(artifact, path)
    return path


def run_training(
    data_path: Path | None = None,
    model_path: Path | None = None,
    demo_csv_path: Path | None = None,
) -> ModelReport:
    """To'liq o'qitish jarayoni: dataset yuklash → o'qitish → saqlash."""

    frame, warnings, sufficient = load_training_data(data_path)
    if sufficient and not is_synthetic_frame(frame):
        dataset_kind = "real"
    elif sufficient:
        # Fayl mavjud, lekin sintetik marker bilan belgilangan — demo sifatida qoladi.
        dataset_kind = "demo"
        warnings.append(
            "Dataset sintetik demo marker bilan belgilangan (data_source=SYNTHETIC_DEMO)."
        )
    else:
        dataset_kind = "demo"
        frame = generate_demo_dataset()
        warnings.append(
            "Sintetik demo dataset yaratildi (seed=42). Bu real laboratoriya ma'lumoti emas."
        )
        target_csv = demo_csv_path or (DATA_DIR / "soil_samples.csv")
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target_csv, index=False)

    pipeline, report = train_model(frame, dataset_kind=dataset_kind, warnings=warnings)
    save_model(pipeline, report, model_path)
    return report
