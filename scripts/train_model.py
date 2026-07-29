#!/usr/bin/env python3
"""AgroIQ fosfor modelini o'qitish skripti.

Ishlatish:

    python scripts/train_model.py
    python scripts/train_model.py --data data/soil_samples.csv --out models/phosphorus_model.joblib

Skript real dataset topilmasa yoki unda 30 tadan kam yaroqli qator bo'lsa,
takrorlanuvchi (seed=42) sintetik demo dataset yaratadi va uni aniq belgilaydi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import DATA_DIR, DataError  # noqa: E402
from src.model_training import (  # noqa: E402
    MODEL_PATH,
    build_demo_samples,
    feature_importances,
    run_training,
)
from src.model_inference import load_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgroIQ fosfor modelini o'qitish")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "soil_samples.csv",
        help="O'quv dataseti (CSV) yo'li",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MODEL_PATH,
        help="Saqlanadigan model artefakti yo'li",
    )
    parser.add_argument(
        "--skip-demo-samples",
        action="store_true",
        help="data/demo_samples.csv faylini qayta yozmaslik",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        report = run_training(data_path=args.data, model_path=args.out)
    except DataError as exc:
        print(f"[XATO] {exc}", file=sys.stderr)
        return 1

    if not args.skip_demo_samples:
        demo_path = DATA_DIR / "demo_samples.csv"
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        build_demo_samples().to_csv(demo_path, index=False)
        print(f"Demo namunalar saqlandi: {demo_path}")

    print()
    print("=" * 66)
    print("  AgroIQ — fosforni baholash modeli o'qitildi")
    print("=" * 66)
    print(f"  Dataset turi        : {'DEMO (sintetik)' if report.dataset_kind == 'demo' else 'REAL'}")
    print(f"  Namunalar soni      : {report.n_samples} (train {report.n_train} / test {report.n_test})")
    print(f"  Bo'linish strategiya: {report.split_strategy}")
    print(f"  Tanlangan model     : {report.model_name}")
    print(f"  R²                  : {report.metrics['r2']:.4f}")
    print(f"  RMSE                : {report.metrics['rmse']:.3f} mg/kg")
    print(f"  MAE                 : {report.metrics['mae']:.3f} mg/kg")
    print(f"  Qoldiq tarqoqligi   : {report.residual_std:.3f} mg/kg")
    if report.cv_metrics:
        print(
            f"  GroupKFold CV       : R² = {report.cv_metrics['r2']:.4f}, "
            f"RMSE = {report.cv_metrics['rmse']:.3f} "
            f"({int(report.cv_metrics['n_splits'])} fold)"
        )
    print("-" * 66)
    print("  Nomzod modellar (test to'plamida):")
    for candidate in report.candidates:
        print(
            f"    - {candidate['name']:<42} R²={candidate['r2']:.4f}  "
            f"RMSE={candidate['rmse']:.3f}  MAE={candidate['mae']:.3f}"
        )

    importances = feature_importances(load_artifact(args.out)["pipeline"])
    if importances is not None:
        print("-" * 66)
        print("  Eng muhim 8 xususiyat:")
        for _, row in importances.head(8).iterrows():
            print(f"    - {row['feature']:<24} {row['importance']:.4f}")

    if report.warnings:
        print("-" * 66)
        print("  Ogohlantirishlar:")
        for warning in report.warnings:
            print(f"    ! {warning}")

    print("=" * 66)
    print(f"  Model saqlandi: {args.out}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
