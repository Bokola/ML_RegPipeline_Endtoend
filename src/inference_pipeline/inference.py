"""
Inference pipeline for Housing Regression MLE.

- Takes RAW input data (same schema as holdout.csv).
- Applies preprocessing + feature engineering using saved encoders.
- Aligns features with training.
- Returns predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing + feature engineering helpers
from src.feature_pipeline.preprocess import clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_date_features, drop_unused_columns

# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

# Load training feature columns (strict schema from training dataset)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None


# ----------------------------
# Core inference function
# ----------------------------
def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    # Step 1: Preprocess raw input
    df = clean_and_merge(input_df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    # Step 2: Feature engineering
    if "date" in df.columns:
        df = add_date_features(df)

    # Step 3: Encodings
    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors="ignore")

    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        df["city_full_encoded"] = target_encoder.transform(df["city_full"])
        df = df.drop(columns=["city_full"], errors="ignore")

    # Drop leakage columns
    df, _ = drop_unused_columns(df.copy(), df.copy())

    # Separate actuals if present
    y_true = None
    if "price" in df.columns:
        y_true = df["price"]
        df = df.drop(columns=["price"])

    # Align schema
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Predict
    model = load(model_path)
    preds = model.predict(df)

    out = df.copy()
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true.values

    return out


# ----------------------------
# CLI entrypoint
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on new housing data (raw CSV)."
    )
    parser.add_argument("--input", required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model path")
    parser.add_argument("--freq_encoder", default=str(DEFAULT_FREQ_ENCODER))
    parser.add_argument("--target_encoder", default=str(DEFAULT_TARGET_ENCODER))

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
