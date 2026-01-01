# src/api/main.py

from fastapi import FastAPI
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import boto3
import os
import logging

# Import inference pipeline
from src.inference_pipeline.inference import predict
from src.batch.run_monthly import run_monthly_predictions

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("housing-api")

# -------------------------------------------------
# Config (IAM role is picked up automatically)
# -------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "housing-regression-pipeline")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

BASE_DIR = Path(".")
MODEL_LOCAL_PATH = BASE_DIR / "models/xgb_best_model.pkl"
TRAIN_FE_LOCAL_PATH = BASE_DIR / "data/processed/feature_engineered_train.csv"

MODEL_S3_KEY = "models/xgb_best_model.pkl"
TRAIN_FE_S3_KEY = "processed/feature_engineered_train.csv"

TRAIN_FEATURE_COLUMNS: List[str] | None = None

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def download_from_s3(key: str, local_path: Path) -> bool:
    """
    Download file from S3 if it does not exist.
    Returns True if file exists or download succeeded.
    """
    try:
        if local_path.exists():
            return True

        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading s3://{S3_BUCKET}/{key}")
        s3.download_file(S3_BUCKET, key, str(local_path))
        return True

    except Exception as e:
        logger.error(f"Failed to download {key}: {e}")
        return False


def load_training_columns() -> None:
    global TRAIN_FEATURE_COLUMNS
    if TRAIN_FE_LOCAL_PATH.exists():
        df = pd.read_csv(TRAIN_FE_LOCAL_PATH, nrows=1)
        TRAIN_FEATURE_COLUMNS = [c for c in df.columns if c != "price"]


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Housing Regression API")

# -------------------------------------------------
# Startup hook (SAFE)
# -------------------------------------------------
@app.on_event("startup")
def startup_event():
    logger.info("Starting Housing Regression API")

    model_ok = download_from_s3(MODEL_S3_KEY, MODEL_LOCAL_PATH)
    fe_ok = download_from_s3(TRAIN_FE_S3_KEY, TRAIN_FE_LOCAL_PATH)

    if fe_ok:
        load_training_columns()

    if not model_ok:
        logger.warning("Model not available at startup")
    else:
        logger.info("Model ready")

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Housing Regression API is running 🚀"}


@app.get("/health")
def health():
    status: Dict[str, Any] = {
        "status": "healthy",
        "model_present": MODEL_LOCAL_PATH.exists(),
        "bucket": S3_BUCKET,
        "region": AWS_REGION,
    }

    if not MODEL_LOCAL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not available"

    if TRAIN_FEATURE_COLUMNS:
        status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)

    return status


@app.post("/predict")
def predict_batch(data: List[dict]):
    if not MODEL_LOCAL_PATH.exists():
        return {
            "error": "Model not loaded",
            "hint": "Check S3 access and IAM task role"
        }

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    preds_df = predict(df, model_path=MODEL_LOCAL_PATH)

    response = {
        "predictions": preds_df["predicted_price"].astype(float).tolist()
    }

    if "actual_price" in preds_df.columns:
        response["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return response


@app.post("/run_batch")
def run_batch():
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": int(len(preds)),
        "output_dir": "data/predictions/"
    }


@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))

    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)

    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }
