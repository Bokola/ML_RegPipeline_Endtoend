"""
Hyperparameter tuning with Optuna + MLflow.
Optimized for PyArrow/Parquet and MLflow 2.x/3.x compatibility.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq  # Added for Parquet support
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# --- MLflow Safety Import ---
try:
    import mlflow
    import mlflow.sklearn
except ImportError as e:
    print(f"⚠️ MLflow import issue detected: {e}")
    # This specifically catches the 'IS_TRACING_SDK_ONLY' error if the env is messy
    import mlflow.version
    if not hasattr(mlflow.version, 'IS_TRACING_SDK_ONLY'):
        setattr(mlflow.version, 'IS_TRACING_SDK_ONLY', False)
    import mlflow
    import mlflow.sklearn

# Use Parquet as default input
DEFAULT_TRAIN = Path("data/processed/feature_engineered_train.parquet")
DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.parquet")
DEFAULT_OUT = Path("models/xgb_best_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if not sample_frac or not (0 < sample_frac < 1):
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def _load_data(train_path: Path, eval_path: Path, sample_frac: Optional[float], random_state: int):
    # Use PyArrow to read Parquet
    train_df = pq.read_table(train_path).to_pandas()
    eval_df = pq.read_table(eval_path).to_pandas()

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    # In our feature engineering, the target was renamed/kept as 'price'
    target = "price" 
    
    # Ensure target exists
    if target not in train_df.columns:
        # Fallback if your cleaning script kept a different name
        target = train_df.columns[-1] 

    return (
        train_df.drop(columns=[target]),
        train_df[target],
        eval_df.drop(columns=[target]),
        eval_df[target],
    )


def tune_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    n_trials: int = 15,
    sample_frac: Optional[float] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "xgboost_optuna_housing",
    random_state: int = 42,
) -> Tuple[Dict, Dict]:

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Convert strings to Paths
    train_path, eval_path = Path(train_path), Path(eval_path)

    X_train, y_train, X_eval, y_eval = _load_data(
        train_path, eval_path, sample_frac, random_state
    )

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        with mlflow.start_run(nested=True):
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_eval)
            rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
            mae = float(mean_absolute_error(y_eval, preds))
            r2 = float(r2_score(y_eval, preds))

            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_model = XGBRegressor(
        **best_params,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_eval)
    best_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_eval, preds))),
        "mae": float(mean_absolute_error(y_eval, preds)),
        "r2": float(r2_score(y_eval, preds)),
    }

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)

    # ✅ Using mlflow.sklearn to avoid XGBoost-specific version conflicts
    with mlflow.start_run(run_name="best_xgb_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

    return best_params, best_metrics


if __name__ == "__main__":
    tune_model()