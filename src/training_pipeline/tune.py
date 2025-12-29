"""
Hyperparameter tuning with Optuna + MLflow.
Fixed for circular imports and MLflow tracing SDK conflicts.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# --- MLflow Stability Fix (Monkeypatching) ---
def _safe_mlflow_import():
    """
    Handles circular imports and the IS_TRACING_SDK_ONLY error by ensuring 
    mlflow.version is correctly initialized before other imports.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except (ImportError, AttributeError):
        # Specifically fix the IS_TRACING_SDK_ONLY bug in mixed environments
        import mlflow.version
        if not hasattr(mlflow.version, 'IS_TRACING_SDK_ONLY'):
            setattr(mlflow.version, 'IS_TRACING_SDK_ONLY', False)
        import mlflow
        import mlflow.sklearn
    return mlflow

# Default Paths
DEFAULT_TRAIN = Path("data/processed/feature_engineered_train.parquet")
DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.parquet")
DEFAULT_OUT = Path("models/xgb_best_model.pkl")

def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if not sample_frac or not (0 < sample_frac < 1):
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

# def _load_data(train_path: Path, eval_path: Path, sample_frac: Optional[float], random_state: int):
#     # Parquet is used to avoid schema inference issues found in CSV
#     train_df = pq.read_table(train_path).to_pandas()
#     eval_df = pq.read_table(eval_path).to_pandas()

#     train_df = _maybe_sample(train_df, sample_frac, random_state)
#     eval_df = _maybe_sample(eval_df, sample_frac, random_state)

#     target = "price"
#     if target not in train_df.columns:
#         target = train_df.columns[-1] 

#     return (
#         train_df.drop(columns=[target]),
#         train_df[target],
#         eval_df.drop(columns=[target]),
#         eval_df[target],
#     )

def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix == ".parquet":
        return pq.read_table(path).to_pandas()

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_data(
    train_path: Path | str,
    eval_path: Path | str,
    sample_frac: float | None,
    random_state: int,
):
    train_df = _read_table(Path(train_path))
    eval_df = _read_table(Path(eval_path))

    if sample_frac and 0 < sample_frac < 1:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)
        eval_df = eval_df.sample(frac=sample_frac, random_state=random_state)

    target = "price"
    if target not in train_df.columns:
        raise ValueError("Target column 'price' not found")

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

    # Ensure stable MLflow import
    mlflow = _safe_mlflow_import()
    import optuna  # Deferring optuna import to reduce top-level load time

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        # mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(experiment_name)
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

    # Train best model
    best_params = study.best_trial.params
    best_model = XGBRegressor(**best_params, random_state=random_state, n_jobs=-1, tree_method="hist")
    best_model.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluate best model on eval set
    # --------------------------------------------------
    preds = best_model.predict(X_eval)
    best_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_eval, preds))),
        "mae": float(mean_absolute_error(y_eval, preds)),
        "r2": float(r2_score(y_eval, preds)),
    }

    # Save artifact
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)

    # Log best run
    with mlflow.start_run(run_name="best_xgb_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

    return best_params, study.best_trial.values

if __name__ == "__main__":
    tune_model()