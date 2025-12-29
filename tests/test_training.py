import math
import os
from pathlib import Path
from joblib import load

from src.training_pipeline.train import train_model
from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.tune import tune_model


TRAIN_PATH = Path("data/processed/feature_engineered_train.csv")
EVAL_PATH = Path("data/processed/feature_engineered_eval.csv")


def _assert_metrics(m):
    assert set(m.keys()) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in m.values())


def test_train_creates_model_and_metrics(tmp_path):
    out_path = tmp_path / "xgb_model.pkl"

    _, metrics = train_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=out_path,
        model_params={"n_estimators": 20, "max_depth": 4, "learning_rate": 0.1},
        sample_frac=0.02,
    )

    assert out_path.exists()
    _assert_metrics(metrics)

    model = load(out_path)
    assert model is not None


def test_eval_works_with_saved_model(tmp_path):
    model_path = tmp_path / "xgb_model.pkl"

    train_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=model_path,
        model_params={"n_estimators": 20},
        sample_frac=0.02,
    )

    metrics = evaluate_model(
        model_path=model_path,
        eval_path=EVAL_PATH,
        sample_frac=0.02,
    )

    _assert_metrics(metrics)


def test_tune_saves_best_model(tmp_path):
    model_out = tmp_path / "xgb_best.pkl"

    # ✅ FORCE FILE-BASED MLFLOW STORE (NO ALEMBIC)
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = f"file:{tracking_dir.resolve()}"

    # isolate MLflow completely
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

    best_params, best_metrics = tune_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=model_out,
        n_trials=2,
        sample_frac=0.02,
        tracking_uri=tracking_uri,
        experiment_name="test_xgb_optuna",
    )

    assert model_out.exists()
    assert isinstance(best_params, dict) and best_params
    _assert_metrics(best_metrics)

    print("✅ tune_model test passed")
