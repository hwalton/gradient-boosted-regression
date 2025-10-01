import logging
import os
from typing import Tuple, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import mlflow
import hashlib
import itertools

from utils.utils import load_processed_data

logger = logging.getLogger(__name__)

class Cfg:
    """Configuration for training"""
    processed_dir: str = "data/processed"
    model_dir: str = "models"

def train_gbr(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    params: Dict[str, Any] = None
) -> GradientBoostingRegressor:
    params = params or {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": random_state,
    }
    model = GradientBoostingRegressor(**params)
    model.fit(X, y)
    return model

def evaluate(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    rmse = root_mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

# final evaluation moved to a standalone function (not called by default)
def evaluate_final_model_on_test(model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    saved = joblib.load(model_path)
    model = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    metrics = evaluate(model, X_test, y_test)

    # attempt to log metrics/artifact to MLflow, but don't raise on failure
    try:
        mlflow.set_experiment("gradient_boosted_regression")
        metric_dict = {f"test_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(metric_dict)

        # log the model artifact again, in case the top-level run finished
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="models")
    except Exception as e:
        logger.warning("Logging to MLflow failed: %s", e)

    return metrics

def save_model(
    model,
    model_dir: str,
    model_name: str = "gbr.joblib",
    params: Dict[str, Any] | None = None,
    train_metrics: Dict[str, float] | None = None,
    val_metrics: Dict[str, float] | None = None,
) -> str:
    """
    Persist model to model_dir/model_name and (optionally) log artifact to MLflow.
    Returns the local model path.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    try:
        joblib.dump(
            {"model": model, "params": params, "train_metrics": train_metrics, "val_metrics": val_metrics},
            model_path,
        )
        logger.info("Saved model to %s", model_path)
    except Exception as e:
        logger.error("Failed to save model to %s: %s", model_path, e)
        raise

    # attempt to log artifact to MLflow but do not raise on failure
    try:
        if mlflow:
            mlflow.log_artifact(model_path, artifact_path="models")
    except Exception as e:
        logger.warning("Logging model artifact to MLflow failed: %s", e)

    return model_path

def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Dict[str, List[Any]],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Simple grid search (no CV). Train a model for each combination of params
    on X_train and evaluate on X_val. Returns the best params (min val RMSE)
    and a list of trial results.
    """
    results = []
    models = []
    keys = list(param_grid.keys())
    for vals in itertools.product(*(param_grid[k] for k in keys)):
        params = {k: v for k, v in zip(keys, vals)}
        # ensure reproducibility param
        if "random_state" not in params:
            params["random_state"] = random_state

        # train quickly with these params
        model = train_gbr(X_train, y_train, random_state=random_state, params=params)
        metrics = evaluate(model, X_val, y_val)
        trial = {"params": params, "val_rmse": metrics["rmse"], "val_mae": metrics["mae"], "val_r2": metrics["r2"]}
        results.append(trial)
        models.append(model)

    # choose best by val_rmse
    best_idx = int(np.argmin([r["val_rmse"] for r in results]))
    best = results[best_idx]
    best_model = models[best_idx]
    return {
        "best_params": best["params"],
        "best_score": best["val_rmse"],
        "trials": results,
        "best_model": best_model,
    }

def main():
    """
    Minimal training entrypoint: loads processed data and prints basic info.
    Extend this function to add model training, evaluation, and persistence.
    """

    processed_dir = Cfg.processed_dir

    try:
        # load_processed_data now returns: X_train, X_val, X_test, y_train, y_val, y_test
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(processed_dir)
    except FileNotFoundError as exc:
        logger.error("Could not load processed data: %s", exc)
        raise

    logger.info("Loaded processed data from %s", processed_dir)
    logger.info("X_train shape: %s, X_val shape: %s, X_test shape: %s", X_train.shape, X_val.shape, X_test.shape)
    logger.info("y_train length: %d, y_val length: %d, y_test length: %d", len(y_train), len(y_val), len(y_test))

    # Start an MLflow run and log dataset manifest/checksums before training
    mlflow.set_experiment("gradient_boosted_regression")
    # start top-level run if none active, otherwise create a nested training run
    if mlflow.active_run() is None:
        run_ctx = mlflow.start_run(run_name="training")
        nested = False
    else:
        run_ctx = mlflow.start_run(run_name="training", nested=True)
        nested = True
    with run_ctx:
        # log simple dataset stats
        mlflow.log_param("X_train_rows", int(X_train.shape[0]))
        mlflow.log_param("X_train_cols", int(X_train.shape[1]))
        mlflow.log_param("X_val_rows", int(X_val.shape[0]))
        mlflow.log_param("X_test_rows", int(X_test.shape[0]))

        # if manifest exists, attach it as an artifact for lineage
        manifest_path = os.path.join(processed_dir, "manifest.json")
        if os.path.exists(manifest_path):
            mlflow.log_artifact(manifest_path, artifact_path="data_manifest")

        # Simple hyperparameter tuning (grid search on validation set)
        # param_grid = {
        #     "n_estimators": [100, 200],
        #     "learning_rate": [0.01, 0.05, 0.1],
        #     "max_depth": [3, 5],
        #     "subsample": [0.8, 1.0],
        # }
        param_grid = {
            "n_estimators": [200],
            "learning_rate": [0.1],
            "max_depth": [5],
            "subsample": [1.0],
        }
        try:
            tune_res = tune_hyperparams(X_train, y_train, X_val, y_val, param_grid, random_state=42)
            best_params = tune_res["best_params"]
            best_model = tune_res.get("best_model")
            # log best params and best score
            for k, v in best_params.items():
                if mlflow:
                    mlflow.log_param(f"tune_best_{k}", v)
            if mlflow:
                mlflow.log_metric("tune_best_val_rmse", float(tune_res["best_score"]))
            # also log each trial (minimal)
            for i, t in enumerate(tune_res["trials"]):
                for pk, pv in t["params"].items():
                    if mlflow:
                        mlflow.log_param(f"trial_{i}_{pk}", pv)
                if mlflow:
                    mlflow.log_metric(f"trial_{i}_val_rmse", float(t["val_rmse"]))
        except Exception as e:
            logger.warning("Hyperparameter tuning failed, continuing with defaults: %s", e)
            best_params = None
            best_model = None

        logger.info("Reusing best model from tuning (no retrain)")
        model = best_model
        train_metrics = evaluate(model, X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val)
        model_name = "gbr.joblib"
        model_path = save_model(
            model,
            Cfg.model_dir,
            model_name=model_name,
            params=best_params,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        metrics = {"train": train_metrics, "val": val_metrics}

        # log metrics and saved model artifact
        mlflow.log_metrics({f"train_{k}": v for k, v in metrics["train"].items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics["val"].items()})
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="models")

    logger.info("Model saved to %s", model_path)
    logger.info("Training metrics: %s", metrics["train"])
    logger.info("Validation metrics: %s", metrics["val"])

    metrics = evaluate_final_model_on_test(model_path, X_test, y_test)
    logger.info("Test metrics (final holdout): %s", metrics)

if __name__ == "__main__":
    # Configure logging to show INFO messages on the console
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()