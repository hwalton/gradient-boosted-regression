import logging
import os
from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


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

def train_and_save(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    model_name: str = "gbr.joblib",
    params: Dict[str, Any] = None,
    random_state: int = 42
) -> Tuple[str, Dict[str, Any]]:
    model = train_gbr(X_train, y_train, random_state=random_state, params=params)
    train_metrics = evaluate(model, X_train, y_train)
    val_metrics = evaluate(model, X_val, y_val)

    os.makedirs(Cfg.model_dir, exist_ok=True)
    path = os.path.join(Cfg.model_dir, model_name)
    joblib.dump({"model": model, "params": params, "train_metrics": train_metrics, "val_metrics": val_metrics}, path)

    return path, {"train": train_metrics, "val": val_metrics}

# final evaluation moved to a standalone function (not called by default)
def evaluate_final_model_on_test(model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Load a saved model file and evaluate it on the held-out test set.
    This function is intentionally not called from main; run manually when ready
    for the unbiased final evaluation.
    """
    saved = joblib.load(model_path)
    model = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    metrics = evaluate(model, X_test, y_test)
    return metrics

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

    # train on X_train / y_train and validate on X_val / y_val
    model_path, metrics = train_and_save(X_train, X_val, y_train, y_val)
    logger.info("Model saved to %s", model_path)
    logger.info("Training metrics: %s", metrics["train"])
    logger.info("Validation metrics: %s", metrics["val"])

    # metrics = evaluate_final_model_on_test(model_path, X_test, y_test)
    # logger.info("Test metrics (final holdout): %s", metrics)

if __name__ == "__main__":
    # Configure logging to show INFO messages on the console
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()