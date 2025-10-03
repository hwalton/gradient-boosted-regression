import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import importlib.util

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# Debug BEFORE importing heavy libs so we can see the runtime interpreter and sys.path
try:
    LOG.info("DEBUG python executable: %s", sys.executable)
    LOG.info("DEBUG python version: %s", sys.version)
    LOG.info("DEBUG env PATH: %s", os.environ.get("PATH"))
    LOG.info("DEBUG env PYTHONPATH: %s", os.environ.get("PYTHONPATH"))
    LOG.info("DEBUG sys.path (first 20): %s", sys.path[:20])
    LOG.info("DEBUG pandas spec: %s", importlib.util.find_spec("pandas"))
except Exception:
    LOG.exception("Failed to emit debug runtime info")

# Import boto3 and pandas after debug logging so ImportError/visibility is visible in logs
import boto3
try:
    import pandas as pd
except Exception as e:
    LOG.exception("Pandas import failed: %s", e)
    # re-raise so Lambda init shows ImportModuleError and logs above are available
    raise
import subprocess
import importlib
from dotenv import load_dotenv

S3 = boto3.client("s3")

load_dotenv(".env", override=True)

MODEL_BUCKET = os.getenv("MODEL_BUCKET") or os.getenv("BUCKET")
PROCESSED_PREFIX = os.getenv("PROCESSED_PREFIX", "processed/")
MODEL_KEY = os.getenv("MODEL_KEY", "models/gbr.joblib")
TMP_DIR = "/tmp/processed"
REPO_DIR = "/tmp/repo"

# Git cloning config (optional)
GIT_REPO = os.getenv("GIT_REPO")          # e.g. https://github.com/username/repo.git
GIT_REF = os.getenv("GIT_REF", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional, for private repos
FORCE_PULL = os.getenv("FORCE_PULL", "false").lower() in ("1", "true", "yes")

EXPECTED_FILES = [
    "X_train.csv", "X_val.csv", "X_test.csv",
    "y_train.csv", "y_val.csv", "y_test.csv",
]

def _download_processed(bucket: str, prefix: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fname in EXPECTED_FILES:
        key = prefix.rstrip("/") + "/" + fname
        dest = Path(out_dir) / fname
        LOG.info("Downloading s3://%s/%s -> %s", bucket, key, dest)
        S3.download_file(bucket, key, str(dest))

def _upload_model(local_path: str, bucket: str, key: str) -> str:
    S3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

def _ensure_repo_cloned(repo_url: str, dest: str = REPO_DIR, ref: str = "main", token: Optional[str] = None, force: bool = False):
    """
    Ensure repo is cloned into dest. If already exists and force=True, do a fetch+checkout+pull.
    Supports token-auth for private repos if token provided (uses https URL).
    """
    if token:
        # insert token into https URL (NOTE: token may appear in process list/logs; avoid printing)
        if repo_url.startswith("https://"):
            repo_url_auth = repo_url.replace("https://", f"https://{token}@", 1)
        else:
            repo_url_auth = repo_url
    else:
        repo_url_auth = repo_url

    try:
        if Path(dest).exists():
            if force:
                LOG.info("Updating existing repo in %s", dest)
                subprocess.run(["git", "-C", dest, "fetch", "--all"], check=True)
                subprocess.run(["git", "-C", dest, "checkout", ref], check=True)
                subprocess.run(["git", "-C", dest, "pull", "--ff-only"], check=True)
        else:
            LOG.info("Cloning %s -> %s (ref=%s)", repo_url, dest, ref)
            subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, repo_url_auth, dest], check=True)
    except subprocess.CalledProcessError as e:
        LOG.exception("Git operation failed: %s", e)
        raise

def _import_training_from_repo(dest: str = REPO_DIR):
    """
    Add repo root to sys.path and import/reload training.train as train_module.
    """
    if not Path(dest).exists():
        raise RuntimeError("Repo not cloned to %s" % dest)
    if dest not in sys.path:
        sys.path.insert(0, dest)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("training.train")
        importlib.reload(module)
    except Exception:
        # fallback: import training package then find train
        pkg = importlib.import_module("training")
        module = importlib.reload(importlib.import_module("training.train"))
    return module

def handler(event: Dict[str, Any], context):
    """
    Lambda entrypoint. event may override env vars:
      { "model_bucket": "...", "processed_prefix": "processed/", "model_key": "models/gbr.joblib", "param_grid": {...} }
    The function:
      - downloads processed CSVs from S3,
      - runs tune_hyperparams(...) from training.train,
      - saves the best model using save_model(...) and uploads it to S3.
    """
    bucket = event.get("model_bucket") if isinstance(event, dict) else None
    prefix = event.get("processed_prefix") if isinstance(event, dict) else None
    model_key = event.get("model_key") if isinstance(event, dict) else None
    param_grid = event.get("param_grid") if isinstance(event, dict) else None

    bucket = bucket or MODEL_BUCKET
    prefix = prefix or PROCESSED_PREFIX
    model_key = model_key or MODEL_KEY

    if not bucket:
        return {"statusCode": 400, "body": json.dumps({"error": "MODEL_BUCKET/BUCKET not set"})}

    # prepare /tmp
    if Path(TMP_DIR).exists():
        shutil.rmtree(TMP_DIR)
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    try:
        _download_processed(bucket, prefix, TMP_DIR)

        X_train = pd.read_csv(Path(TMP_DIR) / "X_train.csv")
        X_val = pd.read_csv(Path(TMP_DIR) / "X_val.csv")
        X_test = pd.read_csv(Path(TMP_DIR) / "X_test.csv")
        y_train = pd.read_csv(Path(TMP_DIR) / "y_train.csv").squeeze()
        y_val = pd.read_csv(Path(TMP_DIR) / "y_val.csv").squeeze()
        y_test = pd.read_csv(Path(TMP_DIR) / "y_test.csv").squeeze()

        # import training helpers
        # If GIT_REPO configured, clone/pull and import training from that clone so the function runs latest repo code.
        if GIT_REPO:
            _ensure_repo_cloned(GIT_REPO, dest=REPO_DIR, ref=GIT_REF, token=GITHUB_TOKEN, force=FORCE_PULL)
            train_module = _import_training_from_repo(REPO_DIR)
        else:
            from training import train as train_module

        if not hasattr(train_module, "tune_hyperparams") or not hasattr(train_module, "save_model"):
            raise RuntimeError("training.train must expose tune_hyperparams(...) and save_model(...)")

        # default small grid if none provided
        if param_grid is None:
            param_grid = {
                "n_estimators": [200],
                "learning_rate": [0.1],
                "max_depth": [5],
                "subsample": [1.0],
            }

        LOG.info("Starting hyperparameter tuning with grid: %s", param_grid)
        tune_res = train_module.tune_hyperparams(X_train, y_train, X_val, y_val, param_grid, random_state=42)
        best_model = tune_res.get("best_model")
        best_params = tune_res.get("best_params")
        best_score = tune_res.get("best_score")

        if best_model is None:
            LOG.error("Tuning did not return a fitted model")
            return {"statusCode": 500, "body": json.dumps({"error": "no best model from tuning", "tune_result": tune_res})}

        # save best model locally using save_model, then upload to S3
        model_basename = os.path.basename(model_key)
        local_model_dir = "/tmp/models"
        os.makedirs(local_model_dir, exist_ok=True)
        local_path = train_module.save_model(
            best_model,
            model_dir=local_model_dir,
            model_name=model_basename,
            params=best_params,
            train_metrics=train_module.evaluate(best_model, X_train, y_train),
            val_metrics=train_module.evaluate(best_model, X_val, y_val),
        )

        s3_uri = _upload_model(local_path, bucket, model_key)
        LOG.info("Uploaded tuned model to %s", s3_uri)

        body = {
            "s3_uri": s3_uri,
            "best_score": best_score,
            "best_params": best_params,
            "trials": tune_res.get("trials"),
        }
        return {"statusCode": 200, "body": json.dumps(body)}

    except Exception as exc:
        LOG.exception("Training/tuning failed")
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
    
if __name__ == "__main__":
    print("runs")