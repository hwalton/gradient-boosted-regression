import os
import numpy as np
from sklearn.datasets import fetch_california_housing
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import time
import matplotlib.pyplot as plt
# force non-interactive backend to avoid GUI overhead
plt.switch_backend("Agg")
import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import json
import hashlib
import joblib
from typing import List, Optional
import numpy as np
import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
import boto3
from dotenv import load_dotenv

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_log1p_preprocessor(cols: List[str]) -> ColumnTransformer:
    """
    Return a ColumnTransformer that applies log1p to `cols`,
    and passes through the remaining numeric columns unchanged.
    """
    log_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
    ])
    preproc = ColumnTransformer(
        [("log1p", log_tf, cols)],
        remainder="passthrough",
        sparse_threshold=0
    )
    return preproc

def save_log1p_preprocessor(
    X_ref: pd.DataFrame,
    cols: Optional[List[str]] = None,
    path: str = "models/preprocessor.joblib"
) -> str:
    """
    Build & persist a simple log1p preprocessor using X_ref to infer columns if needed.
    Logs the saved file and its sha256 to MLflow (if an active run exists).
    Returns the saved path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if cols is None:
        # choose a reasonable default for skewed numeric cols if not provided
        numeric = X_ref.select_dtypes(include=[np.number]).columns.tolist()
        # prefer MedInc and Population if present
        default = [c for c in ["MedInc", "Population"] if c in numeric]
        cols = default or numeric[:2]

    preproc = build_log1p_preprocessor(cols)
    # Fit is a no-op for FunctionTransformer after imputer, but call for API consistency
    preproc.fit(X_ref)

    joblib.dump({"pipeline": preproc, "cols": cols}, path)

    # log artifact and checksum to MLflow when available
    try:
        if mlflow.active_run() is not None:
            mlflow.log_param("preprocessor_type", "log1p")
            mlflow.log_param("preprocessor_cols", ",".join(cols))
            sha = _file_sha256(path)
            mlflow.log_param("preprocessor_sha256", sha)
            mlflow.log_artifact(path, artifact_path="preprocessor")
    except Exception as e:
        # do not fail training if MLflow logging fails
        print(f"Warning: failed to log preprocessor to MLflow: {e}")

    return path

class Cfg:
    """Configuration for data processing"""
    raw_dir: str = "data/raw"
    drifted_dir: str = "data/drifted"
    processed_dir: str = "data/processed"

def get_data(save_csv=True):
    """Load California housing dataset as pandas DataFrames"""
    # Load with as_frame=True to get pandas DataFrames directly
    data = fetch_california_housing(as_frame=True)
    
    # Extract features and target
    X = data.data  # Already a DataFrame
    y = data.target  # Already a Series
    
    if save_csv:
        os.makedirs(Cfg.raw_dir, exist_ok=True)
        X.to_csv(os.path.join(Cfg.raw_dir, 'features.csv'), index=False)
        y.to_csv(os.path.join(Cfg.raw_dir, 'target.csv'), index=False)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    df = X.copy()
    df['MedHouseVal'] = y

    print(f"Combined DataFrame shape: {df.shape}")
    print(f"Combined DataFrame columns: {list(df.columns)}")
    print(f"First few rows: {df.head()}")
        
    return X, y

def simulate_inflation(X: pd.DataFrame, y: pd.Series, annual_rate=0.03, current_date=datetime.now()):
    # Calculate years since reference date (2025-09-29)
    reference_date = datetime(2025, 9, 29)
    years_elapsed = (current_date - reference_date).days / 365

    print(f"days elapsed: {(current_date - reference_date).days}, years elapsed: {years_elapsed:.3f}")


    print(f"Simulating inflation from {reference_date.date()} to {current_date.date()} ({years_elapsed:.3f} years)")

    # Continuous compounding multiplier
    continuous_multiplier = np.exp(np.log(1+annual_rate) * years_elapsed)
    
    # Mutate in-place (safe & visible to caller)
    X.loc[:, 'MedInc'] *= continuous_multiplier     # in-place update of DataFrame column
    y.loc[:] *= continuous_multiplier               # in-place update of Series values

    print(f"Years elapsed: {years_elapsed:.3f}")
    print(f"Inflation multiplier: {continuous_multiplier:.6f}")

def add_noise(X: pd.DataFrame, y: pd.Series, noise_level=0.01):   
    #Comment out random seed for true randomness
    np.random.seed(42)  # For reproducibility during testing

    # Add noise to each feature column
    for col in X.columns:
        noise = np.random.normal(0, noise_level * X[col].std(), size=X.shape[0])
        X.loc[:, col] += noise
    
    # Add noise to target
    noise = np.random.normal(0, noise_level * y.std(), size=y.shape[0])
    y += noise

def simulate_drift(
    X: pd.DataFrame,
    y: pd.Series,
    save_csv: bool = True
):
    # Simulate inflation and add noise (mutate in-place)
    simulate_inflation(X, y)
    add_noise(X, y, noise_level=0.01)

    if save_csv:
        os.makedirs(Cfg.drifted_dir, exist_ok=True)
        X.to_csv(os.path.join(Cfg.drifted_dir, 'features_drifted.csv'), index=False)
        y.to_csv(os.path.join(Cfg.drifted_dir, 'target_drifted.csv'), index=False)
        print("Drifted data saved in 'drifted/' directory.")


def preprocess_data(X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    save_csv: bool = True
    ):

    # Safe log1p: clip negatives (or shift) and record issues so transform is robust
    def _safe_log1p(series: pd.Series, name: str) -> pd.Series:
        s = series.copy()
        neg_count = int((s < 0).sum())
        if neg_count:
            print(f"Warning: {neg_count} negative values in {name}; clipping to 0 before log1p")
        s = s.clip(lower=0)
        out = np.log1p(s)
        nan_count = int(out.isna().sum())
        if nan_count:
            print(f"Warning: {nan_count} NaNs in {name} after log1p")
        if mlflow.active_run() is not None:
            try:
                mlflow.log_param(f"{name}_negatives", neg_count)
                mlflow.log_param(f"{name}_nans_after_log", nan_count)
            except Exception:
                pass
        return out

    if "MedInc" in X.columns:
        X.loc[:, "MedInc"] = _safe_log1p(X["MedInc"], "MedInc")
    if "Population" in X.columns:
        X.loc[:, "Population"] = _safe_log1p(X["Population"], "Population")
    
    # First split off the test set from the full data
    X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
    )
    # Then split the remaining data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    if save_csv:
        os.makedirs(Cfg.processed_dir, exist_ok=True)
        X_train.to_csv(f"{Cfg.processed_dir}/X_train.csv", index=False)
        X_val.to_csv(f"{Cfg.processed_dir}/X_val.csv", index=False)
        X_test.to_csv(f"{Cfg.processed_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{Cfg.processed_dir}/y_train.csv", index=False)
        y_val.to_csv(f"{Cfg.processed_dir}/y_val.csv", index=False)
        y_test.to_csv(f"{Cfg.processed_dir}/y_test.csv", index=False)
        print("Processed data saved in 'processed/' directory.")

        manifest = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "processed_dir": Cfg.processed_dir,
            "splits": {},
            "target": "MedHouseVal",
            "processing": {"simulate_inflation": {"annual_rate": 0.03}, "add_noise": {"noise_level": 0.01}},
            "random_seed": int(random_state),
        }

        files = {
            "X_train.csv": os.path.join(Cfg.processed_dir, "X_train.csv"),
            "X_val.csv": os.path.join(Cfg.processed_dir, "X_val.csv"),
            "X_test.csv": os.path.join(Cfg.processed_dir, "X_test.csv"),
            "y_train.csv": os.path.join(Cfg.processed_dir, "y_train.csv"),
            "y_val.csv": os.path.join(Cfg.processed_dir, "y_val.csv"),
            "y_test.csv": os.path.join(Cfg.processed_dir, "y_test.csv"),
        }

        # use existing DataFrames for shapes where possible (avoid re-reading)
        manifest["splits"]["X_train.csv"] = {"rows": int(X_train.shape[0]), "cols": int(X_train.shape[1]), "sha256": _file_sha256(files["X_train.csv"])}
        manifest["splits"]["X_val.csv"] = {"rows": int(X_val.shape[0]), "cols": int(X_val.shape[1]), "sha256": _file_sha256(files["X_val.csv"])}
        manifest["splits"]["X_test.csv"] = {"rows": int(X_test.shape[0]), "cols": int(X_test.shape[1]), "sha256": _file_sha256(files["X_test.csv"])}
        manifest["splits"]["y_train.csv"] = {"rows": int(y_train.shape[0]), "cols": 1, "sha256": _file_sha256(files["y_train.csv"])}
        manifest["splits"]["y_val.csv"] = {"rows": int(y_val.shape[0]), "cols": 1, "sha256": _file_sha256(files["y_val.csv"])}
        manifest["splits"]["y_test.csv"] = {"rows": int(y_test.shape[0]), "cols": 1, "sha256": _file_sha256(files["y_test.csv"])}

        # persist manifest to disk
        manifest_path = os.path.join(Cfg.processed_dir, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written to {manifest_path}")

        # Optional: upload processed files + manifest to S3 if bucket configured.
        # Looks for MODEL_BUCKET, BUCKET or S3_BUCKET env var; uses PROCESSED_PREFIX or 'processed/' as prefix.
        s3_bucket = os.getenv("MODEL_BUCKET") or os.getenv("BUCKET") or os.getenv("S3_BUCKET")
        s3_prefix = os.getenv("PROCESSED_PREFIX", "processed/").rstrip("/")
        if s3_bucket:
            s3 = boto3.client("s3")
            try:
                for name, local_path in files.items():
                    s3_key = f"{s3_prefix}/{name}"
                    print(f"Uploading {local_path} -> s3://{s3_bucket}/{s3_key}")
                    s3.upload_file(local_path, s3_bucket, s3_key)
                    manifest["splits"][name]["s3_uri"] = f"s3://{s3_bucket}/{s3_key}"
                # upload manifest too
                manifest_s3_key = f"{s3_prefix}/manifest.json"
                print(f"Uploading {manifest_path} -> s3://{s3_bucket}/{manifest_s3_key}")
                s3.upload_file(manifest_path, s3_bucket, manifest_s3_key)
                manifest["manifest_s3_uri"] = f"s3://{s3_bucket}/{manifest_s3_key}"
                print("Uploaded processed files and manifest to S3.")
            except Exception as e:
                print(f"Warning: failed to upload processed files to S3: {e}")
        else:
            print("No S3 bucket configured; skipping upload of processed files.")

        # log manifest and a few useful params to MLflow if a run is active
        if mlflow.active_run() is not None:
            try:
                mlflow.log_artifact(manifest_path, artifact_path="data_manifest")
                mlflow.log_param("test_size", float(test_size))
                mlflow.log_param("val_size", float(val_size))
                mlflow.log_param("random_seed", int(random_state))
                # log split summaries/checksums
                for name, info in manifest["splits"].items():
                    mlflow.log_param(f"{name}_rows", int(info["rows"]))
                    mlflow.log_param(f"{name}_cols", int(info["cols"]))
                    mlflow.log_param(f"{name}_sha256", info["sha256"])
            except Exception as e:
                # don't let MLflow logging break the data pipeline
                print(f"Warning: failed to log manifest to MLflow: {e}")

        # save and log the deterministic log1p preprocessor fitted on X_train
        try:
            preproc_path = save_log1p_preprocessor(X_train, path=os.path.join("models", "preprocessor.joblib"))
            manifest["preprocessor"] = {"path": preproc_path, "sha256": _file_sha256(preproc_path)}
        except Exception as e:
            print(f"Warning: failed to save/log preprocessor: {e}")
        # overwrite manifest file with any added s3 URIs
        try:
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2)
        except Exception:
            pass

def eda(X: pd.DataFrame, y: pd.Series, outdir="data/reports", sample=2000, kde=False, dpi=80, mlflow_log=True):
    os.makedirs(outdir, exist_ok=True)

    df = X.copy()
    df["target"] = y
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)

    # distribution plots for numeric features
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    skewness = df[numeric].skew().sort_values(ascending=False)
    skew_path = os.path.join(outdir, "skewness.csv")
    skewness.to_csv(skew_path)

    # histograms (one figure per column)
    start = time.time()
    for col in numeric:
        plt.figure(figsize=(6, 4), dpi=dpi)
        # disable KDE (expensive) and limit bins
        sns.histplot(df[col].dropna(), kde=kde, bins=40)
        plt.title(f"Distribution: {col}")
        p = os.path.join(outdir, f"hist_{col}.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
    print(f"Saved {len(numeric)} histograms in {time.time()-start:.2f}s")

    # correlation heatmap (subset if many features)
    corr = df[numeric].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="vlag", center=0)
    corr_path = os.path.join(outdir, "corr_heatmap.png")
    plt.tight_layout()
    plt.savefig(corr_path)
    plt.close()

    # simple summary
    summary = df[numeric].describe().T
    summary_path = os.path.join(outdir, "summary.csv")
    summary.to_csv(summary_path)

    if mlflow_log and mlflow.active_run() is not None:
        try:
            mlflow.log_artifact(skew_path, artifact_path="eda")
            mlflow.log_artifact(corr_path, artifact_path="eda")
            mlflow.log_artifact(summary_path, artifact_path="eda")
            for col in numeric:
                p = os.path.join(outdir, f"hist_{col}.png")
                if os.path.exists(p):
                    mlflow.log_artifact(p, artifact_path="eda")
            # log top skewed features as params for quick reference
            top_skew = skewness.head(5).to_dict()
            for k, v in top_skew.items():
                mlflow.log_param(f"skew_top_{k}", float(v))
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")

    return {"skewness_csv": skew_path, "corr_png": corr_path, "summary_csv": summary_path}

def main(save_csv: bool = True, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """
    Run data pipeline and log as MLflow run (nested if called from an active run).
    """
    load_dotenv(".env", override=True)
    
    mlflow.set_experiment("gradient_boosted_regression")
    # choose nested behavior depending on whether a run is already active
    if mlflow.active_run() is None:
        with mlflow.start_run(run_name="data_preprocessing"):
            X, y = get_data(save_csv=save_csv)
            simulate_drift(X, y)
            preprocess_data(X, y, test_size=test_size, val_size=val_size, random_state=random_state, save_csv=save_csv)
            eda(X, y)

    else:
        with mlflow.start_run(run_name="data_preprocessing", nested=True):
            X, y = get_data(save_csv=save_csv)
            simulate_drift(X, y)
            preprocess_data(X, y, test_size=test_size, val_size=val_size, random_state=random_state, save_csv=save_csv)
            eda(X, y)

if __name__ == "__main__":
    # When running this module directly, let main() decide run nesting.
    main()