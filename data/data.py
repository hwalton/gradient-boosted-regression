import os
from symtable import Class
import numpy as np
from sklearn.datasets import fetch_california_housing
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import hashlib
import mlflow

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

        # write a small manifest for lineage and reproducibility
        def _file_sha256(path: str) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

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


def main():
    X, y = get_data()
    simulate_drift(X, y)
    preprocess_data(X, y)

if __name__ == "__main__":
    mlflow.set_experiment("gradient_boosted_regression")
    # top-level run for full pipeline
    with mlflow.start_run(run_name="data_pipeline"):
        main()