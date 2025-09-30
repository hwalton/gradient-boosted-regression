import os
from typing import Tuple
import pandas as pd

def load_processed_data(processed_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load processed train/val/test CSVs from processed_dir and return:
    (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises FileNotFoundError if any expected file is missing.
    """
    files = {
        "X_train": os.path.join(processed_dir, "X_train.csv"),
        "X_val":  os.path.join(processed_dir, "X_val.csv"),
        "X_test":  os.path.join(processed_dir, "X_test.csv"),
        "y_train": os.path.join(processed_dir, "y_train.csv"),
        "y_val": os.path.join(processed_dir, "y_val.csv"),
        "y_test":  os.path.join(processed_dir, "y_test.csv"),
    }

    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing processed files: {missing}")

    X_train = pd.read_csv(files["X_train"])
    X_val = pd.read_csv(files["X_val"])
    X_test  = pd.read_csv(files["X_test"])

    # Read targets and convert single-column DataFrames to Series
    def _read_target(path: str) -> pd.Series:
        df = pd.read_csv(path)
        return df.iloc[:, 0] if df.shape[1] == 1 else df

    y_train = _read_target(files["y_train"])
    y_val = _read_target(files["y_val"])
    y_test  = _read_target(files["y_test"])

    return X_train, X_val, X_test, y_train, y_val, y_test