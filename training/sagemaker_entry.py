import os
import joblib
import pandas as pd
from training import train

class Cfg:
    """Configuration for training on aws"""
    aws_data_dir: str = "/opt/ml/input/data/training"
    aws_model_dir: str = "/opt/ml/model"

os.makedirs(Cfg.aws_model_dir, exist_ok=True)

def load_csvs(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_csvs(Cfg.aws_data_dir)
    model_path, metrics = train.train_and_save(X_train, X_val, y_train, y_val, model_name="gbr.joblib")
    # copy model to SageMaker model directory for packaging
    joblib.dump({"model_path": model_path}, os.path.join(Cfg.aws_model_dir, "train_meta.joblib"))
    print("Saved model to:", model_path)
    print("Training metrics:", metrics)