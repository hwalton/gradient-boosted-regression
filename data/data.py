import os
from symtable import Class
import numpy as np
from sklearn.datasets import fetch_california_housing
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import pandas as pd

class Cfg:
    """Configuration for data processing"""
    raw_dir: str = "data/raw"
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
    """
    Simulate inflation using continuous compounding: A = P * e^(rt)
    
    Parameters:
    - annual_rate: Annual inflation rate (e.g., 0.03 for 3%)
    - current_date: Current date for calculation
    
    Returns continuous exponential growth over fractional years
    """
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
    """Add Gaussian noise to features and target"""
   
    # Add noise to each feature column
    for col in X.columns:
        noise = np.random.normal(0, noise_level * X[col].std(), size=X.shape[0])
        X.loc[:, col] += noise
    
    # Add noise to target
    noise = np.random.normal(0, noise_level * y.std(), size=y.shape[0])
    y += noise

def process_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    save_csv: bool = True,
):
    """
    Process dataset: simulate inflation, add noise, then split into train/val/test.

    Parameters:
    - test_size: fraction of the full dataset to reserve for test (default 0.2)
    - val_size: fraction of the training portion to reserve for validation (default 0.2)
                  (i.e. validation_size_relative = val_size of the remaining training split)
    - random_state: random seed for reproducible splits
    - save_csv: whether to write split CSVs under 'processed/'

    Returns:
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Simulate inflation and add noise (mutate in-place)
    simulate_inflation(X, y)
    add_noise(X, y, noise_level=0.01)

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

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    X, y = get_data()
    process_data(X, y)

if __name__ == "__main__":
    main()