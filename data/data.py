import os
import numpy as np
from sklearn.datasets import fetch_california_housing
from datetime import datetime, timedelta

def get_data(save_csv=True):
    """Load California housing dataset as pandas DataFrames"""
    # Load with as_frame=True to get pandas DataFrames directly
    data = fetch_california_housing(as_frame=True)
    
    # Extract features and target
    X = data.data  # Already a DataFrame
    y = data.target  # Already a Series
    
    if save_csv:
        os.makedirs('raw', exist_ok=True)
        X.to_csv('raw/features.csv', index=False)
        y.to_csv('raw/target.csv', index=False)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    df = X.copy()
    df['MedHouseVal'] = y

    print(f"Combined DataFrame shape: {df.shape}")
    print(f"Combined DataFrame columns: {list(df.columns)}")
    print(f"First few rows: {df.head()}")
        
    return X, y

def simulate_inflation(X, y, annual_rate=0.03, current_date=datetime.now()):
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
    
    # Make copies to avoid modifying original data
    X_sim = X.copy()
    y_sim = y.copy()

    print(f"Simulating inflation from {reference_date.date()} to {current_date.date()} ({years_elapsed:.3f} years)")

    # Continuous compounding multiplier
    continuous_multiplier = np.exp(np.log(1+annual_rate) * years_elapsed)
    
    # Apply to house prices and median income
    y_sim = y_sim * continuous_multiplier
    X_sim['MedInc'] = X_sim['MedInc'] * continuous_multiplier
    
    print(f"Years elapsed: {years_elapsed:.3f}")
    print(f"Inflation multiplier: {continuous_multiplier:.6f}")
    
    return X_sim, y_sim

def process_data(X, y):
    """Process and save the data"""
    # Simulate inflation
    X1, y1 = simulate_inflation(X, y)

    # Split into train and test sets (80/20 split)
    split_index = int(0.8 * len(X1))
    X_train, X_test = X1.iloc[:split_index], X1.iloc[split_index:]
    y_train, y_test = y1.iloc[:split_index], y1.iloc[split_index:]

    # Save processed data
    os.makedirs('processed', exist_ok=True)
    X_train.to_csv('processed/X_train.csv', index=False)
    X_test.to_csv('processed/X_test.csv', index=False)
    y_train.to_csv('processed/y_train.csv', index=False)
    y_test.to_csv('processed/y_test.csv', index=False)
    print("Processed data saved in 'processed/' directory.")

    return X_train, X_test, y_train, y_test

def main():
    X, y = get_data()
    _, _, _, _ = process_data(X, y)


if __name__ == "__main__":
    main()