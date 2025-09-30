import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data import get_data, simulate_inflation

class TestInflationSimulation:
    
    @pytest.fixture
    def sample_data(self):
        """Get a small sample of data for testing"""
        X, y = get_data(save_csv=False)
        # Use first 100 rows for faster testing
        return X.iloc[:100].copy(), y.iloc[:100].copy()
    
    def test_simulate_inflation_no_time_elapsed(self, sample_data):
        """Test inflation simulation at reference date (no time elapsed)"""
        reference_date = datetime(2025, 9, 29)

        X, y = sample_data
        X_sim, y_sim = X.copy(), y.copy()
        
        simulate_inflation(X_sim, y_sim, annual_rate=0.03, current_date=reference_date)
        
        # Should be identical when no time has elapsed
        pd.testing.assert_frame_equal(X, X_sim)
        pd.testing.assert_series_equal(y, y_sim)
    
    def test_simulate_inflation_6_months(self, sample_data):
        """Test inflation simulation for 6 months (0.5 years)"""
        six_months_later = datetime(2026, 3, 29)  # 6 months later
        annual_rate = 0.03

        X, y = sample_data
        X_sim, y_sim = X.copy(), y.copy()

        simulate_inflation(X_sim, y_sim, annual_rate=annual_rate, current_date=six_months_later)

        years_elapsed = 181 / 365
        
        expected_multiplier = np.exp(np.log(1+annual_rate) * years_elapsed)
        
        np.testing.assert_allclose(y_sim, y * expected_multiplier, rtol=1e-10)
        np.testing.assert_allclose(X_sim['MedInc'], X['MedInc'] * expected_multiplier, rtol=1e-10)
        np.testing.assert_allclose(expected_multiplier, 1.014765881, rtol=1e-6)
        
        # Other columns should remain unchanged
        for col in ['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']:
            pd.testing.assert_series_equal(X[col], X_sim[col])
    
    def test_simulate_inflation_1_year(self, sample_data):
        """Test inflation simulation for exactly 1 year"""
        one_year_later = datetime(2026, 9, 29)  # Exactly 1 year later
        annual_rate = 0.1

        X, y = sample_data
        X_sim, y_sim = X.copy(), y.copy()

        simulate_inflation(X_sim, y_sim, annual_rate=annual_rate, current_date=one_year_later)

        expected_multiplier = 1.1
        
        np.testing.assert_allclose(y_sim, y * expected_multiplier, rtol=1e-10)
        np.testing.assert_allclose(X_sim['MedInc'], X['MedInc'] * expected_multiplier, rtol=1e-10)
    
    
    def test_simulate_inflation_past_date(self, sample_data):
        """Test inflation simulation with past date (should not change values)"""
        past_date = datetime(2024, 9, 29)  # 1 year before reference
        
        X, y = sample_data
        X_sim, y_sim = X.copy(), y.copy()

        simulate_inflation(X_sim, y_sim, current_date=past_date)
        
        expected_multiplier = 0.9708737864
        
        np.testing.assert_allclose(y_sim, y * expected_multiplier, rtol=1e-10)
        np.testing.assert_allclose(X_sim['MedInc'], X['MedInc'] * expected_multiplier, rtol=1e-10)
    
    def test_simulate_inflation_different_rates(self, sample_data):
        """Different annual rates should produce different multipliers (higher rate -> higher values)"""
        one_year_later = datetime(2026, 9, 29)

        X, y = sample_data
        X_low, y_low = X.copy(), y.copy()
        X_high, y_high = X.copy(), y.copy()

        # Run simulation with two different rates
        simulate_inflation(X_low, y_low, annual_rate=0.01, current_date=one_year_later)
        simulate_inflation(X_high, y_high, annual_rate=0.05, current_date=one_year_later)

        # Higher rate must yield strictly larger values
        assert (y_high > y_low).all()
        assert (X_high['MedInc'] > X_low['MedInc']).all()

        # Compute expected multipliers using the same time conversion as the implementation
        days = (one_year_later - datetime(2025, 9, 29)).days
        years_elapsed = days / 365
        low_mult = np.exp(np.log(1.0 + 0.01) * years_elapsed)
        high_mult = np.exp(np.log(1.0 + 0.05) * years_elapsed)

        # Validate values approximately match expected multipliers
        np.testing.assert_allclose(y_low, y * low_mult, rtol=1e-6)
        np.testing.assert_allclose(y_high, y * high_mult, rtol=1e-6)
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])