import numpy as np
import pandas as pd
from prediction_python3 import avgRank, areaUnderCharacter, areaUnderCharacter_scorer, normaliseTenDays

def test_avgRank():
    assert avgRank([50, 30, 40]) == [3, 1, 2]
    assert avgRank([10, 20, 30]) == [1, 2, 3]

def test_areaUnderCharacter():
    actual = np.array([1, 0, 1])
    later = np.array([0.9, 0.1, 0.8])
    assert areaUnderCharacter(actual, later) == 1.0

def test_normaliseTenDays():
    stocks = np.array([
        [100, 2, 3, 50, 5, 1],
        [200, 4, 6, 100, 10, 2]
    ])
    normalised_stocks = normaliseTenDays(stocks)
    expected_normalised_stocks = np.array([
        [1, 0, 1, 0.5, 0, 0.01],
        [1, 0, 1, 0.5, 0, 0.01]
    ])
    np.testing.assert_array_almost_equal(normalised_stocks, expected_normalised_stocks)

def test_areaUnderCharacter_scorer():
    class MockEstimator:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]])

    estimator = MockEstimator()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 0, 1])
    assert areaUnderCharacter_scorer(estimator, X, y) == 1.0

if __name__ == "__main__":
    test_avgRank()
    test_areaUnderCharacter()
    test_normaliseTenDays()
    test_areaUnderCharacter_scorer()
