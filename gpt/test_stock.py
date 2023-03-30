import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Import the functions from the original code
from original_code import avgRank, areaUnderCharacter, areaUnderCharacter_scorer, normaliseTenDays

class TestOriginalCode(unittest.TestCase):

    def test_avgRank(self):
        x = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        expected_output = [6.5, 1.5, 8, 1.5, 9, 11, 3, 10, 9, 6.5, 9]
        self.assertEqual(avgRank(x), expected_output)

    def test_areaUnderCharacter(self):
        actual = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        later = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5]
        expected_output = 0.75
        self.assertAlmostEqual(areaUnderCharacter(actual, later), expected_output, places=2)

    def test_areaUnderCharacter_scorer(self):
        estimator = MagicMock(spec=LogisticRegression)
        estimator.predict_proba.return_value = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.5, 0.5], [0.5, 0.5]])
        X = np.random.rand(10, 2)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        expected_output = 0.75
        self.assertAlmostEqual(areaUnderCharacter_scorer(estimator, X, y), expected_output, places=2)

    def test_normaliseTenDays(self):
        stocks = np.array([
            [100, 1, 2, 3, 4],
            [200, 5, 6, 7, 8],
            [300, 9, 10, 11, 12]
        ])
        expected_output = np.array([
            [1, 0, 0, 3, 0],
            [1, 0, 0, 7, 0],
            [1, 0, 0, 11, 0]
        ])
        np.testing.assert_array_equal(normaliseTenDays(stocks), expected_output)

    def test_cross_val_score(self):
        X = np.random.rand(100, 2)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=areaUnderCharacter_scorer)
        self.assertTrue(0 <= np.mean(cv_scores) <= 1)

if __name__ == '__main__':
    unittest.main()