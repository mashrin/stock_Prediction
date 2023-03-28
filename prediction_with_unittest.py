import numpy as np
import math
from math import log
import pandas as p
from time import gmtime, strftime
import scipy
import sys
from string import punctuation
import time
import sklearn.linear_model as lm
import sklearn.decomposition
from sklearn import metrics, preprocessing, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from scipy import sparse
from matplotlib import *
from itertools import combinations
import operator
from sklearn import svm

# Functions
def avgRank(x):
    sortX = sorted(zip(x, range(len(x))))
    r = [0 for k in x]
    currentValue = sortX[0][0]
    endRank = 0
    for i in range(len(sortX)):
        if currentValue != sortX[i][0]:
            currentValue = sortX[i][0]
            for j in range(endRank, i):
                r[sortX[j][1]] = float(endRank + 1 + i) / 2.0
            endRank = i
        if i == len(sortX) - 1:
            for j in range(endRank, i + 1):
                r[sortX[j][1]] = float(endRank + i + 2) / 2.0
    return r


def areaUnderCharacter(actual, later):
    r = avgRank(later)
    positiveNumber = len([0 for x in actual if x == 1])
    negativeNumber = len(actual) - positiveNumber
    positiveSum = sum([r[i] for i in range(len(r)) if actual[i] == 1])
    areaUnderCharacter = (
        (positiveSum - positiveNumber * (positiveNumber + 1) / 2.0)
        / (negativeNumber * positiveNumber)
    )
    sys.stdout.write(".")
    return areaUnderCharacter


def areaUnderCharacter_scorer(estimator, X, y):
    predicted = estimator.predict_proba(X)[:, 1]
    return areaUnderCharacter(y, predicted)


def normaliseTenDays(stocks):
    def columnProcess(i):
        if operator.mod(i, 5) == 1:
            return stocks[:, i] * 0
        if operator.mod(i, 5) == 2:
            return stocks[:, i] * 0
        if operator.mod(i, 5) == 4:
            return stocks[:, i] * 0
        else:
            return stocks[:, i] / stocks[:, 0]

    n = stocks.shape[0]
    stocks_dat = np.array([columnProcess(i) for i in range(46)]).transpose()
    return stocks_dat


# Unit Tests
import unittest


class TestFunctions(unittest.TestCase):
    def test_avgRank(self):
        self.assertEqual(avgRank([4, 3, 2, 1]), [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(avgRank([1, 2, 3, 4]), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(avgRank([4, 1, 3, 2]), [4.0, 1.0, 3.0, 2.0])

        self.assertEqual(avgRank([1, 1, 1, 1]), [2.5, 2.5, 2.5, 2.5])
        self.assertEqual(avgRank([]), [])

    def test_areaUnderCharacter(self):
        actual = [1, 0, 1, 0]
        later = [0.8, 0.2, 0.6, 0.4]
        self.assertAlmostEqual(areaUnderCharacter(actual, later), 0.75, places=6)

        self.assertEqual(areaUnderCharacter([], []), 0)

    def test_areaUnderCharacter_scorer(self):
        class MockEstimator():
            def predict_proba(self, X):
                return np.array([[0.4, 0.6], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6]])

        estimator = MockEstimator()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([1, 0, 1, 0])

        self.assertAlmostEqual(
            areaUnderCharacter_scorer(estimator, X, y), 0.75, places=6
        )

    def test_normaliseTenDays(self):
        stocks = np.array(
            [[10, 1, 1, 1, 1, 10, 1, 1, 1, 1], [20, 2, 2, 2, 2, 20, 2, 2, 2, 2]]
        )
        expected_result = np.array(
            [
                [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            ]
        )
        np.testing.assert_array_equal(normaliseTenDays(stocks), expected_result)

        stocks = np.array([])
        expected_result = np.array([])
        np.testing.assert_array_equal(normaliseTenDays(stocks), expected_result)


if __name__ == "__main__":
    unittest.main()
    
