import numpy as np
import math
from math import log
import pandas as pd
from time import gmtime, strftime
import scipy
import sys
from string import punctuation
import time
import sklearn.linear_model as lm
import sklearn.decomposition
from sklearn import metrics, preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from scipy import sparse
from matplotlib import *
from itertools import combinations
import operator
from sklearn import svm


def avgRank(x):
    sortX = sorted(zip(x, list(range(len(x)))))
    r = [0 for k in x]
    currentValue = sortX[0][0]
    endRank = 0
    for i in range(len(sortX)):
        if currentValue != sortX[i][0]:
            currentValue = sortX[i][0]
            for j in range(endRank, i):
                r[sortX[j][1]] = float(endRank+1+i)/2.0
            endRank = i
        if i==len(sortX)-1:
            for j in range(endRank, i+1):
                r[sortX[j][1]] = float(endRank+i+2)/2.0
    return r
def areaUnderCharacter(actual, later):
    r = avgRank(later)
    positiveNumber = len([0 for x in actual if x==1])
    negativeNumber = len(actual)-positiveNumber
    positiveSum = sum([r[i] for i in range(len(r)) if actual[i]==1])
    areaUnderCharacter = ((positiveSum - positiveNumber*(positiveNumber+1)/2.0) / (negativeNumber*positiveNumber))
    sys.stdout.write('.')
    return areaUnderCharacter
def areaUnderCharacter_scorer(estimator, X, y):
    predicted = estimator.predict_proba(X)[:,1]
    return areaUnderCharacter(y, predicted)

def normaliseTenDays(stocks):
    print(stocks)
    def columnProcess(i):
        if operator.mod(i, 5) == 1:
            return stocks[:,i] * 0
        if operator.mod(i, 5) == 2:
            return stocks[:,i] * 0
        if operator.mod(i, 5) == 4:
            return stocks[:,i] * 0
        else:
            return stocks[:,i] / stocks[:,0]
    n = stocks.shape[0]
    stocks_dat =  np.array([ columnProcess(i) for i in range(46)]).transpose()
    return stocks_dat

print("Data is being loaded")
train = np.array(pd.read_table('./training.csv', sep = ","))
test = np.array(pd.read_table('./test.csv', sep = ","))
xTestStockData = normaliseTenDays(test[:,range(2, 48)]) # load in test data
xTestStockIndicator = np.vstack((np.identity(94)[:,range(93)] for i in range(25)))
xTest = xTestStockData
nWindows = 490
windows = list(range(nWindows))
xWindows = [train[:,range(1 + 5*w, 47 + 5*w)] for w in windows]
xWindowsNormalised = [normaliseTenDays(w) for w in xWindows]
xStockData = np.vstack(xWindowsNormalised)
xStockIndicator = np.vstack((np.identity(94)[:,range(93)] for i in range(nWindows)))
X = xStockData
yStockData = np.vstack([train[:, [46 + 5*w, 49 + 5*w]] for w in windows])
y = (yStockData[:,1] - yStockData[:,0] > 0) + 0
print("Step completed")
print("Models prepartion")
modelname = "lasso"
if modelname == "lasso":
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [lm.LogisticRegression(penalty = "l1", C = c, solver='liblinear') for c in C]
if modelname == "sgd":
    C = np.linspace(0.00005, .01, num = 5)
    models = [lm.SGDClassifier(loss = "log", penalty = "l2", alpha = c, warm_start = False) for c in C]
if modelname == "ridge":
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [lm.LogisticRegression(penalty = "l2", C = c) for c in C]
if modelname == "randomforest":
    C = np.linspace(50, 300, num = 10)
    models = [RandomForestClassifier(n_estimators = int(c)) for c in C]
print("Calculating scores")
cv_scores = [0] * len(models)
for i, model in enumerate(models):
    cv_scores[i] = np.mean(cross_val_score(model, X, y, cv=5, scoring = areaUnderCharacter_scorer))
    print(" (%d/%d) C = %f: CV = %f" % (i + 1, len(C), C[i], cv_scores[i]))
best = cv_scores.index(max(cv_scores))
bestModel = models[best]
bestCV = cv_scores[best]
bestC = C[best]
print("Best %f: %f" % (bestC, bestCV))
print("Training")
bestModel.fit(X, y)
print("Prediction")
pred = bestModel.predict_proba(xTest)[:,1]
testfile = pd.read_csv('./test.csv', sep=",", na_values=['?'], index_col=[0,1])
testindices = [100 * D + StId for (D, StId) in testfile.index]
pred_df = pd.DataFrame(np.vstack((testindices, pred)).transpose(), columns=["Id", "Prediction"])
pred_df.to_csv('./predictions/' + modelname + '/' + modelname + ' ' + strftime("%m-%d %X") + " C-" + str(round(bestC,4)) + " CV-" + str(round(bestCV, 4)) + ".csv", index = False)
print("Done")