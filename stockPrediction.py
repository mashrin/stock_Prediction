import numpy as np
import math
from math import log
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import sklearn.decomposition
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import pandas as p
from time import gmtime, strftime
import scipy
import sys
from string import punctuation
import time
from scipy import sparse
from matplotlib import *
from itertools import combinations
import operator
def avgRank(x):
    sortX = sorted(zip(x,range(len(x))))
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
    def columnProcess(i):
        if operator.mod(i, 5) == 4:
            return np.log(stocks[:,i] + 1)
        else:
            return stocks[:,i] / stocks[:,0]
    n = stocks.shape[0]
    stocks_dat =  np.array([ columnProcess(i) for i in range(31)]).transpose()
    return stocks_dat
    print "Data is being loaded"
train = np.array(p.read_table('./training.csv', sep = ","))
test = np.array(p.read_table('./test.csv', sep = ","))
X_test = normaliseTenDays(test[:,range(17, 48)])
n_windows = 490
windows = range(n_windows)
X_windows = [train[:,range(16 + 5*w, 47 + 5*w)] for w in windows]
X_windows_normalized = [normaliseTenDays(w) for w in X_windows]
X = np.vstack(X_windows_normalized)
y_stockdata = np.vstack([train[:, [46 + 5*w, 49 + 5*w]] for w in windows])
y = (y_stockdata[:,1] - y_stockdata[:,0] > 0) + 0
X_test = X_test[:,[0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]]
X = X[:,[0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30]]
print "Step completed"
model_ridge = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=9081)
model_randomforest = RandomForestClassifier(n_estimators = 200)
pred_ridge = []
pred_randomforest = []
new_Y = []
for i in range(10):
    indxs = np.arange(i, X.shape[0], 10)
    indxs_to_fit = list(set(range(X.shape[0])) - set(np.arange(i, X.shape[0], 10)))
    pred_ridge = pred_ridge + list(model_ridge.fit(X[indxs_to_fit,:], y[indxs_to_fit,:]).predict_proba(X[indxs,:])[:,1])
    pred_randomforest = pred_randomforest + list(model_randomforest.fit(X[indxs_to_fit,:], y[indxs_to_fit,:]).predict_proba(X[indxs,:])[:,1])
    new_Y = new_Y + list(y[indxs,:])
new_X = np.hstack((np.array(pred_ridge).reshape(len(pred_ridge), 1), np.array(pred_randomforest).reshape(len(pred_randomforest), 1)))
print new_X
new_Y = np.array(new_Y).reshape(len(new_Y), 1)
model_suggested = lm.LogisticRegression()
print np.mean(cross_validation.cross_val_score(model_suggested, new_X, new_Y.reshape(new_Y.shape[0]), cv=5, scoring = areaUnderCharacter_scorer))
model_suggested.fit(new_X, new_Y.reshape(new_Y.shape[0]))
print "Prediction"
pred_ridge_test = model_ridge.fit(X, y).predict_proba(X_test)[:,1]
pred_randomforest_test = model_randomforest.fit(X, y).predict_proba(X_test)[:,1]
new_X_test = np.hstack((np.array(pred_ridge_test).reshape(len(pred_ridge_test), 1), np.array(pred_randomforest_test).reshape(len(pred_randomforest_test), 1)))
pred = model_suggested.predict_proba(new_X_test)[:,1]
testfile = p.read_csv('./test.csv', sep=",", na_values=['?'], index_col=[0,1])
testindices = [100 * D + StId for (D, StId) in testfile.index]
pred_df = p.DataFrame(np.vstack((testindices, pred)).transpose(), columns=["Id", "Prediction"])
pred_df.to_csv('./predictions/' + 'suggested' + '/' + 'suggested' + ' ' + strftime("%m-%d %X") + ".csv", index = False)
print "Test file created"
model_suggested.coef_
