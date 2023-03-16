# Code Documentation

This code takes in stock data from two csv files `training.csv` and `test.csv`, and uses machine learning algorithms to predict the direction of the stock prices in the future. The predicted values are stored in a csv file in the 'predictions' folder.

## Libraries Used

1. numpy
2. math
3. pandas
4. time
5. scipy
6. sys
7. sklearn

## Functions and their Descriptions

### avgRank(x)

This function returns a list of averaged rankings for the values in the input list `x`.

### areaUnderCharacter(actual, later)

This function computes the area under the character curve given the actual and predicted values.

### areaUnderCharacter_scorer(estimator, X, y)

This function returns the area under the character curve for a given estimator, input features, and true values.

### normaliseTenDays(stocks)

This function normalizes the ten days of stock data for a given input in a specific way and returns the normalized data.

## Code Execution

1. Load the data from the csv files `training.csv` and `test.csv`.
2. Normalize the stock data for both the training and test data.
3. Create machine learning models using either lasso, sgd, ridge, or random forest algorithms.
4. Calculate the scores for each model using the cross-validation method.
5. Choose the best model with the highest score.
6. Train the best model with the normalized data.
7. Predict the direction of stock prices for the test data using the trained model.
8. Save the predicted values in 'predictions' folder in csv format.

## Docstrings for Each Function

### avgRank(x)

"""
This function takes in a list x and returns a list of averaged rankings for the values in the input list.

Parameters:
x (list): A list of values to rank.

Returns:
r (list): A list of averaged rankings for the values in the input list.
"""

### areaUnderCharacter(actual, later)

"""
This function computes the area under the character curve given the actual and predicted values.

Parameters:
actual (list): A list of actual values.
later (list): A list of predicted values.

Returns:
areaUnderCharacter (float): The area under the character curve computed for the given inputs.
"""

### areaUnderCharacter_scorer(estimator, X, y)

"""
This function calculates the area under the character curve for a given estimator, input features, and true values.

Parameters:
estimator (estimator object): This is an estimator object implementing 'fit' and 'predict_proba'.
X (array-like): Input features.
y (array-like): True values for the input features.

Returns:
areaUnderCharacter (float): The area under the character curve computed for the given inputs.
"""

### normaliseTenDays(stocks)

"""
This function normalizes the ten days of stock data for a given input in a specific way and returns the normalized data.

Parameters:
stocks (ndarray): An array of stock data.

Returns:
stocks_dat (ndarray): An array of normalized stock data.
"""
