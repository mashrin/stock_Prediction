Test 1
```
def test_data_loading():
    # Test if the code loads the training and test datasets correctly
    # Test 1: Test data loading
    train_data = np.array(p.read_table('./training.csv', sep=","))
    assert train_data.shape == (125000, 47), "Train data loading failed"
    
    test_data = np.array(p.read_table('./test.csv', sep=","))
    assert test_data.shape == (75000, 46), "Test data loading failed"
    
    # Test incorrect data format
    with assert_raises(ValueError):
        p.read_table('./test1_random_data.csv', sep=",")
    with assert_raises(ValueError):
        p.read_table('./test1_correct_data.csv', sep=",")
    
    print("Test 1 passed")
```

Test 2: Test data normalization
Next, we test if the normalization function normaliseTenDays normalizes the training and test data correctly.

We will create a test dataset with some random values between 1 and 100 and a range of values for each feature.

```
test2_data = np.array([[1, 2, 4, 6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    [1, 4, 8, 12, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
                    [2, 3, 6, 9, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]])
```

We will now create a unit test function for this test:

```
def test_data_normalization():
    # Test if the normalization function normaliseTenDays normalizes the training and test data correctly
    # Test 2: Test data normalization
    normalized_data = normaliseTenDays(test2_data)
    
    # Check if the normalized data has the correct shape and if the first column is all zeros
    assert normalized_data.shape == (3, 46), "Data normalization failed"
    assert (normalized_data[:, 0] == 0).all(), "Data normalization failed"
    
    # Check if the last column is correctly normalized
    assert np.allclose(normalized_data[:, -1], [1.0, 1.0, 1.0], atol=0.01), "Data normalization failed"
    
    print("Test 2 passed")
```

Test 3: Test machine learning model training and prediction
Finally, we test if the machine learning model is trained correctly and if the predictions are written to a csv file in the correct format.

We will create a unit test function for this test:

```
def test_model():
    # Test if the machine learning model is trained correctly and if the predictions are written to a csv file in the correct format
    # Test 3: Test machine learning model training and prediction
    
    # Randomly select a small subset of the training data to speed up the testing
    X = xStockData[:1000]
    y = y[:1000]
    xTest = xTestStockData[:100]
    
    # Train the model on the training data
    modelname = "lasso"
    if modelname == "lasso":
        C = np.linspace(300, 5000, num = 10)[::-1]
        models = [lm.LogisticRegression(penalty = "l1", C = c) for c in C]
    if modelname == "sgd":
        C = np.linspace(0.00005, .01, num = 5)
        models = [lm.SGDClassifier(loss = "log", penalty = "l2", alpha = c, warm_start = False) for c in C]
    if modelname == "ridge":
        C = np.linspace(300, 5000, num = 10)[::-1]
        models = [lm.LogisticRegression(penalty = "l2", C = c) for c in C]
    if modelname == "randomforest":
        C = np.linspace(50, 300, num = 10)
        models = [RandomForestClassifier(n_estimators = int(c)) for c in C]
    
    bestModel = models[0]
    bestModel.fit(X, y)
    
    # Make predictions on the test data
    pred = bestModel.predict_proba(xTest)[:,1]
    
    # Save predictions to csv file
    testfile = p.read_csv('./test.csv', sep=",", na_values=['?'], index_col=[0,1])
    testindices = [100 * D + StId for (D, StId) in testfile.index]
    pred_df = p.DataFrame(np.vstack((testindices, pred)).transpose(), columns=["Id", "Prediction"])
    pred_df.to_csv('./test_predictions.csv', index = False)
    
    # Read predictions file and check if it has the correct shape and format
    predictions = np.array(p.read_table('./test_predictions.csv', sep=","))
    assert predictions.shape == (75000, 2), "Model prediction failed"
    assert predictions.dtype.type == np.float64, "Model prediction failed"
    
    print("Test 3 passed")
```

Finally, we will call all the unit test functions:

```
def run_unit_tests():
    test_data_loading()
    test_data_normalization()
    test_model()

run_unit_tests()
``` 
