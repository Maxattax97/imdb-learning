#!/usr/bin/env python3
import numpy as np
import random
import pandas

# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of scores of k rows, 1 column
def kfoldscv(model,k,X,y):
    #print("in")
    X = X.values
    X = np.asarray(X)
    y = y.values
    y = np.asarray(y)

    z = np.zeros((k,1))
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    for i in range(k):

        lower = ((len(X) * 1.0)/k) * i
        lower = np.floor(lower)
        lower = lower.astype(int)

        upper = ((len(X) * 1.0)/k) * (i+1.0)
        upper = np.floor(upper)
        upper = upper.astype(int)

        lowerSamples = np.asarray(X[:lower])
        upperSamples = np.asarray(X[upper:])

        #print(lowerSamples.shape)
        #print(upperSamples.shape)

        X_train = 0
        if np.size(lowerSamples, 0) != 0 and np.size(upperSamples, 0) != 0:
            X_train = np.concatenate((lowerSamples, upperSamples), axis=0)

        if np.size(lowerSamples, 0) == 0:
            X_train = upperSamples

        if np.size(upperSamples, 0) == 0:
            X_train = lowerSamples

        X_test = np.asarray(X[lower:upper])

        lowerSamples = np.asarray(y[:lower])
        upperSamples = np.asarray(y[upper:])

        Y_train = 0
        if np.size(lowerSamples, 0) != 0 and np.size(upperSamples, 0) != 0:
            Y_train = np.concatenate((lowerSamples, upperSamples), axis=0)

        if np.size(lowerSamples, 0) == 0:
            Y_train = upperSamples

        if np.size(upperSamples, 0) == 0:
            Y_train = lowerSamples

        Y_test = np.asarray(y[lower:upper])

        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)
        #print("predictions matrix: ")
        #print(predictions)

        correct = 0.0
        for j, prediction in enumerate(predictions):
            if prediction.all() == Y_test[(j + i) - 1].all():
                correct += 1

        z[i] = correct / len(predictions)

    #print(z)
    return z
