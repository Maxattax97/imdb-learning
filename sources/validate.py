#!/usr/bin/env python3
import numpy as np
import random
import pandas

# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of scores of k rows, 1 column
def kfoldscv(model,k,X,y):
    print ("\n\n======= K Folds CV ======")
    X = X.values
    X = np.asarray(X)
    X = X.reshape(len(X), len(X[0]) if len(X[0]) > 0 else 1)
    y = y.values
    y = np.asarray(y)
    y = y.reshape(len(y), len(y[0]) if len(y.shape) > 1 else 1)
    print(y)
    print(X.shape)
    print(y.shape)

    z = np.zeros((k,1))
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    for i in range(k):
        print("\nFold: " + str(i+1))

        lower = ((len(X) * 1.0)/k) * i
        lower = np.floor(lower)
        lower = lower.astype(int)

        upper = ((len(X) * 1.0)/k) * (i+1.0)
        upper = np.floor(upper)
        upper = upper.astype(int)

        print(str(lower) + " to " + str(upper))

        lowerSamples = np.asarray(X[:lower])
        upperSamples = np.asarray(X[upper:])

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

        print("X_Train: " + str(X_train.shape))
        print("Y_train: " + str(Y_train.shape))
        model.fit(X_train, Y_train)

        print("X_Test: " + str(X_test.shape))
        print("Y_test: " + str(Y_test.shape))
        predictions = model.predict(X_test)

        correct = 0.0
        for j, prediction in enumerate(predictions):
#            print(prediction)
#            print(Y_test[(j+i) - 1])
            if len(y[0]) > 1 and not any(abs(prediction - Y_test[j])):
                if j % 1000:
                    print("genre")
                correct += 1
            elif abs(prediction[0] - Y_test[j][0]) < 0.5:
                if j % 1000:
                    print("rating")
                correct += 1
            if j % 100 == 0:
                pass
                #print("\tprediction: {}\n\tlabel: {}\n\tcorrect: {}".format(prediction, Y_test[(j+i)-1].astype(float), not any(abs(prediction - Y_test[(j + i) - 1]))))

        z[i] = correct / len(predictions)

    print(z)
    return z
