#!/usr/bin/env python3
import numpy as np
import random

# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of scores of k rows, 1 column
def kfoldscv(model,k,X,y):
    
    z = np.zeros((n,1))
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

        X_train = np.concatenate(X[:lower], X[upper:])
        X_test = X[lower:upper]

        Y_train = np.concatenate(y[:lower], y[upper:])
        Y_test = y[lower:upper]

        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)
        print("predictions matrix: ")
        print(predictions)

        correct = 0.0
        for j, prediction in enumerate(predictions):
            if prediction == y_test[j + i]:
                correct += 1

        z[i] = correct / len(predictions)

    return z

