#!/usr/bin/env python3
import numpy as np
import random

# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
def run(model,k,X,y):
    
    z = np.zeros((n,1))
    random.shuffle(X)
    random.shuffle(y)

    for i in range(k):
        all_except_i = range(i) + range(i+1, n)
        X_train = X[all_except_i]
        Y_train = y[all_except_i]

        model.fit(X_train, Y_train)

        predictions = model.predict(X[i])
        print(predictions)

        correct = 0.0
        for prediction in enumerate(predictions):
            if prediction == y[i]:
                correct += 1

        z[i] = correct / len(predictions)

    return z
