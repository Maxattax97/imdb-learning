#!/usr/bin/env python3
import numpy as np
from linreg import run as linreg

# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
def run(k,X,y):
    T = np.zeros(k)
    z = np.zeros((k,1))
    for i in range(0, k):

        lower = ((len(X) * 1.0 * i)/k) * 1.0
        #print(lower)
        lower = np.floor(lower)
        lower = lower.astype(int)

        upper = ((len(X) * (i+1.0))/k)-1.0
        #print(upper)
        upper = np.floor(upper)
        upper = upper.astype(int)

        T = np.arange(lower, upper+1)
        T = T.astype(int)
        #print(T)
        S = np.arange(0, len(X)-1)

        S = np.setdiff1d(S, T)
        thetaPrime = linreg(X[S], y[S])

        val = 0
        for t in range(0, len(T)):
            val += (y[T[t]] - np.dot(thetaPrime.reshape(len(thetaPrime)), X[T[t]])) ** 2
            #print(val)

        z[i] = (val*1.0)/(len(T)*1.0)

    return z
