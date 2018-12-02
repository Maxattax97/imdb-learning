#!/usr/bin/env python3
import mord
from sklearn import tree, metrics, preprocessing
import pandas
import random
from validate import kfoldscv

#  csv_file = open("../dataset/dataset.csv", "r")
#
#  data = pandas.read_csv("../dataset/dataset.csv")
#  print(data)


def train_genres(data):
    X = data.loc[
        :,
        (
            "runtimeMinutes",
            "numVotes",
            "directorsExperience",
            "actorsExperience",
            "averageRating",
        ),
    ]
    y = data.drop(
        [
            "runtimeMinutes",
            "numVotes",
            "directorsExperience",
            "actorsExperience",
            "producersExperience",
            "averageRating",
        ],
        1,
    )
    
    zipped = list(zip(X, y))
    random.shuffle(zipped)
    shuffled_X, shuffled_y = zip(*zipped)

    split = int(0.8 * len(X))
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    model = tree.DecisionTreeClassifier()
    k = 10
    accuracy = kfoldscv(model, k, X, y)

    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    correct = 0.0
    perfect = 0.0
    for i, prediction in enumerate(predictions):
        differences = sum(abs(prediction - test_y.loc[i + split]))
        score = (len(prediction) - differences) / len(prediction)
        correct += score
        #  print(i)
        #  print("Predict:\t{}".format(list(prediction.astype(int))))
        #  print(
        #  "Labels: \t{}".format(list(test_y.loc[i + split].values.flatten().astype(int)))
        #  )
        if differences == 0:
            perfect += 1
    correct = correct / len(predictions)
    #  print("Correct: {}%".format(correct))
    perfect = perfect / len(predictions)
    #  print("Perfect: {}%".format(perfect))


    coef = model.feature_importances_
    strengths = {feature:abs(coef[i]) for i, feature in enumerate(X.columns.tolist())}
    return {"strengths": strengths, "accuracy": sum(accuracy)/len(accuracy)}
