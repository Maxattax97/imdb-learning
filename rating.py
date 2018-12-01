#!/usr/bin/env python3
import mord
from sklearn import linear_model, metrics, preprocessing
import pandas
import random

#  csv_file = open("data/dataset.csv", "r")
#
#  data = pandas.read_csv("data/dataset.csv")
#  print(data)


def train_rating(data):
    X = data.loc[
        :,
        (
            "isAdult",
            "runtimeMinutes",
            "isDrama",
            "isComedy",
            "isShort",
            "isDocumentary",
            "isTalkShow",
            "isRomance",
            "isFamily",
            "isNews",
            "isAnimation",
            "isRealityTV",
            "isMusic",
            "isCrime",
            "isAdventure",
            "isGameShow",
            "isMystery",
            "isSport",
            "isFantasy",
            "isHorror",
            "isThriller",
            "isSciFi",
            "isHistory",
            "isBiography",
            "isMusical",
            "isWestern",
            "isWar",
            "isFilmNoir",
            "numVotes",
            "directorsExperience",
            "actorsExperience",
            "producersExperience",
        ),
    ]
    y = data.loc[:, ("averageRating")]
    zipped = list(zip(X, y))
    random.shuffle(zipped)
    shuffled_X, shuffled_y = zip(*zipped)

    split = int(0.8 * len(X))
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    model = linear_model.LinearRegression(fit_intercept=True)

    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    correct = 0.0
    for i, prediction in enumerate(predictions):
        if abs(prediction - test_y.loc[i + split]) <= 0.5:
            correct += 1.0
        #  print("Predictions: {}, Label: {}".format(prediction, test_y.loc[i + split]))

    correct = correct / len(predictions)

    return {"strengths": {"feature_a": 0.2, "feature_b": 0.6}, "accuracy": correct}
