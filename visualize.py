#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random

import rating
import genre

plt.style.use("seaborn")


csv_file = open("data/dataset.csv", "r")
data = pandas.read_csv("data/dataset.csv")

sample_10k = data.sample(frac=1).reset_index(drop=True)
sample_5k = data.sample(n=5000).reset_index(drop=True)
sample_2k = data.sample(n=2000).reset_index(drop=True)

genre_only = sample_10k.drop(
    [
        "runtimeMinutes",
        "numVotes",
        "directorsExperience",
        "actorsExperience",
        "averageRating",
    ],
    1,
)
genre_10 = sample_10k.drop(
    [
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
    ],
    1,
)
genre_5 = genre_10.drop(
    ["isTalkShow", "isRomance", "isFamily", "isNews", "isAnimation"], 1
)
genre_2 = genre_5.drop(["isComedy", "isShort", "isDocumentary"], 1)

# Plots of predictor strength for rating against each feature
plt.title("Rating Prediction Strength by Feature")
plt.xlabel("Feature")
plt.ylabel("Prediction Strength")

result = rating.train_rating(sample_10k)["strengths"]

for key, value in result.items():
    plt.bar(key, value)

axes = plt.gca()
axes.set_ylim([0, 1])

plt.savefig("visuals/rating_predictor_strength.png", bbox_inches="tight")
plt.clf()



# Plots of predictor strength for genre against each feature
plt.title("Genre Prediction Strength by Feature")
plt.xlabel("Feature")
plt.ylabel("Prediction Strength")

result = genre.train_genres(sample_10k)["strengths"]

for key, value in result.items():
    plt.bar(key, value)

axes = plt.gca()
axes.set_ylim([0, 1])

plt.savefig("visuals/genre_predictor_strength.png", bbox_inches="tight")
plt.clf()



# Plots of accuracy predicting rating with all (27), 10, 5, and 2 genres
plt.title("Accuracy Predicting Rating by Quantity of Genres")
plt.xlabel("Quantity of Genres")
plt.ylabel("Accuracy")
plt.bar("27", rating.train_rating(sample_10k)["accuracy"])
plt.bar("10", rating.train_rating(genre_10)["accuracy"])
plt.bar("5", rating.train_rating(genre_5)["accuracy"])
plt.bar("2", rating.train_rating(genre_2)["accuracy"])

axes = plt.gca()
axes.set_ylim([0, 1])

plt.savefig("visuals/rating_accuracy_quantity_genre.png", bbox_inches="tight")
plt.clf()



# Plots of accuracy predicting rating with 10k, 5k, and 2k samples
plt.title("Accuracy Predicting Rating by Sample Size")
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.bar("10k", rating.train_rating(sample_10k)["accuracy"])
plt.bar("5k", rating.train_rating(sample_5k)["accuracy"])
plt.bar("2k", rating.train_rating(sample_2k)["accuracy"])

axes = plt.gca()
axes.set_ylim([0, 1])

plt.savefig("visuals/rating_accuracy_sample_size.png", bbox_inches="tight")
plt.clf()
