#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random

import rating
import genre

plt.style.use("seaborn")

MEGA_FIGURE = True
FIX_LIMITS = True
IMAGE_DPI = 100
IMAGE_SIZE = (4, 3)

csv_file = open("data/dataset.csv", "r")
data = pandas.read_csv("data/dataset.csv")

# We never care about these.
data = data.drop(["type", "title", "startYear"], 1)

sample_10k = data.sample(frac=1).reset_index(drop=True)
sample_5k = data.sample(n=5000).reset_index(drop=True)
sample_2k = data.sample(n=2000).reset_index(drop=True)

genre_only = sample_10k.drop(
    [
        "runtimeMinutes",
        "numVotes",
        "directorsExperience",
        "actorsExperience",
        "producersExperience",
        # We need this one.
        #  "averageRating",
    ],
    1,
)
genre_10 = sample_10k.drop(
    [
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
    ["isRomance", "isFamily", "isNews", "isAnimation", "isRealityTV"], 1
)
genre_2 = genre_5.drop(["isTalkShow", "isShort", "isDocumentary"], 1)
genre_none = genre_2.drop(["isDrama", "isComedy"], 1)


def plot_rating_predictor_strength():
    # Plots of predictor strength for rating against each feature
    plt.title("Rating Prediction Strength by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Prediction Strength")

    result = rating.train_rating(sample_10k)["strengths"]

    for key, value in result.items():
        plt.bar(key, value)

    if FIX_LIMITS:
        axes = plt.gca()
        axes.set_ylim([0, 1])

    if not MEGA_FIGURE:
        plt.savefig(
            "visuals/rating_predictor_strength.png",
            bbox_inches="tight",
            figsize=IMAGE_SIZE,
            dpi=IMAGE_DPI,
        )


def plot_genre_predictor_strength():
    # Plots of predictor strength for genre against each feature
    plt.title("Genre Prediction Strength by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Prediction Strength")

    result = genre.train_genres(sample_10k)["strengths"]

    for key, value in result.items():
        plt.bar(key, value)

    if FIX_LIMITS:
        axes = plt.gca()
        axes.set_ylim([0, 1])

    if not MEGA_FIGURE:
        plt.savefig(
            "visuals/genre_predictor_strength.png",
            bbox_inches="tight",
            figsize=IMAGE_SIZE,
            dpi=IMAGE_DPI,
        )


def plot_rating_accuracy_quantity_genre():
    # Plots of accuracy predicting rating with all (27), 10, 5, and 2 genres
    plt.title("Accuracy Predicting Rating by Quantity of Genres")
    plt.xlabel("Quantity of Genres")
    plt.ylabel("Accuracy")
    plt.bar("27", rating.train_rating(sample_10k)["accuracy"])
    plt.bar("10", rating.train_rating(genre_10)["accuracy"])
    plt.bar("5", rating.train_rating(genre_5)["accuracy"])
    plt.bar("2", rating.train_rating(genre_2)["accuracy"])

    if FIX_LIMITS:
        axes = plt.gca()
        axes.set_ylim([0, 1])

    if not MEGA_FIGURE:
        plt.savefig(
            "visuals/rating_accuracy_quantity_genre.png",
            bbox_inches="tight",
            figsize=IMAGE_SIZE,
            dpi=IMAGE_DPI,
        )


def plot_rating_accuracy_sample_size():
    # Plots of accuracy predicting rating with 10k, 5k, and 2k samples
    plt.title("Accuracy Predicting Rating by Sample Size")
    plt.xlabel("Sample Size")
    plt.ylabel("Accuracy")
    plt.bar("10k", rating.train_rating(sample_10k)["accuracy"])
    plt.bar("5k", rating.train_rating(sample_5k)["accuracy"])
    plt.bar("2k", rating.train_rating(sample_2k)["accuracy"])

    if FIX_LIMITS:
        axes = plt.gca()
        axes.set_ylim([0, 1])

    if not MEGA_FIGURE:
        plt.savefig(
            "visuals/rating_accuracy_sample_size.png",
            bbox_inches="tight",
            figsize=IMAGE_SIZE,
            dpi=IMAGE_DPI,
        )


def plot_rating_accuracy_wwo_genre():
    # Plots of accuracy predicting rating with and without non-genre features
    plt.title("Accuracy Predicting Rating With/out Non-Genre")
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    plt.bar("All Features", rating.train_rating(sample_10k)["accuracy"])
    plt.bar("Only Genres", rating.train_rating(genre_only)["accuracy"])
    plt.bar("Only Non-Genres", rating.train_rating(genre_none)["accuracy"])

    if FIX_LIMITS:
        axes = plt.gca()
        axes.set_ylim([0, 1])

    if not MEGA_FIGURE:
        plt.savefig(
            "visuals/rating_accuracy_wwo_genre.png",
            bbox_inches="tight",
            figsize=IMAGE_SIZE,
            dpi=IMAGE_DPI,
        )


def plot_genre_accuracy_sample_size():
    # Plots of accuracy predicting genre with 10k, 5k, and 2k samples
    plt.title("Accuracy Predicting Genre by Sample Size")
    plt.xlabel("Sample Size")
    plt.ylabel("Accuracy")
    plt.bar("10k", genre.train_genres(sample_10k)["accuracy"])
    plt.bar("5k", genre.train_genres(sample_5k)["accuracy"])
    plt.bar("2k", genre.train_genres(sample_2k)["accuracy"])

    if FIX_LIMITS:
        axes = plt.gca()
        axes.set_ylim([0, 1])

    if not MEGA_FIGURE:
        plt.savefig(
            "visuals/genre_accuracy_sample_size.png",
            bbox_inches="tight",
            figsize=IMAGE_SIZE,
            dpi=IMAGE_DPI,
        )


def plot_genre_accuracy_quantity_genre():
    # Plots of accuracy predicting genre with all (27), 10, 5, and 2 genres
    plt.title("Accuracy Predicting Genre by Quantity of Genres")
    plt.xlabel("Quantity of Genres")
    plt.ylabel("Accuracy")
    plt.bar("27", genre.train_genres(sample_10k)["accuracy"])
    plt.bar("10", genre.train_genres(genre_10)["accuracy"])
    plt.bar("5", genre.train_genres(genre_5)["accuracy"])
    plt.bar("2", genre.train_genres(genre_2)["accuracy"])

    if FIX_LIMITS:
        axes = plt.gca()
        axes.set_ylim([0, 1])

    if not MEGA_FIGURE:
        plt.savefig(
            "visuals/genre_accuracy_quantity_genre.png",
            bbox_inches="tight",
            figsize=IMAGE_SIZE,
            dpi=IMAGE_DPI,
        )


def plot_rating_numvotes():
    plt.title("Rating vs. Number of Votes")
    plt.xlabel("Rating")
    plt.ylabel("Number of Votes")
    scatter_data = {
        "rating": sample_10k["averageRating"].tolist(),
        "votes": sample_10k["numVotes"].tolist(),
        "actors": sample_10k["actorsExperience"].tolist(),
        "size": sample_10k["actorsExperience"].tolist(),
    }

    actor_sum = max(scatter_data["size"])
    scatter_data["size"] = [float(i) / actor_sum for i in scatter_data["size"]]
    scatter_data["size"] = [x * 700 for x in scatter_data["size"]]

    plt.scatter(
        "rating", "votes", s="size", c="actors", cmap="viridis", data=scatter_data
    )
    cbar = plt.colorbar()
    cbar.set_label("Actor Experience")

    if FIX_LIMITS:
        axes = plt.gca()
        axes.set_xlim([3, 9])
        axes.set_ylim([-1000, 60000])

    if not MEGA_FIGURE:
        plt.savefig(
            "visuals/rating_votes.png",
            bbox_inches="tight",
            figsize=IMAGE_SIZE,
            dpi=IMAGE_DPI,
        )


plots_to_generate = [
    plot_rating_numvotes,
    plot_rating_predictor_strength,
    plot_genre_predictor_strength,
    plot_rating_accuracy_quantity_genre,
    plot_rating_accuracy_sample_size,
    plot_rating_accuracy_wwo_genre,
    plot_genre_accuracy_sample_size,
    plot_genre_accuracy_quantity_genre,
]

if MEGA_FIGURE:
    i = 1
    plt.figure(1)
    for func in plots_to_generate:
        plt.subplot(2, 4, i)
        func()
        i += 1
    plt.tight_layout()
    plt.savefig("visuals/all.png", figsize=IMAGE_SIZE, dpi=IMAGE_DPI)
    plt.show()

plt.figure(2)
MEGA_FIGURE = False
for func in plots_to_generate:
    func()
    plt.clf()
