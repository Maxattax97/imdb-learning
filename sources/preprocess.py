#!/usr/bin/env python3
import csv
import json
import sys
import numpy as np

csv.field_size_limit(sys.maxsize)

# Target data structure:
# { type, isAdult, startYear, runtimeMinutes, isDrama, isComedy, isShort, isDocumentary, isTalkShow, isRomance, isFamily, isNews, isAnimation, isRealityTV, isMusic, isCrime, isAction, isAdventure, isGameShow, isMystery, isSport, isFantasy, isHorror, isThriller, isSciFi, isHistory, isBiography, isMusical, isWestern, isWar, isFilmNoir, averageRating, numVotes, directorsExperience, actorsExperience }

GENRES = [
    "Drama",
    "Comedy",
    "Short",
    "Documentary",
    "TalkShow",
    "Romance",
    "Family",
    "News",
    "Animation",
    "RealityTV",
    "Music",
    "Crime",
    "Adventure",
    "GameShow",
    "Mystery",
    "Sport",
    "Fantasy",
    "Horror",
    "Thriller",
    "SciFi",
    "History",
    "Biography",
    "Musical",
    "Western",
    "War",
    "FilmNoir",
]
PROGRESS_NOTIFY_RATE = 10000


def parse_genres(obj, genre_string):
    split_genres = genre_string.split(",")

    for genre in GENRES:
        obj["is" + genre] = False
        if genre in split_genres:
            obj["is" + genre] = True


def progress_notifier(count):
    if ((count) % PROGRESS_NOTIFY_RATE) == 0:
        print(".", end="", flush=True)


film_storage = {}
crew_storage = {}

try:
    with open("data/crew.json", "r") as crew_file:
        print("Loading cached data/crew.json ...")
        crew_storage = json.load(crew_file)
except:
    print("Loading data/title.principals.tsv ...")
    with open("data/title.principals.tsv", "r") as tsv_in:
        tsv_in = csv.reader(tsv_in, delimiter="\t")

        crew_count = 0
        for row in tsv_in:
            member = None

            if row[0] == "tconst":
                continue  # The first row contains labels for columns.
            if (
                row[3] != "director"
                and row[3] != "actor"
                and row[3] != "actress"
                and row[3] != "self"
                and row[3] != "producer"
            ):
                continue

            try:
                member = crew_storage[row[2]]
            except:
                member = {}

            if "type" not in member:
                member["type"] = "actor"
                if row[3] == "producer":
                    member["type"] = "producer"
                elif row[3] == "director":
                    member["type"] = "director"
                member["experience"] = 1
                crew_count += 1
                progress_notifier(crew_count)
            else:
                member["experience"] += 1

            crew_storage[row[2]] = member  # Store in a dictionary indexed by ID.
        print("\nParsed " + str(crew_count) + " crew members.")

    print("Dumping crew storage to file ...")
    with open("data/crew.json", "w") as crew_file:
        json.dump(crew_storage, crew_file)

try:
    with open("data/films.json", "r") as films_file:
        print("Loading cached data/films.json ...")
        film_storage = json.load(films_file)
except:
    print("Loading data/title.basics.tsv ...")
    with open("data/title.basics.tsv", "r") as tsv_in:
        tsv_in = csv.reader(tsv_in, delimiter="\t")

        film_count = 0
        for row in tsv_in:
            film = {}

            if row[0] == "tconst":
                continue  # The first row contains labels for columns.
            if (
                row[1] == "tvSeries"
                or row[1] == "tvEpisode"
                or row[1] == "tvMiniSeries"
            ):
                continue

            exit = False
            for col in [0, 1, 2, 4, 5, 7, 8]:
                if row[col] == "\\N":
                    exit = True
                    break
            if exit:  # Skip films with missing data.
                continue

            film["type"] = row[1]
            film["title"] = row[2]
            film["isAdult"] = False

            if row[4] == "1":
                film["isAdult"] = True

            # if row[5] == "\\N":
            # continue
            elif int(row[5]) <= (2018 - 50):
                continue  # Skip films older than before color television.
            film["startYear"] = row[5]

            # if row[7] == "\\N":
            # continue  # Null field, skip to next.
            film["runtimeMinutes"] = int(row[7])

            parse_genres(film, row[8])
            film["averageRating"] = 0
            film["numVotes"] = 0
            film["directorsExperience"] = 0
            film["actorsExperience"] = 0
            film["producersExperience"] = 0

            film_storage[row[0]] = film  # Store in a dictionary indexed by ID.
            film_count += 1
            progress_notifier(film_count)
        print("\nParsed " + str(film_count) + " films.")

        print("Loading data/title.akas.tsv ...")
        with open("data/title.akas.tsv", "r") as tsv_in:
            tsv_in = csv.reader(tsv_in, delimiter="\t")

            deleted_count = 0

            for row in tsv_in:
                if row[0] == "tconst":
                    continue  # The first row contains labels for columns.

                try:
                    film = film_storage[row[0]]
                except:
                    continue
                if not film:
                    continue

                # Delete films which are not originally released in the US.
                if row[7] == "1" and row[3] != "US":
                    deleted_count += 1
                    progress_notifier(deleted_count)
                    del film_storage[row[0]]
            print("\nDeleted " + str(deleted_count) + " entries not made in the US.")

        print("Loading data/title.ratings.tsv ...")
        with open("data/title.ratings.tsv", "r") as tsv_in:
            tsv_in = csv.reader(tsv_in, delimiter="\t")

            update_count = 0
            delete_count = 0
            for row in tsv_in:
                if row[0] == "tconst":
                    continue  # The first row contains labels for columns.

                try:
                    film = film_storage[row[0]]
                except:
                    continue
                if not film:
                    continue

                if row[1] == "\\N":
                    progress_notifier(update_count + deleted_count)
                    del film_storage[row[0]]
                    deleted_count += 1
                    continue
                if row[2] == "\\N" or int(row[2]) == 0:
                    progress_notifier(update_count + deleted_count)
                    del film_storage[row[0]]
                    deleted_count += 1
                    continue

                film["averageRating"] = float(row[1])
                film["numVotes"] = int(row[2])
                update_count += 1
                progress_notifier(update_count + deleted_count)
            print(
                "\nUpdated "
                + str(update_count)
                + " film ratings, and deleted "
                + str(deleted_count)
                + " films in the process."
            )

        print("Purging films without ratings ...")
        remove_list = [k for k in film_storage if film_storage[k]["numVotes"] <= 0]
        purge_count = 0
        for k in remove_list:
            progress_notifier(purge_count)
            del film_storage[k]
            purge_count += 1
        print("\nPurged " + str(purge_count) + " films with missing ratings.")

        print("Loading data/title.principals.tsv ...")
        with open("data/title.principals.tsv", "r") as tsv_in:
            tsv_in = csv.reader(tsv_in, delimiter="\t")

            match_count = 0
            for row in tsv_in:
                member = None

                if row[0] == "tconst":
                    continue  # The first row contains labels for columns.
                if (
                    row[3] != "director"
                    and row[3] != "actor"
                    and row[3] != "actress"
                    and row[3] != "self"
                    and row[3] != "producer"
                ):
                    continue

                try:
                    film = film_storage[row[0]]
                except:
                    continue

                try:
                    member = crew_storage[row[2]]
                except:
                    continue

                if member["type"] == "director":
                    film["directorsExperience"] += member["experience"]
                    match_count += 1

                if member["type"] == "actor":
                    film["actorsExperience"] += member["experience"]
                    match_count += 1

                if member["type"] == "producer":
                    film["producersExperience"] += member["experience"]
                    match_count += 1

                progress_notifier(match_count)

            print("\nMatched " + str(match_count) + " crew members to films.")

        print("Dumping film storage to file ...")
        with open("data/films.json", "w") as films_file:
            json.dump(film_storage, films_file)

print("Sampling 10,000 films ...")
random_sample = np.random.choice(list(film_storage.keys()), 10000, replace=False)

print("Writing dataset to data/dataset.csv ...")
with open("data/dataset.csv", "w") as csv_out:
    field_names = list(film_storage[random_sample[0]].keys())
    writer = csv.DictWriter(csv_out, fieldnames=field_names)

    writer.writeheader()

    for key in random_sample:
        writer.writerow(film_storage[key])

print("Writing dataset to data/dataset.json ...")
with open("data/dataset.json", "w") as json_out:
    dataset_dict = {}
    for key in random_sample:
        dataset_dict[key] = film_storage[key]
    json.dump(dataset_dict, json_out)
