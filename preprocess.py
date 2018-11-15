#! /usr/bin/python3
import csv
import json
import sys
import numpy as np

csv.field_size_limit(sys.maxsize)

# Target data structure:
# { type, isAdult, startYear, runtimeMinutes, isDrama, isComedy, isShort, isDocumentary, isTalkShow, isRomance, isFamily, isNews, isAnimation, isRealityTV, isMusic, isCrime, isAction, isAdventure, isGameShow, isMystery, isSport, isFantasy, isHorror, isThriller, isSciFi, isHistory, isBiography, isMusical, isWestern, isWar, isFilmNoir, averageRating, numVotes, directorsExperience, actorsExperience }

GENRES = ['Drama', 'Comedy', 'Short', 'Documentary', 'TalkShow', 'Romance', 'Family', 'News', 'Animation', 'RealityTV','Music', 'Crime', 'Adventure', 'GameShow', 'Mystery', 'Sport', 'Fantasy', 'Horror', 'Thriller', 'SciFi', 'History', 'Biography', 'Musical', 'Western', 'War', 'FilmNoir']

def parse_genres(obj, genre_string):
    split_genres = genre_string.split(',')

    for genre in GENRES:
        obj['is' + genre] = False
        if genre in split_genres:
            obj['is' + genre] = True

film_storage = {}
crew_storage = {}

try:
    with open('data/crew.json', 'r') as crew_file:
        print('Loading cached data/crew.json ...')
        crew_storage = json.load(crew_file)
except:
    print('Loading data/title.principals.tsv ...')
    with open('data/title.principals.tsv', 'r') as tsv_in:
        tsv_in = csv.reader(tsv_in, delimiter='\t')

        crew_count = 0
        for row in tsv_in:
            member = None

            if row[0] == 'tconst':
                continue # The first row contains labels for columns.
            if row[3] != 'director' and row[3] != 'actor' and row[3] != 'actress' and row[3] != 'self':
                continue

            try:
                member = crew_storage[row[2]]
            except:
                member = {}

            if 'type' not in member:
                member['type'] = 'actor'
                if row[3] == 'director':
                    member['type'] = 'director'
                member['experience'] = 1
                crew_count += 1
            else:
                member['experience'] += 1

            crew_storage[row[2]] = member # Store in a dictionary indexed by ID.
        print(' > Parsed ' + str(crew_count) + ' crew members.')

    print('Dumping crew storage to file ...')
    with open('data/crew.json', 'w') as crew_file:
        json.dump(crew_storage, crew_file)

try:
    with open('data/films.json', 'r') as films_file:
        print('Loading cached data/films.json ...')
        film_storage = json.load(films_file)
except:
    print('Loading data/title.basics.tsv ...')
    with open('data/title.basics.tsv', 'r') as tsv_in:
        tsv_in = csv.reader(tsv_in, delimiter='\t')

        film_count = 0
        for row in tsv_in:
            film = {}

            if row[0] == 'tconst':
                continue # The first row contains labels for columns.
            if row[1] == 'tvSeries' or row[1] == 'tvEpisode' or row[1] == 'tvMiniSeries':
                continue

            film['type'] = row[1]
            film['title'] = row[2]
            film['isAdult'] = False
            if row[4] == '1':
                film['isAdult'] = True
            film['startYear'] = row[5]
            if row[7] == '\\N':
                continue # Null field, skip to next.
            film['runtimeMinutes'] = int(row[7])

            parse_genres(film, row[8])
            film['averageRating'] = 0
            film['numVotes'] = 0
            film['directorsExperience'] = 0
            film['actorsExperience'] = 0

            film_storage[row[0]] = film # Store in a dictionary indexed by ID.
            film_count += 1
        print(' > Parsed ' + str(film_count) + ' films.')

    print('Loading data/title.akas.tsv ...')
    with open('data/title.akas.tsv', 'r') as tsv_in:
        tsv_in = csv.reader(tsv_in, delimiter='\t')

        deleted_count = 0

        for row in tsv_in:
            if row[0] == 'tconst':
                continue # The first row contains labels for columns.

            try:
                film = film_storage[row[0]]
            except:
                continue
            if not film:
                continue

            # Delete films which are not originally released in the US.
            if row[7] == '1' and row[3] != 'US':
                deleted_count += 1
                del film_storage[row[0]]
        print(' > Deleted ' + str(deleted_count) + ' entries not made in the US.')

    print('Dumping film storage to file ...')
    with open('data/films.json', 'w') as films_file:
        json.dump(film_storage, films_file)

    print('Loading data/title.principals.tsv ...')
    with open('data/title.principals.tsv', 'r') as tsv_in:
        tsv_in = csv.reader(tsv_in, delimiter='\t')

        match_count = 0
        for row in tsv_in:
            member = None

            if row[0] == 'tconst':
                continue # The first row contains labels for columns.
            if row[3] != 'director' and row[3] != 'actor' and row[3] != 'actress' and row[3] != 'self':
                continue

            try:
                film = film_storage[row[0]]
            except:
                continue

            try:
                member = crew_storage[row[2]]
            except:
                continue

            if member['type'] == 'director':
                film['directorsExperience'] += member['experience']
                match_count += 1

            if member['type'] == 'actor':
                film['actorsExperience'] += member['experience']
                match_count += 1

        print(' > Matched ' + str(match_count) + ' crew members to films.')

    print('Loading data/title.ratings.tsv ...')
    with open('data/title.ratings.tsv', 'r') as tsv_in:
        tsv_in = csv.reader(tsv_in, delimiter='\t')

        for row in tsv_in:
            if row[0] == 'tconst':
                continue # The first row contains labels for columns.

            try:
                film = film_storage[row[0]]
            except:
                continue
            if not film:
                continue

            if row[1] == '\\N':
                del film_storage[row[0]]
                continue
            if row[2] == '\\N':
                del film_storage[row[0]]
                continue

            film['averageRating'] = float(row[1])
            film['numVotes'] = int(row[2])

    print('Dumping film storage to file ...')
    with open('data/films.json', 'w') as films_file:
        json.dump(film_storage, films_file)

print('Sampling 10,000 films ...')
random_sample = np.random.choice(list(film_storage.keys()), 10000, replace=False)

print('Writing dataset to data/dataset.csv ...')
with open('data/dataset.csv', 'w') as csv_out:
    print(film_storage[random_sample[0]].keys())
    field_names = list(film_storage[random_sample[0]].keys())
    writer = csv.DictWriter(csv_out, fieldnames=field_names)

    writer.writeheader()

    for key in random_sample:
        writer.writerow(film_storage[key])

