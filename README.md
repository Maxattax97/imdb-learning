# imdb-learning
Machine learning project over the IMDB dataset to predict success of movies as well as their genres.

This project is written in Python 3.x and uses a virtual environment.

## Setup
Create a Python 3.x virtual environment and install the necessary requirements:
```
python3 -m venv env
source ./env/bin/activate # You must run this in EVERY fresh session.
pip3 install -r ./libraries/requirements.txt
```

## Overview

### `run.py`
Runs everything, with some automation included for collected and preprocessing the dataset (if necessary).

### `visuals/`
Contains images of plots that are generated after training and validating the dataset.

### `sources/preprocess.py`
Used for converting IMDB's dataset into a more digestable form for our algorithms. It also reduces the dataset down to 10,000 samples.

### `sources/genre.py` and `sources/rating.py`
Contains the logic to train statistical models to eventually predict movie rating and genre.

### `sources/validate.py`
Validates the models using k-folds cross validation.

### `sources/visualize.py`
Produces plots, charts, and graphs to represent the results of the models and give insight to the dataset.

### `scripts/collect.sh`
Downloads all datasets originally from IMDB (listed in `./scripts/sources.txt`) and `gunzip`s them to the `dataset/` directory.

### `dataset/`
Contains the unzipped IMDB dataset's `.tsv` files, and the post-processed `dataset.csv` and `dataset.json` which is ready to be put into the model. IMDB's original dataset is obfuscated via `.gitignore` for purposes of size. Access it at [IMDB's website](https://www.imdb.com/interfaces/).

### `libraries/requirements.txt`
Contains a list of packages and their respective versions for usage in Python 3.x virtual environments. Load with `pip3 install -r ./libraries/requirements.txt` after entering a virtual environment. This may be updated using the command `pip3 freeze > ./libraries/requirements.txt`.

## Contributors
Kevin Kochpatcharin (kkochpat@purdue.edu), Max O’Cull (mocull@purdue.edu), Nameer Qureshi (nquresh@purdue.edu), Ryan Sullivan (sulli196@purdue.edu)
