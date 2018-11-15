# imdb-learning
Machine learning project over the IMDB dataset to predict success of movies as well as their genres.

This project is written in Python 3.x and uses a virtual environment.

## Setup
Create a Python 3.x virtual environment and install the necessary requirements:
```
python3 -m venv env
source ./env/bin/activate
pip3 install -r ./requirements.txt
```

## Overview

### `preprocess.py`
Used for converting IMDB's dataset into a more digestable form for our algorithms. It also reduces the dataset down to 10,000 samples.

### `model.py`
Contains the logic to train statistical models to eventually predict movie rating and genre.

### `validate.py`
Validates the models using k-folds cross validation.

### `visualize.py`
Produces plots, charts, and graphs to represent the results of the models and give insight to the dataset.

### `data/`
Contains the unzipped IMDB dataset's `.tsv` files, and the post-processed `dataset.csv` which is ready to be put into the model. IMDB's original dataset is obfuscated via `.gitignore` for purposes of size. Access it at [IMDB's website](https://www.imdb.com/interfaces/).

### `requirements.txt`
Contains a list of packages and their respective versions for usage in Python 3.x virtual environments. Load with `pip3 install -r requirements.txt` after entering a virtual environment. This may be updated using the command `pip3 freeze > requirements.txt`.

## Contributors
Kevin Kochpatcharin (kkochpat@purdue.edu), Max O’Cull (mocull@purdue.edu), Nameer Qureshi (nquresh@purdue.edu), Ryan Sullivan (sulli196@purdue.edu)
