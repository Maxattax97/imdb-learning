#!/usr/bin/env python3
import os
import sys
import inspect
from pathlib import Path
from subprocess import call

cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])
)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# Use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(
    os.path.abspath(
        os.path.join(
            os.path.split(inspect.getfile(inspect.currentframe()))[0], "sources"
        )
    )
)
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

preprocess_needed = False

data_csv = Path("./dataset/dataset.csv")
data_json = Path("./dataset/dataset.json")
if (not data_csv.is_file()) or (not data_json.is_file()):
    preprocess_needed = True
    print("Collecting dataset from IMDB ...")
    os.chdir("./scripts/")
    call(["sh", "./scripts/collect.sh"])
else:
    print("Pre-processed dataset detected, skipping download ...")

print("Loading program ...")
os.chdir("./sources/")
#  import validate
if preprocess_needed:
    import preprocess
import rating
import genre
import visualize
