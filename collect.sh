#! /bin/bash

wget --directory-prefix=data/ --quiet --show-progress --no-clobber --input-file=sources.txt
gunzip --verbose --force --keep data/*.gz
