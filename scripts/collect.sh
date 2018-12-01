#! /bin/bash

wget --directory-prefix=../dataset/ --quiet --show-progress --no-clobber --input-file=sources.txt
gunzip --verbose --force --keep ../dataset/*.gz
