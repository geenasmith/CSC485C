#!/bin/bash
# ooops. sort for bad files
shopt -s extglob


for f in "time/new*"; do
    (( $(wc -l < "$f") < 3 )) && echo "$f"
done