#!/bin/bash

files=${@:2}
range=$1

echo "Reducing to this column range: ${range}"

for file in $files; do
    echo "Briefing ${file}..."
    filenoext=${file%.csv}
    cut -d, -f${range} ${file} > "${filenoext}_brief.csv"
done
