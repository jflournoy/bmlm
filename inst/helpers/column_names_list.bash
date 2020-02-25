#!/bin/bash

file=$1
outfile=$2

head -n 39 "${file}" | tail -n 1 | awk -F, '{for(n = 1; n < NF; ++n) {printf("%05d %s",n,$n); printf("\n");} }' | tee "${outfile}"
