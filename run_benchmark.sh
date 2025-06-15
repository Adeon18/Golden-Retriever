#!/bin/bash

MODEL=${1:-"intfloat/multilingual-e5-base"}
BENCHMARK=${2:-"UkrWikiRetrieval"}

python benchmarks.py "$MODEL" "$BENCHMARK"