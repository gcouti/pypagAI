#!/usr/bin/env bash

SCRIPT="python examples/main.py"
TASKS=(
    "-m lstm "
    "-m lstm -hidden 64"
    "-m lstm -hidden 256"
)

DATA_SETS=(
    "babi:10k:1"
    "babi:10k:2"
    "babi:10k:3"
    "babi:10k:4"
    "babi:10k:5"
    "babi:10k:6"
    "babi:10k:7"
    "babi:10k:8"
    "babi:10k:9"
    "babi:10k:10"
    "babi:10k:11"
    "babi:10k:12"
    "babi:10k:13"
    "babi:10k:14"
    "babi:10k:15"
    "babi:10k:17"
    "babi:10k:18"
    "babi:10k:19"
    "babi:10k:20"
)

source .env/bin/activate
cd source/
python setup.py install

for t in "${TASKS[@]}"; do
    for d in "${DATA_SETS[@]}"; do
        COMMAND="$SCRIPT -d $d $t"
        echo "Experiment $COMMAND"
        ${COMMAND}
    done
done
