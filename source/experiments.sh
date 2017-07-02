#!/usr/bin/env bash

pushd .
source env/bin/activate
popd

echo "Python version $(python --version)"

MODEL=$1
for i in $(seq 1 20);
    do
    echo "Experiment model $MODEL $i";
    python main.py --model=$MODEL -t babi:task10k:$i --dict-file=/tmp/dict.txt --num-its=1000
done
