#!/usr/bin/env bash

pushd .
source ../../.env/bin/activate
popd

echo "Python version $(python --version)"

cd ..


MODELS="$(NAME=LSTM PARAMS='as a') $()"

echo $MODELS

for m in $MODELS;
  do
  for i in $(seq 1 20);
    do
    echo "Experiment model ${m}  $i";
    #python main.py --model=$MODEL -t babi:task10k:$i --dict-file=/tmp/dict.txt --num-its=1000
  done
done
