#!/usr/bin/env bash

pushd .
source ../../.env/bin/activate
popd

echo "Python version $(python --version)"

cd ..
MODELS="agents.agent_lstm:LSTMAgent"

echo $MODELS

for m in $MODELS;
  do
  for i in $(seq 1 20);
    do
    echo "Experiment model ${m}  $i";
    python main.py -m $m -t babi:task10k:$i --dict-file=/tmp/dict10.txt -vtim 1200 -ltim 60 -vp 5 -bs 256
  done
done
