#!/usr/bin/env bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/;

SCRIPT="python examples/main.py"
COMMON_PARAMETERS=""
TASKS=(
    "model_cfg.model=pypagai.models.model_embed_lstm.EmbedLSTM"
    "model_cfg.model=pypagai.models.model_lstm.SimpleLSTM"
)

DATA_SETS=(
    "dataset_cfg.task='1' model_cfg.epochs=1"
#    "babi:10k:2"
#    "babi:10k:3"
#    "babi:10k:4"
#    "babi:10k:5"
#    "babi:10k:6"
#    "babi:10k:7"
#    "babi:10k:8"
#    "babi:10k:9"
#    "babi:10k:10"
#    "babi:10k:11"
#    "babi:10k:12"
#    "babi:10k:13"
#    "babi:10k:14"
#    "babi:10k:15"
#    "babi:10k:17"
#    "babi:10k:18"
#    "babi:10k:19"
#    "babi:10k:20"
)

source .env/bin/activate
cd source/
python setup.py install

for t in "${TASKS[@]}"; do
    for d in "${DATA_SETS[@]}"; do
        COMMAND="$SCRIPT with $d $t"
        echo "Experiment $COMMAND"
        ${COMMAND}
    done
done
