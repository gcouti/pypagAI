#!/usr/bin/env bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/;

SCRIPT="python examples/main.py"
COMMON_PARAMETERS="model_cfg.epochs=1000 model_cfg.verbose=False"
TASKS=(
    "model_cfg.model=pypagai.models.model_embed_lstm.EmbedLSTM"
    "model_cfg.model=pypagai.models.model_lstm.SimpleLSTM"
    "model_cfg.model=pypagai.models.model_encoder.EncoderModel"
    "model_cfg.model=pypagai.models.model_n2nmemory.N2NMemory"
    "model_cfg.model=pypagai.models.model_rn.RN dataset_cfg.strip_sentences=True"
    "model_cfg.model=pypagai.models.model_rnn.RNNModel"
    "model_cfg.model=pypagai.models.model_scikit.RFModel"
    "model_cfg.model=pypagai.models.model_scikit.SVMModel"
)

DATA_SETS=(
    "dataset_cfg.task=1"
    "dataset_cfg.task=2"
    "dataset_cfg.task=3"
    "dataset_cfg.task=4"
    "dataset_cfg.task=5"
    "dataset_cfg.task=6"
    "dataset_cfg.task=7"
    "dataset_cfg.task=8"
    "dataset_cfg.task=9"
    "dataset_cfg.task=10"
    "dataset_cfg.task=11"
    "dataset_cfg.task=12"
    "dataset_cfg.task=13"
    "dataset_cfg.task=14"
    "dataset_cfg.task=15"
    "dataset_cfg.task=16"
    "dataset_cfg.task=17"
    "dataset_cfg.task=18"
    "dataset_cfg.task=19"
    "dataset_cfg.task=20"
)

source .env/bin/activate
cd source/
python setup.py install

for t in "${TASKS[@]}"; do
    for d in "${DATA_SETS[@]}"; do
        COMMAND="$SCRIPT with $d $t $COMMON_PARAMETERS"
        echo "#####################################"
        echo "[START] Invoke new experiment $COMMAND"
        echo "#####################################"

        ${COMMAND}
    done
done
