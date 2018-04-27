#!/usr/bin/env bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/;

SCRIPT=".env/bin/python source/examples/qa_experiment.py"
COMMON_PARAMETERS="model_cfg.verbose=True"
COMMON_DEEP_LEARNING="model_cfg.epochs=1000"
TASKS=(
#  "rf_config"
#  "svm_config"
  "model_cfg.model=pypagai.models.model_embed_lstm.EmbedLSTM"
  "model_cfg.model=pypagai.models.model_lstm.SimpleLSTM"
  "model_cfg.model=pypagai.models.model_encoder.EncoderModel"
  "model_cfg.model=pypagai.models.model_n2nmemory.N2NMemory"
  "model_cfg.model=pypagai.models.model_rn.RN dataset_cfg.strip_sentences=True"
  "model_cfg.model=pypagai.models.model_rnn.RNNModel model_cfg.batch=128"
)

DATA_SETS=(
    "dataset_cfg.task=1"
    "dataset_cfg.task=2"
#    "dataset_cfg.task=3"
#    "dataset_cfg.task=4"
#    "dataset_cfg.task=5"
#    "dataset_cfg.task=6"
#    "dataset_cfg.task=7"
#    "dataset_cfg.task=8"
#    "dataset_cfg.task=9"
#    "dataset_cfg.task=10"
#    "dataset_cfg.task=11"
#    "dataset_cfg.task=12"
#    "dataset_cfg.task=13"
#    "dataset_cfg.task=14"
#    "dataset_cfg.task=15"
#    "dataset_cfg.task=16"
#    "dataset_cfg.task=17"
#    "dataset_cfg.task=18"
#    "dataset_cfg.task=19"
#    "dataset_cfg.task=20"
)

for t in "${TASKS[@]}"; do
    for d in "${DATA_SETS[@]}"; do
        COMMAND="$SCRIPT with $d $t $COMMON_PARAMETERS -n $t"

        echo "#########################################################################################################"
        echo "[START] Invoke new experiment $COMMAND"
        echo "#########################################################################################################"

        ${COMMAND}

        echo "#########################################################################################################"
        echo "[FINISH] $COMMAND"
        echo "#########################################################################################################"


    done
done
