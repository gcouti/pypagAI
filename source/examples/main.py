"""
Execute experiments with PypagAI framework

This main has the purpose realize experiments train and experiment models. It implement the same interface of the train_model.py
file that was implemented in the PypagAI framework.

You can run this file like:

```
    python source/main.py -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
```

To evaluate model run with -vtim or --validation-every-n-secs. It will eval model every X seconds (defau)
To save model in the end of the execution run with --save or -s

To see other options -h

"""
import logging


from pypagai import settings
from pypagai.models.model_embed_lstm import EmbedLSTM
from pypagai.models.model_lstm import SimpleLSTM
from pypagai.preprocessing.dataset_babi import BaBIDataset
from pypagai.preprocessing.parser import SimpleParser
from sacred import Experiment, Ingredient

from pypagai.util.class_loader import ModelLoader

logging.basicConfig(level=settings.LOG_LEVEL)
LOG = logging.getLogger(__name__)

data_ingredient = Ingredient('dataset_cfg')
model_ingredient = Ingredient('model_cfg')


@data_ingredient.config
def default_dataset_configuration():
    reader=BaBIDataset      # Dataset reader
    parser=SimpleParser     # Dataset parser
    size='10k'              # Dataset task size
    task='1'                # Dataset task index
    strip_sentences=False   # Dataset strip sentences


@model_ingredient.config
def default_model_configuration():
    model = SimpleLSTM      # Default model
    verbose = True          # Verbose
    epochs = 0


@model_ingredient.capture
def transform_parameter(model_cfg):
    if isinstance(model_cfg['model'], str):
        model_cfg['model'] = ModelLoader().load(model_cfg['model'])


ex = Experiment('PypagAI', ingredients=[data_ingredient, model_ingredient])


@ex.config
def default_config(dataset_cfg, model_cfg):
    pass


@ex.automain
def run(dataset_cfg, model_cfg):

    LOG.info("[START] Experiments")
    transform_parameter(model_cfg)
    train, validation = dataset_cfg['reader'](dataset_cfg, model_cfg).read()

    model = model_cfg['model'](model_cfg)
    metric = model.train(train, validation)

    return metric
