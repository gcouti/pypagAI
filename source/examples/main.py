import logging

from sacred import Experiment, Ingredient

from pypagai.models.model_lstm import SimpleLSTM
from pypagai.preprocessing.dataset_babi import BaBIDataset
from pypagai.preprocessing.parser import SimpleParser
from pypagai.util.class_loader import ModelLoader

LOG = logging.getLogger(__name__)

data_ingredient = Ingredient('dataset_cfg')
model_ingredient = Ingredient('model_cfg')


@data_ingredient.config
def default_dataset_configuration():
    """
    Dataset configuration
    """
    reader=BaBIDataset      # Path to dataset reader ex.: pypagai.preprocessing.dataset_babi.BaBIDataset
    parser=SimpleParser     # Path to dataset parser ex.: pypagai.preprocessing.parser.SimpleParser
    strip_sentences=False   # Property to split sentences

    task='1'                # Task
    size='10k'              # Dataset size


@model_ingredient.config
def default_model_configuration():
    """
    Model configuration
    """
    model = SimpleLSTM      # Path to the ML model
    verbose = True          # True to print info about train
    epochs= 1


ex = Experiment('PypagAI', ingredients=[data_ingredient, model_ingredient])


@ex.capture
def transform_parameter(model_cfg):
    if isinstance(model_cfg['model'], str):
        model_cfg['model'] = ModelLoader().load(model_cfg['model'])


@ex.capture
def read_data(dataset_cfg, model_cfg):
    return dataset_cfg['reader'](dataset_cfg, model_cfg).read()


@ex.automain
def run(model_cfg):

    LOG.info("[START] Experiments")
    transform_parameter()
    train, validation = read_data()

    model = model_cfg['model'](model_cfg)
    model.train(train, validation)

    return model.valid(validation)
