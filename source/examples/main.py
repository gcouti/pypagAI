import logging

from sacred import Experiment

from pypagai.models.base import model_ingredient
from pypagai.preprocessing.read_data import data_ingredient
from pypagai.util.class_loader import ModelLoader, DatasetLoader

LOG = logging.getLogger(__name__)
ex = Experiment('PypagAI', ingredients=[data_ingredient, model_ingredient])


@ex.config
def default_config(dataset_default_cfg, model_default_cfg):

    reader = None   # Reader Instance
    if isinstance(dataset_default_cfg['reader'], str):
        reader = DatasetLoader().load(dataset_default_cfg['reader'])
    else:
        reader = dataset_default_cfg['reader']

    model = None    # Model Instance
    if isinstance(model_default_cfg['model'], str):
        model = ModelLoader().load(model_default_cfg['model'])
    else:
        model = model_default_cfg['model']

    dataset_cfg = {}
    dataset_cfg.update(dataset_default_cfg)
    dataset_cfg.update(reader.default_config())

    model_cfg = {}
    model_cfg.update(model_default_cfg)
    model_cfg.update(model.default_config())


@ex.capture
def read_data(dataset_cfg, model_cfg, reader):
    return reader(dataset_cfg, model_cfg).read()


@ex.capture
def read_model(model, model_cfg):
    return model(model_cfg)


@ex.automain
def run():

    LOG.info("[START] Experiments")
    train, validation = read_data()

    model = read_model()
    model.train(train, validation)

    return model.valid(validation)
