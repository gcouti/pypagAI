import logging

from sacred import Experiment

from pypagai.models.base import model_ingredient
from pypagai.preprocessing.read_data import data_ingredient
from pypagai.util.class_loader import ModelLoader

LOG = logging.getLogger(__name__)
ex = Experiment('PypagAI', ingredients=[data_ingredient, model_ingredient])


@ex.config
def default_config(dataset_cfg, model_cfg):
    if isinstance(model_cfg['model'], str):
        model_cfg['model'] = ModelLoader().load(model_cfg['model'])

    # model_cfg.update(model_cfg['model'].default_config())
    # dataset_cfg.update(dataset_cfg['dataset'])


@ex.capture
def read_data(dataset_cfg, model_cfg):
    return dataset_cfg['reader'](dataset_cfg, model_cfg).read()


@ex.automain
def run(model_cfg):

    LOG.info("[START] Experiments")
    train, validation = read_data()

    model = model_cfg['model'](model_cfg)
    model.train(train, validation)

    return model.valid(validation)
