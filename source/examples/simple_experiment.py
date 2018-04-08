from datetime import datetime
import logging
import logging.config

from sacred import Experiment

from pypagai.models.base import model_ingredient
from pypagai.preprocessing.read_data import data_ingredient
from pypagai.util.class_loader import ModelLoader, DatasetLoader

ex = Experiment('PypagAI', ingredients=[data_ingredient, model_ingredient])
LOG = logging.getLogger('pypagai-logger')


@ex.config
def default_config(dataset_default_cfg, model_default_cfg):
    reader = None  # Reader Instance
    if isinstance(dataset_default_cfg['reader'], str):
        reader = DatasetLoader().load(dataset_default_cfg['reader'])
    else:
        reader = dataset_default_cfg['reader']

    model = None  # Model Instance
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

    framework_cfg = {
        'TEMPORARY_MODEL_PATH': '/tmp/tmp_model_%i' % datetime.now().timestamp(),
        'TEMPORARY_RESULT_PATH': '/tmp/tmp_model_result_%i.json' % datetime.now().timestamp(),
        'LOG_LEVEL': logging.INFO,
        'LOG_FORMAT': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    }

    logging.Formatter(framework_cfg['LOG_FORMAT'])
    LOG.setLevel(framework_cfg['LOG_LEVEL'])


@ex.capture
def read_data(dataset_cfg, model_cfg, reader):
    return reader(dataset_cfg, model_cfg).read()


@ex.capture
def read_model(model, model_cfg):
    return model(model_cfg)


@ex.automain
def run():
    """
    This main use sacred experiments framework. To show all parameters type:

    main.py print_config
    """

    LOG.info("[START] Experiments")

    train, validation = read_data()

    model = read_model()
    model.train(train, validation)

    return model.valid(validation)
