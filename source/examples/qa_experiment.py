from datetime import datetime
import logging
import logging.config

import pandas as pd
from sacred import Experiment

from pypagai.models.base import model_ingredient
from pypagai.models.model_embed_lstm import EmbedLSTM
from pypagai.models.model_encoder import EncoderModel
from pypagai.models.model_lstm import SimpleLSTM
from pypagai.models.model_n2nmemory import N2NMemory
from pypagai.models.model_rn import RN
from pypagai.models.model_rnn import RNNModel
from pypagai.models.model_scikit import SVMModel, RFModel
from pypagai.preprocessing.dataset_babi import BaBIDataset
from pypagai.preprocessing.read_data import data_ingredient
from pypagai.util.class_loader import DatasetLoader

ex = Experiment('PypagAI', ingredients=[data_ingredient, model_ingredient])
LOG = logging.getLogger('pypagai-logger')


@ex.config
def default_config(dataset_default_cfg, model_default_cfg):
    reader = None  # Reader Instance
    if isinstance(dataset_default_cfg['reader'], str):
        reader = DatasetLoader().load(dataset_default_cfg['reader'])
    else:
        reader = dataset_default_cfg['reader']

    dataset_cfg = {}
    dataset_cfg.update(dataset_default_cfg)
    dataset_cfg.update(reader.default_config())

    framework_cfg = {
        'TEMPORARY_MODEL_PATH': '/tmp/tmp_model_%i' % datetime.now().timestamp(),
        'TEMPORARY_RESULT_PATH': '/tmp/tmp_model_result_%i.json' % datetime.now().timestamp(),
        'LOG_LEVEL': logging.INFO,
        'LOG_FORMAT': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    }

    logging.Formatter(framework_cfg['LOG_FORMAT'])
    LOG.setLevel(framework_cfg['LOG_LEVEL'])


# @ex.capture
def read_data(dataset_cfg, model_cfg, reader):
    return reader(dataset_cfg, model_cfg).read()


# @ex.capture
def read_model(model, model_cfg):
    return model(model_cfg)


@ex.named_config
def baseline_config():
    models = [
        {
            'model': SimpleLSTM,
            'parameters': [{'hidden': h} for h in [32, 64, 128, 256]]
        },

        {
            'model': EmbedLSTM,
            'parameters': [{'hidden': h} for h in [32, 64, 128, 256]]
        },

        {
            'model': EncoderModel,
        },

        {
            'model': N2NMemory,
        },

        {
            'model': RN,
            'reader_cfg': {
                'strip_sentences': True
            }
        },

        {
            'model': RNNModel,
            'parameters': [{'hidden': h}for h in [32, 64, 128, 256]]
        },

        {
            'model': RFModel,
        },

        {
            'model': SVMModel,
        },

    ]

    dbs = [{'reader': BaBIDataset, 'task': t} for t in range(1, 21)]


@ex.automain
def run(models, dbs, reader, dataset_cfg, model_default_cfg):
    """
    This main use sacred experiments framework. To show all parameters type:

    main.py print_config
    """

    LOG.info("[START] Experiments")

    results = []
    for cfg in models:
        model = cfg['model']
        for db_cfg in dbs:

            for m_cfg in cfg['parameters'] if 'parameters' in cfg else [{}]:
                model_cfg = {}
                model_cfg.update(model_default_cfg)
                model_cfg.update(model.default_config())
                model_cfg.update(m_cfg)

                dataset_cfg.update(db_cfg)
                dataset_cfg.update(cfg['reader_cfg'] if 'reader_cfg' in cfg else {})
                train, validation = read_data(dataset_cfg, model_cfg, reader)

                estimator = read_model(model, model_cfg)
                estimator.train(train, validation)
                acc, f1 = estimator.valid(validation)

                r = {
                    'model': model,
                    'acc': acc,
                    'f1': f1,
                    'db': db_cfg['reader'].ALIAS,
                    'db_parameters': dataset_cfg,
                    'model_cfg': model_cfg
                }

                results.append(r)
                df = pd.DataFrame(results)
                df.to_csv('result.csv', sep=';', index=False)

    return results
