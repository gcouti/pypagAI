import logging
import logging.config
from datetime import datetime

from sacred import Experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from tensorflow.contrib.learn import SVM

from pypagai.models.base import model_ingredient
from pypagai.models.model_lstm import SimpleLSTM
from pypagai.models.model_my import ReluN2NMemory
from pypagai.preprocessing.read_data import data_ingredient
from pypagai.util.class_loader import DatasetLoader, ModelLoader

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


# @ex.named_config
# def baby_babi_config():
#     dbs = [{'reader': BaBIDataset, 'task': t, 'size': ''} for t in range(1, 2)]
#
#
# @ex.named_config
# def babi_config():
#     dbs = [{'reader': BaBIDataset, 'task': t} for t in range(1, 20)]


@ex.named_config
def svm_config(model_default_cfg):
    model_default_cfg['model'] = SVM
    model_default_cfg['grid_search'] = {
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    }


@ex.named_config
def rf_config(model_cfg):
    model_cfg['model'] = GridSearchCV(RandomForestClassifier(), param_grid={
        "max_depth": [3, 10, 100, None],
        "max_features": [1, 3, 10],
        "min_samples_split": [2, 3, 10],
        "min_samples_leaf": [1, 3, 10],
        "bootstrap": [True, False],
        "n_estimators": [50, 100, 200, 300],
        "criterion": ["gini", "entropy"]
    })


@ex.named_config
def simple_lstm(model_cfg):
    model_cfg['model'] = SimpleLSTM
        #     # 'parameters': [{'batch_size': 1024, 'hidden': h} for h in [32, 64, 128, 256]]
        #     'model': EmbedLSTM,
        #     # 'parameters': [{'hidden': h} for h in [32, 64, 128, 256]]
        #     'model': EncoderModel,
        #     'model': N2NMemory,
        #     'model': RN,
        #     'reader_cfg': {
        #         'strip_sentences': True
        #     }
        #     'model': RNNModel,
        #     'parameters': [{'hidden': h}for h in [32, 64, 128, 256]]



@ex.named_config
def my_models_config():

    models = [
        {
             'model': ReluN2NMemory,
             'parameters': [{}]
        },
    ]


@ex.automain
def run(dataset_cfg, model_cfg):
    """
    This main use sacred experiments framework. To show all parameters type:

    main.py print_config
    """
    LOG.info("[START] Experiments")

    train, validation = read_data()
    estimator = read_model()
    estimator.train(train, validation)
    acc, f1 = estimator.valid(validation)

    return estimator, acc, f1
