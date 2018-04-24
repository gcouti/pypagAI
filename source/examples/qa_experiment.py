import os
import logging
import logging.config

from datetime import datetime

from pypagai.experiments import core
from pypagai.experiments.evaluation import evaluate_results, make_result_frame, classification_report
from sacred import Experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC

from pypagai.experiments.observers import PypagAIFileStorageObserver
from pypagai.models.base import model_ingredient, SciKitModel
from pypagai.models.model_lstm import SimpleLSTM
from pypagai.preprocessing.read_data import data_ingredient
from pypagai.util.class_loader import DatasetLoader, ModelLoader

path_template = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report_template.ipynb')

ex = Experiment('PypagAI', ingredients=[data_ingredient, model_ingredient])
ex.observers.append(PypagAIFileStorageObserver.create('results/', template=path_template))

LOG = logging.getLogger('pypagai-logger')


@ex.config
def default_framework_config():
    framework_cfg = {
        'TEMPORARY_MODEL_PATH': '/tmp/tmp_model_%i' % datetime.now().timestamp(),
        'TEMPORARY_RESULT_PATH': '/tmp/tmp_model_result_%i.json' % datetime.now().timestamp(),
        'LOG_LEVEL': logging.INFO,
        'LOG_FORMAT': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        'test_size': .2,
    }

    logging.Formatter(framework_cfg['LOG_FORMAT'])
    LOG.setLevel(framework_cfg['LOG_LEVEL'])
    ex.logger = LOG


@ex.config
def default_dataset_config(dataset_default_cfg):
    reader = None  # Reader Instance
    if isinstance(dataset_default_cfg['reader'], str):
        reader = DatasetLoader().load(dataset_default_cfg['reader'])
    else:
        reader = dataset_default_cfg['reader']

    dataset_cfg = {}
    dataset_cfg.update(dataset_default_cfg)
    dataset_cfg.update(reader.default_config())


@ex.config
def default_model_config(model_default_cfg):
    model = None  # Model Instance
    if isinstance(model_default_cfg['model'], str):
        model = ModelLoader().load(model_default_cfg['model'])
    else:
        model = model_default_cfg['model']

    model_cfg = {}
    model_cfg.update(model_default_cfg)
    model_cfg.update(model.default_config())


@ex.capture
def read_data(dataset_cfg, model_cfg, reader):
    return reader(dataset_cfg, model_cfg).read()


@ex.capture
def read_model(model, model_cfg):
    return model(model_cfg)


@ex.named_config
def svm_config():
    model = SciKitModel

    model_cfg = {}
    model_cfg.update(model.default_config())
    model_cfg['model'] = GridSearchCV(SVC(), param_grid={
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    })


@ex.named_config
def rf_config():
    model = SciKitModel

    model_cfg = {}
    model_cfg.update(model.default_config())
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
def lstm_config():
    model = SimpleLSTM

    model_cfg = {}
    model_cfg.update(model.default_config())
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


@ex.automain
def run(_run):
    """
    This main use sacred experiments framework. To show all parameters type:

    main.py print_config
    """
    LOG.info("[START] Experiments")

    train, test = read_data()
    estimator = read_model()

    estimator.train(train)

    # Test estimators
    test_pred = estimator.predict(test)
    test_results = make_result_frame(test_pred, test.answer)

    # Print results
    _run.info = {
        'raw_results': {
            'test': test_results
        },

        'report': {
            'test': classification_report(test_pred, test.answer)
        },

        'metrics': {
            'test': evaluate_results(test_results, metrics=['f1_macro', 'f1_micro', 'accuracy'])
        },
    }

    # return estimator
