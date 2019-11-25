# aux pkg imports
import logging

import numpy as np
import pandas as pd

# sci-kit imports
from sklearn.metrics import get_scorer, precision_recall_fscore_support, accuracy_score, f1_score

LOG = logging.getLogger('pypagai-logger')


def metrics(pred, true):
    """
    Print metrics based on predicted answers and true answers

    :param pred: Predicted
    :param true: True answers
    """
    acc = accuracy_score(np.argsort(pred)[:, ::-1][:, 0], true)
    f1 = f1_score(np.argsort(pred)[:, ::-1][:, 0], true, average="macro")
    LOG.info("Accuracy: %f F1: %f", acc, f1)

    return acc, f1


def make_examples(y_true, y_pred, data, vocab, size=3):
    """ Show some correct examples and incorrect ones

    :param y_true:
    :param y_pred:
    :param data:
    :param vocab: Vocabulary to show in examples
    :param size: Size of examples5

    :return: List of correct and incorrect examples
    """
    report = {'true': [], 'false': []}
    indexes = np.where(y_true == y_pred)[0][:size]
    extract_sample(data, indexes, report, vocab, y_pred, 'true')
    indexes = np.where(y_true != y_pred)[0][:size]
    extract_sample(data, indexes, report, vocab, y_pred, 'false')

    report = pd.DataFrame(report)
    return report


def extract_sample(data, indexes, report, vocab, y_pred, dimension='true'):

    for i in indexes:
        text = " ".join(data[i][0])
        sample = {
            'text': text,
            'question': " ".join(data[i][1]),
            'correct': data[i][2],
            'predicted': vocab[y_pred[i]-1]
        }
        report[dimension].append(sample)

    for i in range(len(report[dimension]), 3):
        report[dimension].append({})

    return report


def make_result_frame(y_true, y_pred, index=None, repeat=0, fold=0):
    """Makes a pandas.DataFrame containing the results info.
    it will have the following columns:
    - repeat: experiment repetition id
    - fold: cross-validation fold id
    - id: instance id
    - y_true: instance ground truth
    - y_pred: instance predicted value

    :param fold:
    :param y_true: ground truth
    :param y_pred: predictions
    :param index: instance ids
    :param repeat: current repetition id
    :param repeat: current fold id
    """
    if index is None:
        index = np.arange(y_true.shape[0])

    ones = np.ones(y_true.shape[0], dtype=int)

    result = {
        'id': index,
        'fold': ones + fold,
        'repeat': ones + repeat,
        'y_true': y_true,
        'y_pred': y_pred
    }

    return pd.DataFrame(result)


def evaluate_results(df_results, metrics=['accuracy']):
    """
    Evaluates a result frame base on the given metrics.
    All scikit-learning string metrics are available.

    :param metrics:
    :param df_results:

    :returns: metrics relatives to both repeat and fold values
    """
    def exec_score_func(metric, y_true, y_pred):
        scorer = get_scorer(metric)
        return scorer._score_func(y_true, y_pred, **scorer._kwargs)

    return df_results.groupby(['fold']). \
        apply(lambda r: pd.Series([exec_score_func(m, r.y_true, r.y_pred) for m in metrics], index=metrics))


def classification_report(y_true, y_pred, target_names=None):
    d = {k: v for k, v in zip(['precision', 'recall', 'f1', 'suport'], precision_recall_fscore_support(y_true, y_pred))}
    return pd.DataFrame(d, index=target_names)
