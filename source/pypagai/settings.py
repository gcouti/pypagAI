import logging

from datetime import datetime

LOG_FILE = 'log'
LOG_LEVEL = logging.INFO

TEMPORARY_MODEL_PATH = '/tmp/tmp_model_%i' % datetime.now().timestamp()
TEMPORARY_RESULT_PATH = '/tmp/tmp_model_result_%i.json' % datetime.now().timestamp()

try:
    from custom_settings import *
except ImportError as exp:
    pass
