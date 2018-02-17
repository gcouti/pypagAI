"""
Execute experiments with PypagAI framework

This main has the purpose of train and experiment models. It implement the same interface of the train_model.py
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
from pypagai.util.arguments import PypagaiParser
from pypagai.experiments.flow import ExperimentFlow

logging.basicConfig(level=settings.LOG_LEVEL)
LOG = logging.getLogger(__name__)

if __name__ == '__main__':
    LOG.info("[START] Experiments")

    LOG.debug("Reading params")
    args = PypagaiParser()

    LOG.debug("Init flow")
    flow = ExperimentFlow(args)
    flow.run()

    LOG.info("[END] Experiments")
