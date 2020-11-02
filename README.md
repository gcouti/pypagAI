[![Coverage Status](https://coveralls.io/repos/github/gcouti/qa/badge.svg?branch=master)](https://coveralls.io/github/gcouti/qa?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

PypagAI 
=======

PypagAI is a framework that aims to quickly develop new QA models, test new datasets and reproduce experiments. The main objective of the framework is to speed up the development of new QA models. Thus, it is necessary to normalize experiments, prepare data, and create a repository of QA models. By building and publishing this framework, we would like to lower the entry barrier for more people to try out.


Why PypagAI?
============

PypagAI was inspired on Facebook's dialog framework, [ParlAI](), but it is easier and faster to test models. 
It uses [Sacred]() as experiments backend and is very easy to integrate new [Keras]() and [TensorFlow]() models


How to run
==========

To run with default configurations 

```shell
python -m experiment.qa_experiment -u
```

If you want list all available parameters just type

```shell
python -m experiment.qa_experiment print_config
```

Changing the models

```shell
python -m experiment.qa_experiment with model_default_cfg.model=pypagai.models.model_rnn.RNNModel -u
 
```

Changing dataset

```shell
python -m experiment.qa_experiment with  dataset_cfg.task=3 -u
```

You can also override the default experiment main and create your own flow. Just load data and model libraries. 
