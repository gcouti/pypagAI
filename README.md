[![Coverage Status](https://coveralls.io/repos/github/gcouti/qa/badge.svg?branch=master)](https://coveralls.io/github/gcouti/qa?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

PypagAI 
=======

PypagAI is a easy and fast framework experiment Question Answering (QA) problems

Why PypagAI?
============

It was inspired on Facebook's dialog framework, [ParlAI](), but it is easier and faster to test models. 
It is very easy to integrate new [Keras]() and [TensorFlow]() models

The framework uses [Sacred]() as experiments backend

How to run
==========

Easy and if you know Sacred it's easier!

```shell
python -m experiment.qa_experiment -u
```

If you want list all available parameters just type

```shell
python -m experiment.qa_experiment print_config
```

You can also override the default experiment main and create your own flow. Just load data and model libraries. 
