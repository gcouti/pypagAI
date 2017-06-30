#!/usr/bin/env bash
sudo apt-get install -y virtualenv
sudo apt-get install -y gcc python3-dev

virtualenv -p /usr/bin/python3.5 env

source env/bin/activate
pip install -r requirements