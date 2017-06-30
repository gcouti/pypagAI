#!/usr/bin/env bash
sudo apt-get install -y virtualenv
sudo apt-get install -y python3-pip python3-dev python-virtualenv
sudo apt-get install -y g++ python3-dev libjpeg-dev libfreetype6 libfreetype6-dev zlib1g-dev

pushd .
cd ..
virtualenv -p /usr/bin/python3 env
source env/bin/activate
popd

pip install -r requirements