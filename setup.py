import sys
from setuptools import setup, find_packages

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', encoding="utf8") as f:
    readme = f.read()

setup(
    name='pypagai',
    version='0.0.2',
    description='',
    long_description=readme,
    url='http://pypaga.io/',
    license=license,
    packages=find_packages(exclude=(
        'examples',
    )),
    install_requires=[
        'numpy==1.13.3',
        'Pillow==4.3.0',
        'Keras==2.1.5',
        'tensorflow-gpu==1.7.0',
        'tensorflow-tensorboard==0.4.0rc3',
        'h5py==2.7.1',
        'spacy==2.0.3',
        'nltk==3.2.5',
        'scikit-learn==0.19.1',
        'sacred==0.7.2',
        'pandas==0.22.0',
    ]
)
