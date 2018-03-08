from setuptools import setup, find_packages
import sys

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', encoding="utf8") as f:
    readme = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='pypagai',
    version='0.0.0',
    description='',
    long_description=readme,
    url='http://pypaga.io/',
    license=license,
    packages=find_packages(exclude=(
        'examples',
    )),
    install_requires=reqs.strip().split('\n'),
)
