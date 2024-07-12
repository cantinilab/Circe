# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['circe']

package_data = \
{'': ['*']}

setup_kwargs = {
    'name': 'circe',
    'version': '0.2.0',
    'description': '',
    'long_description': 'None',
    'author': 'Remi-Trimbour',
    'author_email': 'remi.trimbour@pasteur.fr',
    'maintainer': 'Remi-Trimbour',
    'maintainer_email': 'remi.trimbour@gmail.com',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.8,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
