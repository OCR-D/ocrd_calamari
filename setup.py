# -*- coding: utf-8 -*-
from pathlib import Path

from setuptools import setup, find_packages

setup(
    name='ocrd_calamari',
    version='1.0.1',
    description='Calamari bindings',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Konstantin Baierer, Mike Gerber',
    author_email='unixprog@gmail.com, mike.gerber@sbb.spk-berlin.de',
    url='https://github.com/OCR-D/ocrd_calamari',
    license='Apache License 2.0',
    packages=find_packages(exclude=('test', 'docs')),
    install_requires=Path('requirements.txt').read_text().split('\n'),
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-calamari-recognize=ocrd_calamari.cli:ocrd_calamari_recognize',
        ]
    },
)
