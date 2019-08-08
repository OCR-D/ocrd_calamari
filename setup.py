# -*- coding: utf-8 -*-
"""
Installs one executable:

    - ocrd_calamari_ocr
"""
import codecs

from setuptools import setup, find_packages

setup(
    name='ocrd_calamari',
    version='0.0.1',
    description='Calamari bindings',
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    author='Konstantin Baierer, Mike Gerber',
    author_email='unixprog@gmail.com, mike.gerber@sbb.spk-berlin.de',
    url='https://github.com/kba/ocrd_calamari',
    license='Apache License 2.0',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=open('requirements.txt').read().split('\n'),
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-calamari-ocr=ocrd_calamari.cli:ocrd_calamari_ocr',
        ]
    },
)
