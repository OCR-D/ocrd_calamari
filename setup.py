# -*- coding: utf-8 -*-
import json
from pathlib import Path

from setuptools import find_packages, setup

with open("./ocrd-tool.json", "r") as f:
    version = json.load(f)["version"]

setup(
    name="ocrd_calamari",
    version=version,
    description="Recognize text using Calamari OCR and the OCR-D framework",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Konstantin Baierer, Mike Gerber",
    author_email="unixprog@gmail.com, mike.gerber@sbb.spk-berlin.de",
    url="https://github.com/OCR-D/ocrd_calamari",
    license="Apache License 2.0",
    packages=find_packages(exclude=("test", "docs")),
    install_requires=Path("requirements.txt").read_text().split("\n"),
    package_data={
        "": ["*.json", "*.yml", "*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "ocrd-calamari-recognize=ocrd_calamari.cli:ocrd_calamari_recognize",
            "fix-calamari1-model=ocrd_calamari.fix_calamari1_model:fix_calamari1_model",
        ]
    },
)
