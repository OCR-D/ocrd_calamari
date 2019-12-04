# ocrd_calamari

> Recognize text using [Calamari OCR](https://github.com/Calamari-OCR/calamari).

[![image](https://circleci.com/gh/OCR-D/ocrd_calamari.svg?style=svg)](https://circleci.com/gh/OCR-D/ocrd_calamari)
[![image](https://img.shields.io/pypi/v/ocrd_calamari.svg)](https://pypi.org/project/ocrd_calamari/)
[![image](https://codecov.io/gh/OCR-D/ocrd_calamari/branch/master/graph/badge.svg)](https://codecov.io/gh/OCR-D/ocrd_calamari)

## Introduction

This offers a OCR-D compliant workspace processor for some of the functionality of Calamari OCR.

This processor only operates on the text line level and so needs a line segmentation (and by extension a binarized 
image) as its input.

## Installation

### From PyPI

```
pip install ocrd_calamari
```

### From Repo

```sh
pip install .
```

## Install models

Download standard models:

```
wget https://github.com/Calamari-OCR/calamari_models/archive/master.zip
unzip master.zip
```

Download models trained on GT4HistOCR data:

```
wget https://file.spk-berlin.de:8443/calamari-models/GT4HistOCR/model.tar.xz
mkdir gt4hist-calamari
cd gt4hist-calamari
tar xf ../model.tar.xz
```

## Example Usage

~~~
ocrd-calamari-recognize -p test-parameters.json -m mets.xml -I OCR-D-SEG-LINE -O OCR-D-OCR-CALAMARI
~~~

With `test-parameters.json`:
~~~
{
    "checkpoint": "/path/to/some/trained/models/*.ckpt.json"
}
~~~

TODO
----

* Support Calamari's "extended prediction data" output
* Currently, the processor only supports a prediction using confidence voting of multiple models. While this is
  superior, it makes sense to support single model prediction, too.
