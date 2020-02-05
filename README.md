# ocrd_calamari

> Recognize text using [Calamari OCR](https://github.com/Calamari-OCR/calamari).

[![image](https://circleci.com/gh/OCR-D/ocrd_calamari.svg?style=svg)](https://circleci.com/gh/OCR-D/ocrd_calamari)
[![image](https://img.shields.io/pypi/v/ocrd_calamari.svg)](https://pypi.org/project/ocrd_calamari/)
[![image](https://codecov.io/gh/OCR-D/ocrd_calamari/branch/master/graph/badge.svg)](https://codecov.io/gh/OCR-D/ocrd_calamari)

## Introduction

This offers a OCR-D compliant workspace processor for some of the functionality of Calamari OCR.

This processor only operates on the text line level and so needs a line segmentation (and by extension a binarized 
image) as its input.

In addition to the line text it may also output word and glyph segmentation
including per-glyph confidence values and per-glyph alternative predictions as
provided by the Calamari OCR engine, using a `textequiv_level` of `word` or
`glyph`. Note that while Calamari does not provide word segmentation, this
processor produces word segmentation inferred from text
segmentation and the glyph positions. The provided glyph and word segmentation
can be used for text extraction and highlighting, but is probably not useful for
further image-based processing.

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

Download models trained on GT4HistOCR data:

```
make gt4histocr-calamari
ls gt4histocr-calamari
```

## Example Usage

~~~
ocrd-calamari-recognize -p test-parameters.json -m mets.xml -I OCR-D-SEG-LINE -O OCR-D-OCR-CALAMARI
~~~

With `test-parameters.json`:
~~~
{
    "checkpoint": "/path/to/some/trained/models/*.ckpt.json",
    "textequiv_level": "line"
}
~~~

You may want to have a look at the [ocrd-tool.json](ocrd-tool.json) descriptions
for additional parameters and default values.

## Development & Testing
For information regarding development and testing, please see
[README-DEV.md](README-DEV.md).
