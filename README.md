# ocrd_calamari

> Recognize text using [Calamari OCR](https://github.com/Calamari-OCR/calamari).

[![image](https://circleci.com/gh/OCR-D/ocrd_calamari.svg?style=svg)](https://circleci.com/gh/OCR-D/ocrd_calamari)
[![image](https://img.shields.io/pypi/v/ocrd_calamari.svg)](https://pypi.org/project/ocrd_calamari/)
[![image](https://codecov.io/gh/OCR-D/ocrd_calamari/branch/master/graph/badge.svg)](https://codecov.io/gh/OCR-D/ocrd_calamari)

## Introduction

**ocrd_calamari** offers a [OCR-D](https://ocr-d.de) compliant workspace processor for the functionality of Calamari OCR. It uses OCR-D workspaces (METS) with [PAGE XML](https://github.com/PRImA-Research-Lab/PAGE-XML) documents as input and output.

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

![Example output as viewed in PAGE Viewer](https://github.com/OCR-D/ocrd_calamari/raw/screenshots/output-in-page-viewer.jpg)

## Installation

### From PyPI

```sh
pip install ocrd_calamari
```

### From the git repository

```sh
pip install .
```

## Install models

Download models trained on GT4HistOCR data:

```
make qurator-gt4histocr-1.0
ls .local/share/ocrd-resources/ocrd-calamari-recognize/*
```

Manual download: [model.tar.xz](https://qurator-data.de/calamari-models/GT4HistOCR/2019-12-11T11_10+0100/model.tar.xz)

## Example Usage
Before using `ocrd-calamari-recognize` get some example data and model:

```
# Download model and example data
make qurator-gt4histocr-1.0
make example
```

The example already contains a binarized and line-segmented page, so we are ready to go. Recognize
the text using ocrd_calamari and the downloaded model:

```
cd actevedef_718448162.first-page+binarization+segmentation
ocrd-calamari-recognize \
  -P checkpoint_dir qurator-gt4histocr-1.0 \
  -I OCR-D-SEG-LINE-SBB -O OCR-D-OCR-CALAMARI
```

You may want to have a look at the [ocrd-tool.json](ocrd_calamari/ocrd-tool.json) descriptions
for additional parameters and default values.

## Development & Testing
For information regarding development and testing, please see
[README-DEV.md](README-DEV.md).
