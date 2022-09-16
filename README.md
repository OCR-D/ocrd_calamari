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
make gt4histocr-calamari1
ls gt4histocr-calamari1
```

Manual download: [model.tar.xz](https://qurator-data.de/calamari-models/GT4HistOCR/2019-12-11T11_10+0100/model.tar.xz)

## Example Usage
Before using `ocrd-calamari-recognize` get some example data and model, and
prepare the document for OCR:
```
# Download model and example data
make gt4histocr-calamari1
make actevedef_718448162

# Create binarized images and line segmentation using other OCR-D projects
cd actevedef_718448162
ocrd-olena-binarize -P impl sauvola-ms-split -I OCR-D-IMG -O OCR-D-IMG-BIN
ocrd-tesserocr-segment-region -I OCR-D-IMG-BIN -O OCR-D-SEG-REGION
ocrd-tesserocr-segment-line -I OCR-D-SEG-REGION -O OCR-D-SEG-LINE
```

Finally recognize the text using ocrd_calamari and the downloaded model:
```
ocrd-calamari-recognize -P checkpoint_dir "../gt4histocr-calamari1" -I OCR-D-SEG-LINE -O OCR-D-OCR-CALAMARI
```


You may want to have a look at the [ocrd-tool.json](ocrd_calamari/ocrd-tool.json) descriptions
for additional parameters and default values.

## Development & Testing
For information regarding development and testing, please see
[README-DEV.md](README-DEV.md).
