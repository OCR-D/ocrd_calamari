import os
import shutil
import subprocess
import urllib.request
from lxml import etree

import pytest
from ocrd.resolver import Resolver

from ocrd_calamari import CalamariRecognize
from .base import assets

METS_KANT = assets.url_of('kant_aufklaerung_1784-page-block-line-word_glyph/data/mets.xml')
WORKSPACE_DIR = '/tmp/test-ocrd-calamari'
CHECKPOINT = os.path.join(os.getcwd(), 'gt4histocr-calamari/*.ckpt.json')


@pytest.fixture
def workspace():
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)

    resolver = Resolver()
    workspace = resolver.workspace_from_url(METS_KANT, dst_dir=WORKSPACE_DIR)

    # XXX Work around data bug(?):
    #     PAGE-XML links to OCR-D-IMG/INPUT_0017.tif, but this is nothing core can download
    os.makedirs(os.path.join(WORKSPACE_DIR, 'OCR-D-IMG'))
    for f in ['INPUT_0017.tif', 'INPUT_0020.tif']:
        urllib.request.urlretrieve(
            "https://github.com/OCR-D/assets/raw/master/data/kant_aufklaerung_1784/data/OCR-D-IMG/" + f,
            os.path.join(WORKSPACE_DIR, 'OCR-D-IMG', f))

    # The binarization options I have are:
    #
    # a. ocrd_kraken which tries to install cltsm, whose installation is borken on my machine (protobuf)
    # b. ocrd_olena which 1. I cannot fully install via pip and 2. whose dependency olena doesn't compile on my
    #    machine
    # c. just fumble with the original files
    #
    # So I'm going for option c.
    for f in ['INPUT_0017.tif', 'INPUT_0020.tif']:
        ff = os.path.join(WORKSPACE_DIR, 'OCR-D-IMG', f)
        subprocess.call(['convert', ff, '-threshold', '50%', ff])

    return workspace


def test_recognize(workspace):
    # XXX Should remove GT text to really test this

    CalamariRecognize(
        workspace,
        input_file_grp="OCR-D-GT-SEG-LINE",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={
            "checkpoint": CHECKPOINT,
        }
    ).process()
    workspace.save_mets()

    page1 = os.path.join(workspace.directory, "OCR-D-OCR-CALAMARI/OCR-D-OCR-CALAMARI_0001.xml")
    assert os.path.exists(page1)
    with open(page1, "r", encoding="utf-8") as f:
        assert "verſchuldeten" in f.read()


def test_word_segmentation(workspace):
    CalamariRecognize(
        workspace,
        input_file_grp="OCR-D-GT-SEG-LINE",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={
            "checkpoint": CHECKPOINT,
        }
    ).process()
    workspace.save_mets()

    page1 = os.path.join(workspace.directory, "OCR-D-OCR-CALAMARI/OCR-D-OCR-CALAMARI_0001.xml")
    assert os.path.exists(page1)
    tree = etree.parse(page1)

    NSMAP = { "pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15" }

    # The result should contain a TextLine that contains the text "December"
    line = tree.xpath(".//pc:TextLine[pc:TextEquiv/pc:Unicode[contains(text(),'December')]]", namespaces=NSMAP)[0]
    assert line

    # The textline should a. contain multiple words and b. these should concatenate fine to produce the same line text
    words = line.xpath(".//pc:Word", namespaces=NSMAP)
    assert len(words) >= 2
    words_text = " ".join(word.xpath("pc:TextEquiv/pc:Unicode", namespaces=NSMAP)[0].text for word in words)
    line_text = line.xpath("pc:TextEquiv/pc:Unicode", namespaces=NSMAP)[0].text
    assert words_text == line_text


# vim:tw=120:
