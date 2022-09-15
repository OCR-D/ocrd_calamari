import os
import shutil
import subprocess
import tempfile
import urllib.request
from lxml import etree
from glob import glob

import pytest
import logging
from ocrd.resolver import Resolver

from ocrd_calamari import CalamariRecognize
from .base import assets

METS_KANT = assets.url_of('kant_aufklaerung_1784-page-region-line-word_glyph/data/mets.xml')
WORKSPACE_DIR = tempfile.mkdtemp(prefix='test-ocrd-calamari-')
CHECKPOINT_DIR = os.getenv('MODEL')


def page_namespace(tree):
    """Return the PAGE content namespace used in the given ElementTree.

    This relies on the assumption that, in any given PAGE content file, the root element has the local name "PcGts". We
    do not check if the files uses any valid PAGE namespace.
    """
    root_name = etree.QName(tree.getroot().tag)
    if root_name.localname == "PcGts":
        return root_name.namespace
    else:
        raise ValueError("Not a PAGE tree")

def assertFileContains(fn, text):
    """Assert that the given file contains a given string."""
    with open(fn, "r", encoding="utf-8") as f:
        assert text in f.read()

def assertFileDoesNotContain(fn, text):
    """Assert that the given file does not contain given string."""
    with open(fn, "r", encoding="utf-8") as f:
        assert not text in f.read()


@pytest.fixture
def workspace():
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)

    resolver = Resolver()
    # due to core#809 this does not always work:
    #workspace = resolver.workspace_from_url(METS_KANT, dst_dir=WORKSPACE_DIR)
    # workaround:
    shutil.rmtree(WORKSPACE_DIR)
    shutil.copytree(os.path.dirname(METS_KANT), WORKSPACE_DIR)
    workspace = resolver.workspace_from_url(os.path.join(WORKSPACE_DIR, 'mets.xml'))

    # The binarization options I have are:
    #
    # a. ocrd_kraken which tries to install cltsm, whose installation is borken on my machine (protobuf)
    # b. ocrd_olena which 1. I cannot fully install via pip and 2. whose dependency olena doesn't compile on my
    #    machine
    # c. just fumble with the original files
    #
    # So I'm going for option c.
    for imgf in workspace.mets.find_files(fileGrp="OCR-D-IMG"):
        imgf = workspace.download_file(imgf)
        path = os.path.join(workspace.directory, imgf.local_filename)
        subprocess.call(['mogrify', '-threshold', '50%', path])

    # Remove GT Words and TextEquivs, to not accidently check GT text instead of the OCR text
    # XXX Review data again
    for of in workspace.mets.find_files(fileGrp="OCR-D-GT-SEG-WORD-GLYPH"):
        workspace.download_file(of)
        path = os.path.join(workspace.directory, of.local_filename)
        tree = etree.parse(path)
        nsmap_gt = { "pc": page_namespace(tree) }
        for to_remove in ["//pc:Word", "//pc:TextEquiv"]:
            for e in tree.xpath(to_remove, namespaces=nsmap_gt):
                e.getparent().remove(e)
        tree.write(path, xml_declaration=True, encoding="utf-8")
        assertFileDoesNotContain(path, "TextEquiv")

    yield workspace

    shutil.rmtree(WORKSPACE_DIR)


def test_recognize(workspace):
    CalamariRecognize(
        workspace,
        input_file_grp="OCR-D-GT-SEG-WORD-GLYPH",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={
            "checkpoint_dir": CHECKPOINT_DIR,
        }
    ).process()
    workspace.save_mets()

    page1 = os.path.join(workspace.directory, "OCR-D-OCR-CALAMARI/OCR-D-OCR-CALAMARI_0001.xml")
    assert os.path.exists(page1)
    assertFileContains(page1, "verſchuldeten")


def test_recognize_should_warn_if_given_rgb_image_and_single_channel_model(workspace, caplog):
    caplog.set_level(logging.WARNING)
    CalamariRecognize(
        workspace,
        input_file_grp="OCR-D-GT-SEG-WORD-GLYPH",
        output_file_grp="OCR-D-OCR-CALAMARI-BROKEN",
        parameter={'checkpoint_dir': CHECKPOINT_DIR}
    ).process()

    interesting_log_messages = [t[2] for t in caplog.record_tuples if "Using raw image" in t[2]]
    assert len(interesting_log_messages) > 10  # For every line!


def test_word_segmentation(workspace):
    CalamariRecognize(
        workspace,
        input_file_grp="OCR-D-GT-SEG-WORD-GLYPH",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={
            "checkpoint_dir": CHECKPOINT_DIR,
            "textequiv_level": "word",   # Note that we're going down to word level here
        }
    ).process()
    workspace.save_mets()

    page1 = os.path.join(workspace.directory, "OCR-D-OCR-CALAMARI/OCR-D-OCR-CALAMARI_0001.xml")
    assert os.path.exists(page1)
    tree = etree.parse(page1)
    nsmap = { "pc": page_namespace(tree) }

    # The result should contain a TextLine that contains the text "December"
    line = tree.xpath(".//pc:TextLine[pc:TextEquiv/pc:Unicode[contains(text(),'December')]]", namespaces=nsmap)[0]
    assert line is not None

    # The textline should a. contain multiple words and b. these should concatenate fine to produce the same line text
    words = line.xpath(".//pc:Word", namespaces=nsmap)
    assert len(words) >= 2
    words_text = " ".join(word.xpath("pc:TextEquiv/pc:Unicode", namespaces=nsmap)[0].text for word in words)
    line_text = line.xpath("pc:TextEquiv/pc:Unicode", namespaces=nsmap)[0].text
    assert words_text == line_text

    # For extra measure, check that we're not seeing any glyphs, as we asked for textequiv_level == "word"
    glyphs = tree.xpath("//pc:Glyph", namespaces=nsmap)
    assert len(glyphs) == 0


def test_glyphs(workspace):
    CalamariRecognize(
        workspace,
        input_file_grp="OCR-D-GT-SEG-WORD-GLYPH",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={
            "checkpoint_dir": CHECKPOINT_DIR,
            "textequiv_level": "glyph",   # Note that we're going down to glyph level here
        }
    ).process()
    workspace.save_mets()

    page1 = os.path.join(workspace.directory, "OCR-D-OCR-CALAMARI/OCR-D-OCR-CALAMARI_0001.xml")
    assert os.path.exists(page1)
    tree = etree.parse(page1)
    nsmap = { "pc": page_namespace(tree) }

    # The result should contain a lot of glyphs
    glyphs = tree.xpath("//pc:Glyph", namespaces=nsmap)
    assert len(glyphs) >= 100


# vim:tw=120:
