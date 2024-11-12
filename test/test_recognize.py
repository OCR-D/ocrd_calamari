import logging
import os
import shutil

from lxml import etree

from ocrd import run_processor
from ocrd_utils import MIMETYPE_PAGE as PAGE
from ocrd_models.constants import NAMESPACES as NS
from ocrd_modelfactory import page_from_file
from ocrd_calamari import CalamariRecognize

CHECKPOINT_DIR = os.getenv("MODEL", "qurator-gt4histocr-1.0")
DEBUG = os.getenv("DEBUG", False)


def assertFileContains(fn, text, msg=""):
    """Assert that the given file contains a given string."""
    with open(fn, "r", encoding="utf-8") as f:
        assert text in f.read(), msg


def assertFileDoesNotContain(fn, text, msg=""):
    """Assert that the given file does not contain given string."""
    with open(fn, "r", encoding="utf-8") as f:
        assert text not in f.read(), msg


def test_recognize(workspace_aufklaerung_binarized, caplog):
    caplog.set_level(logging.WARNING)
    ws = workspace_aufklaerung_binarized
    page1 = ws.mets.physical_pages[0]
    file1 = list(ws.find_files(file_grp="OCR-D-GT-WORD", page_id=page1, mimetype=PAGE))[0]
    text1 = page_from_file(file1).etree.xpath(
        '//page:TextLine/page:TextEquiv[1]/page:Unicode/text()', namespaces=NS)
    assert len(text1) > 10
    assert "verſchuldeten" in "\n".join(text1)
    run_processor(
        CalamariRecognize,
        input_file_grp="OCR-D-GT-WORD",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={
            "checkpoint_dir": CHECKPOINT_DIR,
        },
        workspace=ws,
    )
    overwrite_text_log_messages = [t[2] for t in caplog.record_tuples
                                   if "already contained text results" in t[2]]
    assert len(overwrite_text_log_messages) > 10  # For every line!
    overwrite_word_log_messages = [t[2] for t in caplog.record_tuples
                                   if "already contained word segmentation" in t[2]]
    assert len(overwrite_word_log_messages) > 10  # For every line!
    ws.save_mets()
    file1 = next(ws.find_files(file_grp="OCR-D-OCR-CALAMARI", page_id=page1, mimetype=PAGE), False)
    assert file1, "result for first page not referenced in METS"
    assert os.path.exists(file1.local_filename), "result for first page not found in filesystem"
    text1_out = page_from_file(file1).etree.xpath(
        '//page:TextLine/page:TextEquiv[1]/page:Unicode/text()', namespaces=NS)
    assert len(text1_out) == len(text1), "not all lines have been recognized"
    assert "verſchuldeten" in "\n".join(text1_out), "result for first page is inaccurate"
    assert "\n".join(text1_out) != "\n".join(text1), "result is suspiciously identical to GT"


def test_recognize_rgb(workspace_aufklaerung, caplog):
    caplog.set_level(logging.WARNING)
    run_processor(
        CalamariRecognize,
        input_file_grp="OCR-D-GT-PAGE",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={"checkpoint_dir": CHECKPOINT_DIR},
        workspace=workspace_aufklaerung,
    )
    interesting_log_messages = [t[2] for t in caplog.record_tuples
                                if "Using raw image" in t[2]]
    assert len(interesting_log_messages) > 10  # For every line!


def test_words(workspace_aufklaerung_binarized):
    run_processor(
        CalamariRecognize,
        input_file_grp="OCR-D-GT-WORD",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={
            "checkpoint_dir": CHECKPOINT_DIR,
            "textequiv_level": "word",
        },
        workspace=workspace_aufklaerung_binarized
    )
    ws = workspace_aufklaerung_binarized
    ws.save_mets()
    page1 = ws.mets.physical_pages[0]
    file1 = next(ws.find_files(file_grp="OCR-D-OCR-CALAMARI", page_id=page1, mimetype=PAGE), False)
    assert file1, "result for first page not referenced in METS"
    assert os.path.exists(file1.local_filename), "result for first page not found in filesystem"
    tree1 = page_from_file(file1).etree
    # The result should contain a TextLine that contains the text "Berliniſche"
    line = tree1.xpath(
        "//page:TextLine[page:TextEquiv/page:Unicode[contains(text(),'Berliniſche')]]",
        namespaces=NS,
    )
    assert len(line) == 1, "result is inaccurate"
    line = line[0]
    # The textline should
    # a. contain multiple words and
    # b. these should concatenate fine to produce the same line text
    words = line.xpath(".//page:Word", namespaces=NS)
    assert len(words) >= 2, "result does not contain words"
    words_text = " ".join(
        word.xpath("page:TextEquiv[1]/page:Unicode/text()", namespaces=NS)[0]
        for word in words
    )
    line_text = line.xpath("page:TextEquiv[1]/page:Unicode/text()", namespaces=NS)[0]
    assert words_text == line_text, "word-level text result does not concatenate to line-level text result"
    # For extra measure, check that we're not seeing any glyphs, as we asked for
    # textequiv_level == "word"
    glyphs = tree1.xpath("//page:Glyph", namespaces=NS)
    assert len(glyphs) == 0, "result must not contain glyph-level segments"


def test_glyphs(workspace_aufklaerung_binarized):
    run_processor(
        CalamariRecognize,
        input_file_grp="OCR-D-GT-WORD",
        output_file_grp="OCR-D-OCR-CALAMARI",
        parameter={
            "checkpoint_dir": CHECKPOINT_DIR,
            "textequiv_level": "glyph",
        },
        workspace=workspace_aufklaerung_binarized,
    )
    ws = workspace_aufklaerung_binarized
    ws.save_mets()
    page1 = ws.mets.physical_pages[0]
    file1 = next(ws.find_files(file_grp="OCR-D-OCR-CALAMARI", page_id=page1, mimetype=PAGE), False)
    assert file1, "result for first page not referenced in METS"
    assert os.path.exists(file1.local_filename), "result for first page not found in filesystem"
    tree1 = page_from_file(file1).etree
    # The result should contain a lot of glyphs
    glyphs = tree1.xpath("//page:Glyph", namespaces=NS)
    assert len(glyphs) >= 100, "result must contain lots of glyphs"
