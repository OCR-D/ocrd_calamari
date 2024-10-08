from __future__ import absolute_import

import itertools
import os
from glob import glob

import numpy as np
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    CoordsType,
    GlyphType,
    TextEquivType,
    WordType,
    to_xml,
)
from ocrd_utils import (
    MIMETYPE_PAGE,
    assert_file_grp_cardinality,
    coordinates_for_segment,
    getLogger,
    make_file_id,
    points_from_polygon,
    polygon_from_x0y0x1y1,
    tf_disable_interactive_logs,
)

# Disable tensorflow/keras logging via print before importing calamari
# (and disable ruff's import checks and sorting here)
# ruff: noqa: E402
# ruff: isort: off
tf_disable_interactive_logs()

from tensorflow import __version__ as tensorflow_version
from calamari_ocr import __version__ as calamari_version
from calamari_ocr.ocr import MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams
from tensorflow import config as tensorflow_config

# ruff: isort: on

from ocrd_calamari.config import OCRD_TOOL

TOOL = "ocrd-calamari-recognize"

BATCH_SIZE = 64

def batched_length_limited(iterable, n, limit=32000):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        # implement poor man's batch bucketing to avoid OOM:
        maxlen = max(image.shape[1] for image in batch)
        if maxlen * n > limit and n > 1:
            yield from batched_length_limited(batch, n//2)
        else:
            yield batch

class CalamariRecognize(Processor):
    def __init__(self, *args, **kwargs):
        kwargs["ocrd_tool"] = OCRD_TOOL["tools"][TOOL]
        kwargs["version"] = "%s (calamari %s, tensorflow %s)" % (
            OCRD_TOOL["version"],
            calamari_version,
            tensorflow_version,
        )
        super(CalamariRecognize, self).__init__(*args, **kwargs)
        if hasattr(self, "output_file_grp"):
            # processing context
            self.setup()

    def setup(self):
        """
        Set up the model prior to processing.
        """
        log = getLogger("processor.CalamariRecognize")
        devices = tensorflow_config.list_physical_devices("GPU")
        for device in devices:
            log.info("using GPU device %s", device)
            tensorflow_config.experimental.set_memory_growth(device, True)
        resolved = self.resolve_resource(self.parameter["checkpoint_dir"])
        checkpoints = glob("%s/*.ckpt.json" % resolved)
        self.predictor = MultiPredictor(checkpoints=checkpoints, batch_size=BATCH_SIZE)
        log.info("loaded model %s", resolved)

        self.network_input_channels = self.predictor.predictors[
            0
        ].network.input_channels

        # not used:
        # self.network_input_channels = \
        #        self.predictor.predictors[0].network_params.channels
        # not used:
        # binarization = \
        #        self.predictor.predictors[0].model_params\
        #        .data_preprocessor.binarization
        # self.features = ('' if self.network_input_channels != 1 else
        #                  'binarized' if binarization != 'GRAY' else
        #                  'grayscale_normalized')
        self.features = ""

        voter_params = VoterParams()
        voter_params.type = VoterParams.Type.Value(self.parameter["voter"].upper())
        self.voter = voter_from_proto(voter_params)

    def process(self):
        """
        Perform text recognition with Calamari on the workspace.

        If ``texequiv_level`` is ``word`` or ``glyph``, then additionally create word /
        glyph level segments by splitting at white space characters / glyph boundaries.
        In the case of ``glyph``, add all alternative character hypotheses down to
        ``glyph_conf_cutoff`` confidence threshold.
        """
        log = getLogger("processor.CalamariRecognize")

        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            log.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))

            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id, feature_selector=self.features
            )

            lines = []
            for region in page.get_AllRegions(classes=["Text"]):
                region_image, region_coords = self.workspace.image_from_segment(
                    region, page_image, page_coords, feature_selector=self.features
                )

                textlines = region.get_TextLine()
                log.info(
                    "About to recognize %i lines of region '%s'",
                    len(textlines),
                    region.id,
                )
                for line in textlines:
                    log.debug(
                        "Recognizing line '%s' in region '%s'", line.id, region.id
                    )

                    line_image, line_coords = self.workspace.image_from_segment(
                        line,
                        region_image,
                        region_coords,
                        feature_selector=self.features,
                    )
                    if (
                        "binarized" not in line_coords["features"]
                        and "grayscale_normalized" not in line_coords["features"]
                        and self.network_input_channels == 1
                    ):
                        # We cannot use a feature selector for this since we don't
                        # know whether the model expects (has been trained on)
                        # binarized or grayscale images; but raw images are likely
                        # always inadequate:
                        log.warning(
                            "Using raw image for line '%s' in region '%s'",
                            line.id,
                            region.id,
                        )

                    if (
                        not all(line_image.size)
                        or line_image.height <= 8
                        or line_image.width <= 8
                        or "binarized" in line_coords["features"]
                        and line_image.convert("1").getextrema()[0] == 255
                    ):
                        # empty size or too tiny or no foreground at all: skip
                        log.warning(
                            "Skipping empty line '%s' in region '%s'",
                            line.id,
                            region.id,
                        )
                        continue
                    lines.append((line, line_coords, np.array(line_image, dtype=np.uint8)))

            if len(lines):
                lines, coords, images = zip(*lines)
            else:
                log.warning("No text lines on page '%s'", page_id)
                lines, coords, images = [], [], []

            # not exposed in MultiPredictor yet, cf. calamari#361:
            # raw_results_all = self.predictor.predict_raw(images, progress_bar=False, batch_size=BATCH_SIZE)
            # avoid too large a batch size (causing OOM on CPU or GPU)
            fun = lambda x: self.predictor.predict_raw(x, progress_bar=False)
            results = itertools.chain.from_iterable(
                map(fun, batched_length_limited(images, BATCH_SIZE)))

            for line, line_coords, raw_results in zip(lines, coords, results):
                for i, p in enumerate(raw_results):
                    p.prediction.id = "fold_{}".format(i)

                prediction = self.voter.vote_prediction_result(raw_results)
                prediction.id = "voted"

                # Build line text on our own
                #
                # Calamari does whitespace post-processing on prediction.sentence,
                # while it does not do the same on prediction.positions. Do it on
                # our own to have consistency.
                #
                # XXX Check Calamari's built-in post-processing on
                #     prediction.sentence

                def _sort_chars(p):
                    """Filter and sort chars of prediction p"""
                    chars = p.chars
                    chars = [
                        c for c in chars if c.char
                    ]  # XXX Note that omission probabilities are not normalized?!
                    chars = [
                        c
                        for c in chars
                        if c.probability >= self.parameter["glyph_conf_cutoff"]
                    ]
                    chars = sorted(chars, key=lambda k: k.probability, reverse=True)
                    return chars

                def _drop_leading_spaces(positions):
                    return list(
                        itertools.dropwhile(
                            lambda p: _sort_chars(p)[0].char == " ", positions
                        )
                    )

                def _drop_trailing_spaces(positions):
                    return list(reversed(_drop_leading_spaces(reversed(positions))))

                def _drop_double_spaces(positions):
                    def _drop_double_spaces_generator(positions):
                        last_was_space = False
                        for p in positions:
                            if p.chars[0].char == " ":
                                if not last_was_space:
                                    yield p
                                last_was_space = True
                            else:
                                yield p
                                last_was_space = False

                    return list(_drop_double_spaces_generator(positions))

                positions = prediction.positions
                positions = _drop_leading_spaces(positions)
                positions = _drop_trailing_spaces(positions)
                positions = _drop_double_spaces(positions)
                positions = list(positions)

                line_text = "".join(_sort_chars(p)[0].char for p in positions)
                if line_text != prediction.sentence:
                    log.warning(
                        f"Our own line text is not the same as Calamari's:"
                        f"'{line_text}' != '{prediction.sentence}'"
                    )

                # Delete existing results
                if line.get_TextEquiv():
                    log.warning("Line '%s' already contained text results", line.id)
                line.set_TextEquiv([])
                if line.get_Word():
                    log.warning(
                        "Line '%s' already contained word segmentation", line.id
                    )
                line.set_Word([])

                # Save line results
                line_conf = prediction.avg_char_probability
                line.set_TextEquiv(
                    [TextEquivType(Unicode=line_text, conf=line_conf)]
                )

                # Save word results
                #
                # Calamari OCR does not provide word positions, so we infer word
                # positions from a. text segmentation and b. the glyph positions.
                # This is necessary because the PAGE XML format enforces a strict
                # hierarchy of lines > words > glyphs.

                def _words(s):
                    """Split words based on spaces and include spaces as 'words'"""
                    spaces = None
                    word = ""
                    for c in s:
                        if c == " " and spaces is True:
                            word += c
                        elif c != " " and spaces is False:
                            word += c
                        else:
                            if word:
                                yield word
                            word = c
                            spaces = c == " "
                    yield word

                if self.parameter["textequiv_level"] in ["word", "glyph"]:
                    word_no = 0
                    i = 0

                    for word_text in _words(line_text):
                        word_length = len(word_text)
                        if not all(c == " " for c in word_text):
                            word_positions = positions[i : i + word_length]
                            word_start = word_positions[0].global_start
                            word_end = word_positions[-1].global_end

                            polygon = polygon_from_x0y0x1y1(
                                [word_start, 0, word_end, line_image.height]
                            )
                            points = points_from_polygon(
                                coordinates_for_segment(polygon, None, line_coords)
                            )
                            # XXX Crop to line polygon?

                            word = WordType(
                                id="%s_word%04d" % (line.id, word_no),
                                Coords=CoordsType(points),
                            )
                            word.add_TextEquiv(TextEquivType(Unicode=word_text))

                            if self.parameter["textequiv_level"] == "glyph":
                                for glyph_no, p in enumerate(word_positions):
                                    glyph_start = p.global_start
                                    glyph_end = p.global_end

                                    polygon = polygon_from_x0y0x1y1(
                                        [
                                            glyph_start,
                                            0,
                                            glyph_end,
                                            line_image.height,
                                        ]
                                    )
                                    points = points_from_polygon(
                                        coordinates_for_segment(
                                            polygon, None, line_coords
                                        )
                                    )

                                    glyph = GlyphType(
                                        id="%s_glyph%04d" % (word.id, glyph_no),
                                        Coords=CoordsType(points),
                                    )

                                    # Add predictions (= TextEquivs)
                                    char_index_start = 1
                                    # Index must start with 1, see
                                    # https://ocr-d.github.io/page#multiple-textequivs
                                    for char_index, char in enumerate(
                                        _sort_chars(p), start=char_index_start
                                    ):
                                        glyph.add_TextEquiv(
                                            TextEquivType(
                                                Unicode=char.char,
                                                index=char_index,
                                                conf=char.probability,
                                            )
                                        )

                                    word.add_Glyph(glyph)

                            line.add_Word(word)
                            word_no += 1

                        i += word_length

            _page_update_higher_textequiv_levels("line", pcgts)

            # Add metadata about this operation and its runtime parameters:
            self.add_metadata(pcgts)
            file_id = make_file_id(input_file, self.output_file_grp)
            pcgts.set_pcGtsId(file_id)
            self.workspace.add_file(
                file_id=file_id,
                file_grp=self.output_file_grp,
                page_id=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + ".xml"),
                content=to_xml(pcgts),
            )


# TODO: This is a copy of ocrd_tesserocr's function, and should probably be moved to a
#       ocrd lib
def _page_update_higher_textequiv_levels(level, pcgts):
    """Update the TextEquivs of all higher PAGE-XML hierarchy levels for consistency.

    Starting with the hierarchy level `level`chosen for processing, join all first
    TextEquiv (by the rules governing the respective level) into TextEquiv of the next
    higher level, replacing them.
    """
    regions = pcgts.get_Page().get_TextRegion()
    if level != "region":
        for region in regions:
            lines = region.get_TextLine()
            if level != "line":
                for line in lines:
                    words = line.get_Word()
                    if level != "word":
                        for word in words:
                            glyphs = word.get_Glyph()
                            word_unicode = "".join(
                                (
                                    glyph.get_TextEquiv()[0].Unicode
                                    if glyph.get_TextEquiv()
                                    else ""
                                )
                                for glyph in glyphs
                            )
                            word.set_TextEquiv(
                                [TextEquivType(Unicode=word_unicode)]
                            )  # remove old
                    line_unicode = " ".join(
                        word.get_TextEquiv()[0].Unicode if word.get_TextEquiv() else ""
                        for word in words
                    )
                    line.set_TextEquiv(
                        [TextEquivType(Unicode=line_unicode)]
                    )  # remove old
            region_unicode = "\n".join(
                line.get_TextEquiv()[0].Unicode if line.get_TextEquiv() else ""
                for line in lines
            )
            region.set_TextEquiv([TextEquivType(Unicode=region_unicode)])  # remove old
