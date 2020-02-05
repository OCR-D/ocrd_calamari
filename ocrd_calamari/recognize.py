from __future__ import absolute_import

import os
from glob import glob

import numpy as np
from calamari_ocr.ocr import MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
        LabelType, LabelsType,
        MetadataItemType,
        TextEquivType,
        WordType, GlyphType, CoordsType,
        to_xml
)
from ocrd_utils import (
        getLogger, concat_padded,
        coordinates_for_segment, points_from_polygon, polygon_from_x0y0x1y1,
        MIMETYPE_PAGE
)

from ocrd_calamari.config import OCRD_TOOL, TF_CPP_MIN_LOG_LEVEL

TOOL = 'ocrd-calamari-recognize'
log = getLogger('processor.CalamariRecognize')


class CalamariRecognize(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(CalamariRecognize, self).__init__(*args, **kwargs)

    def _init_calamari(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL

        checkpoints = glob(self.parameter['checkpoint'])
        self.predictor = MultiPredictor(checkpoints=checkpoints)

        voter_params = VoterParams()
        voter_params.type = VoterParams.Type.Value(self.parameter['voter'].upper())
        self.voter = voter_from_proto(voter_params)

    def _make_file_id(self, input_file, n):
        file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)
        if file_id == input_file.ID:
            file_id = concat_padded(self.output_file_grp, n)
        return file_id

    def process(self):
        """
        Performs the recognition.
        """

        self._init_calamari()

        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            log.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))

            page = pcgts.get_Page()
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(page, page_id)

            for region in pcgts.get_Page().get_TextRegion():
                region_image, region_xywh = self.workspace.image_from_segment(region, page_image, page_xywh)

                textlines = region.get_TextLine()
                log.info("About to recognize %i lines of region '%s'", len(textlines), region.id)
                for (line_no, line) in enumerate(textlines):
                    log.debug("Recognizing line '%s' in region '%s'", line_no, region.id)

                    line_image, line_coords = self.workspace.image_from_segment(line, region_image, region_xywh)
                    line_image_np = np.array(line_image, dtype=np.uint8)

                    raw_results = list(self.predictor.predict_raw([line_image_np], progress_bar=False))[0]
                    for i, p in enumerate(raw_results):
                        p.prediction.id = "fold_{}".format(i)

                    prediction = self.voter.vote_prediction_result(raw_results)
                    prediction.id = "voted"

                    line_text = prediction.sentence
                    line_conf = prediction.avg_char_probability

                    # Delete existing results
                    if line.get_TextEquiv():
                        log.warning("Line '%s' already contained text results", line.id)
                    line.set_TextEquiv([])
                    if line.get_Word():
                        log.warning("Line '%s' already contained word segmentation", line.id)
                    line.set_Word([])

                    # Save line results
                    line.set_TextEquiv([TextEquivType(Unicode=line_text, conf=line_conf)])

                    # Save word results
                    #
                    # Calamari OCR does not provide word positions, so we infer word positions from a. text segmentation
                    # and b. the glyph positions. This is necessary because the PAGE XML format enforces a strict
                    # hierarchy of lines > words > glyphs.

                    def _words(s):
                        """Split words based on spaces and include spaces as 'words'"""
                        spaces = None
                        word = ''
                        for c in s:
                            if c == ' ' and spaces is True:
                                word += c
                            elif c != ' ' and spaces is False:
                                word += c
                            else:
                                if word:
                                    yield word
                                word = c
                                spaces = (c == ' ')
                        yield word

                    if self.parameter['textequiv_level'] in ['word', 'glyph']:
                        word_no = 0
                        i = 0

                        for word_text in _words(prediction.sentence):
                            word_length = len(word_text)
                            if not all(c == ' ' for c in word_text):
                                word_positions = prediction.positions[i:i+word_length]
                                word_start = word_positions[0].global_start
                                word_end = word_positions[-1].global_end

                                polygon = polygon_from_x0y0x1y1([word_start, 0, word_end, line_image.height])
                                points = points_from_polygon(coordinates_for_segment(polygon, None, line_coords))
                                # XXX Crop to line polygon?

                                word = WordType(id='%s_word%04d' % (line.id, word_no), Coords=CoordsType(points))
                                word.add_TextEquiv(TextEquivType(Unicode=word_text))

                                if self.parameter['textequiv_level'] == 'glyph':
                                    for glyph_no, p in enumerate(word_positions):
                                        glyph_start = p.global_start
                                        glyph_end = p.global_end

                                        polygon = polygon_from_x0y0x1y1([glyph_start, 0, glyph_end, line_image.height])
                                        points = points_from_polygon(coordinates_for_segment(polygon, None, line_coords))

                                        glyph = GlyphType(id='%s_glyph%04d' % (word.id, glyph_no), Coords=CoordsType(points))

                                        # Filter predictions
                                        chars = p.chars
                                        chars = [c for c in chars if c.char]  # XXX Note that omission probabilities are not normalized?!
                                        chars = [c for c in chars if c.probability >= self.parameter['glyph_conf_cutoff']]

                                        # Sort and add predictions (= TextEquivs)
                                        chars = sorted(chars, key=lambda k: k.probability, reverse=True)
                                        char_index = 1  # Must start with 1, see https://ocr-d.github.io/page#multiple-textequivs
                                        for char in chars:
                                            glyph.add_TextEquiv(TextEquivType(Unicode=char.char, index=char_index, conf=char.probability))
                                            char_index += 1

                                        word.add_Glyph(glyph)

                                line.add_Word(word)
                                word_no += 1

                            i += word_length


            _page_update_higher_textequiv_levels('line', pcgts)


            # Add metadata about this operation and its runtime parameters:
            metadata = pcgts.get_Metadata()  # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(
                                     externalModel="ocrd-tool",
                                     externalId="parameters",
                                     Label=[LabelType(type_=name, value=self.parameter[name])
                                            for name in self.parameter.keys()])]))


            file_id = self._make_file_id(input_file, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts))


# TODO: This is a copy of ocrd_tesserocr's function, and should probably be moved to a ocrd lib
def _page_update_higher_textequiv_levels(level, pcgts):
    """Update the TextEquivs of all PAGE-XML hierarchy levels above `level` for consistency.

    Starting with the hierarchy level chosen for processing,
    join all first TextEquiv (by the rules governing the respective level)
    into TextEquiv of the next higher level, replacing them.
    """
    regions = pcgts.get_Page().get_TextRegion()
    if level != 'region':
        for region in regions:
            lines = region.get_TextLine()
            if level != 'line':
                for line in lines:
                    words = line.get_Word()
                    if level != 'word':
                        for word in words:
                            glyphs = word.get_Glyph()
                            word_unicode = u''.join(glyph.get_TextEquiv()[0].Unicode
                                                    if glyph.get_TextEquiv()
                                                    else u'' for glyph in glyphs)
                            word.set_TextEquiv(
                                [TextEquivType(Unicode=word_unicode)])  # remove old
                    line_unicode = u' '.join(word.get_TextEquiv()[0].Unicode
                                             if word.get_TextEquiv()
                                             else u'' for word in words)
                    line.set_TextEquiv(
                        [TextEquivType(Unicode=line_unicode)])  # remove old
            region_unicode = u'\n'.join(line.get_TextEquiv()[0].Unicode
                                        if line.get_TextEquiv()
                                        else u'' for line in lines)
            region.set_TextEquiv(
                [TextEquivType(Unicode=region_unicode)])  # remove old

# vim:tw=120:
