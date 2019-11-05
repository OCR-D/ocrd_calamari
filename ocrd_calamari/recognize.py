from __future__ import absolute_import

import os
from glob import glob

import numpy as np
from calamari_ocr.ocr import MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_models.ocrd_page_generateds import TextEquivType
from ocrd_utils import getLogger, concat_padded, MIMETYPE_PAGE

from ocrd_calamari.config import OCRD_TOOL, TF_CPP_MIN_LOG_LEVEL

log = getLogger('processor.CalamariRecognize')


class CalamariRecognize(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-calamari-recognize']
        kwargs['version'] = OCRD_TOOL['version']
        super(CalamariRecognize, self).__init__(*args, **kwargs)

    def _init_calamari(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL

        checkpoints = glob(self.parameter['checkpoint'])
        self.predictor = MultiPredictor(checkpoints=checkpoints)

        self.input_channels = self.predictor.predictors[0].network.input_channels
        #self.input_channels = self.predictor.predictors[0].network_params.channels # not used!
        # binarization = self.predictor.predictors[0].model_params.data_preprocessor.binarization # not used!
        # self.features = ('' if self.input_channels != 1 else
        #                  'binarized' if binarization != 'GRAY' else
        #                  'grayscale_normalized')
        self.features = ''

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
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id, feature_selector=self.features)

            for region in page.get_TextRegion():
                region_image, region_coords = self.workspace.image_from_segment(
                    region, page_image, page_coords, feature_selector=self.features)

                textlines = region.get_TextLine()
                log.info("About to recognize %i lines of region '%s'", len(textlines), region.id)
                for line in textlines:
                    log.debug("Recognizing line '%s' in region '%s'", line.id, region.id)

                    line_image, line_coords = self.workspace.image_from_segment(
                        line, region_image, region_coords, feature_selector=self.features)
                    if ('binarized' not in line_coords['features'] and
                        'grayscale_normalized' not in line_coords['features'] and
                        self.input_channels == 1):
                        # We cannot use a feature selector for this since we don't
                        # know whether the model expects (has been trained on)
                        # binarized or grayscale images; but raw images are likely
                        # always inadequate:
                        log.warning("Using raw image for line '%s' in region '%s'",
                                    line.id, region.id)
                    
                    line_image_np = np.array(line_image, dtype=np.uint8)

                    raw_results = list(self.predictor.predict_raw([line_image_np], progress_bar=False))[0]
                    for i, p in enumerate(raw_results):
                        p.prediction.id = "fold_{}".format(i)

                    prediction = self.voter.vote_prediction_result(raw_results)
                    prediction.id = "voted"

                    line_text = prediction.sentence
                    line_conf = prediction.avg_char_probability

                    if line.get_TextEquiv():
                        log.warning("Line '%s' already contained text results", line.id)
                    line.set_TextEquiv([TextEquivType(Unicode=line_text, conf=line_conf)])
                    
                    if line.get_Word():
                        log.warning("Line '%s' already contained word segmentation", line.id)
                    line.set_Word([])

            _page_update_higher_textequiv_levels('line', pcgts)

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
