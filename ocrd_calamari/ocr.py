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
from ocrd_utils import getLogger, concat_padded, polygon_from_points, MIMETYPE_PAGE

from ocrd_calamari.config import OCRD_TOOL, TF_CPP_MIN_LOG_LEVEL

log = getLogger('processor.CalamariOcr')

# TODO: Should this be "recognize", not "ocr" akin ocrd_tesserocr?


class CalamariOcr(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-calamari-ocr']
        super(CalamariOcr, self).__init__(*args, **kwargs)

    def _init_calamari(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL

        checkpoints = glob('/home/mike/devel/experiments/train-calamari-gt4histocr/models/*.ckpt.json')  # XXX
        self.predictor = MultiPredictor(checkpoints=checkpoints)

        voter_params = VoterParams()
        voter_params.type = VoterParams.Type.Value(self.parameter['voter'].upper())
        self.voter = voter_from_proto(voter_params)

    def resolve_image_as_np(self, image_url, coords):
        return np.array(self.workspace.resolve_image_as_pil(image_url, coords), dtype=np.uint8)

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
            log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            image_url = pcgts.get_Page().imageFilename
            log.info("pcgts %s", pcgts)
            for region in pcgts.get_Page().get_TextRegion():
                textlines = region.get_TextLine()
                log.info("About to recognize %i lines of region '%s'", len(textlines), region.id)
                for (line_no, line) in enumerate(textlines):
                    log.debug("Recognizing line '%s' in region '%s'", line_no, region.id)

                    image = self.resolve_image_as_np(image_url, polygon_from_points(line.get_Coords().points))

                    raw_results = list(self.predictor.predict_raw([image], progress_bar=False))[0]
                    for i, p in enumerate(raw_results):
                        p.prediction.id = "fold_{}".format(i)

                    prediction = self.voter.vote_prediction_result(raw_results)
                    prediction.id = "voted"

                    line_text = prediction.sentence
                    line_conf = prediction.avg_char_probability

                    line.add_TextEquiv(TextEquivType(Unicode=line_text, conf=line_conf))

            file_id = self._make_file_id(input_file, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts))
