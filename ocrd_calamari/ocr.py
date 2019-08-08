from __future__ import absolute_import

from glob import glob

import numpy as np
from calamari_ocr.ocr import MultiPredictor
from calamari_ocr.ocr.voting import voter_from_proto
from calamari_ocr.proto import VoterParams
from ocrd import Processor
from ocrd.logging import getLogger
from ocrd.model import ocrd_page
from ocrd.utils import polygon_from_points

from ocrd_calamari.config import OCRD_TOOL

log = getLogger('processor.CalamariOcr')

# TODO: Should this be "recognize", not "ocr" akin ocrd_tesserocr?


class CalamariOcr(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-calamari-ocr']
        super(CalamariOcr, self).__init__(*args, **kwargs)


    def _init_calamari(self):
        checkpoints = glob('/home/mike/devel/experiments/train-calamari-gt4histocr/models/*.ckpt.json')  # XXX
        self.predictor = MultiPredictor(checkpoints=checkpoints)

        voter_params = VoterParams()
        voter_params.type = VoterParams.Type.Value('confidence_voter_default_ctc'.upper())
        self.voter = voter_from_proto(voter_params)


    def process(self):
        """
        Performs the recognition.
        """

        self._init_calamari()

        for (n, input_file) in enumerate(self.input_files):
            log.info("INPUT FILE %i / %s", n, input_file)
            pcgts = ocrd_page.from_file(self.workspace.download_file(input_file))
            image_url = pcgts.get_Page().imageFilename
            log.info("pcgts %s", pcgts)
            for region in pcgts.get_Page().get_TextRegion():
                textlines = region.get_TextLine()
                log.info("About to recognize %i lines of region '%s'", len(textlines), region.id)
                for (line_no, line) in enumerate(textlines):
                    log.debug("Recognizing line '%s' in region '%s'", line_no, region.id)
                    image = self.workspace.resolve_image_as_pil(image_url,
                                                                polygon_from_points(line.get_Coords().points))
                    image_np = np.array(image, dtype=np.uint8)  # XXX better way?

                    raw_results = list(self.predictor.predict_raw([image_np], progress_bar=False))[0]

                    for i, p in enumerate(raw_results):
                        p.prediction.id = "fold_{}".format(i)

                    prediction = self.voter.vote_prediction_result(raw_results)
                    prediction.id = "voted"

                    print('***', prediction.sentence)
                    print(prediction.avg_char_probability)
                    for raw_result in raw_results:
                        print(raw_result.sentence)
