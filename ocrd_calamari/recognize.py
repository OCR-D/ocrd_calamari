from __future__ import absolute_import

from typing import Optional
import itertools
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ocrd import Processor, OcrdPage, OcrdPageResult
from ocrd_models.ocrd_page import (
    CoordsType,
    GlyphType,
    TextEquivType,
    WordType,
)
from ocrd_utils import (
    VERSION as OCRD_VERSION,
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_x0y0x1y1,
    tf_disable_interactive_logs,
)

# ruff: isort: on

BATCH_SIZE = 12
GROUP_BOUNDS = [100, 200, 400, 800, 1600, 3200, 6400]
# default tfaip bucket_batch_sizes is buggy (inverse quotient)
BATCH_GROUPS = [max(1, (max(GROUP_BOUNDS) * BATCH_SIZE) // length)
                for length in GROUP_BOUNDS] + [BATCH_SIZE]

class CalamariRecognize(Processor):
    @property
    def executable(self):
        return 'ocrd-calamari-recognize'

    def show_version(self):
        from tensorflow import __version__ as tensorflow_version
        from calamari_ocr import __version__ as calamari_version
        from tfaip import __version__ as tfaip_version
        print(f"Version {self.version}, "
              f"calamari {calamari_version}, "
              f"tfaip {tfaip_version}, "
              f"tensorflow {tensorflow_version}, "
              f"ocrd/core {OCRD_VERSION}"
        )

    def setup_calamari(self):
        """
        Set up the model prior to processing.
        """
        from calamari_ocr.ocr.predict.predictor import MultiPredictor, PredictorParams
        from calamari_ocr.ocr.voting import VoterParams, VoterType
        from tfaip.data.databaseparams import DataPipelineParams
        from tfaip import DeviceConfigParams
        from tfaip.device.device_config import DistributionStrategy
        tf_disable_interactive_logs()
        # load model
        pred_params = PredictorParams(
            silent=True,
            progress_bar=False,
            # TODO: expose device parameter
            device=DeviceConfigParams(gpus=[0], dist_strategy=DistributionStrategy.CENTRAL_STORAGE),
            pipeline=DataPipelineParams(
                batch_size=BATCH_SIZE,
                # Number of processes for data loading.
                num_processes=4,
                # group lines with similar lengths to reduce need for padding
                # and optimally utilise batch size;
                bucket_boundaries=GROUP_BOUNDS,
                bucket_batch_sizes=BATCH_GROUPS,
            )
        )
        voter_params = VoterParams()
        voter_params.type = VoterType(self.parameter["voter"])

        resolved = self.resolve_resource(self.parameter["checkpoint_dir"])
        checkpoints = glob("%s/*.ckpt.json" % resolved)
        self.logger.info("loading %d checkpoints", len(checkpoints))
        self.predictor = MultiPredictor.from_paths(
            checkpoints,
            voter_params=voter_params,
            predictor_params=pred_params,
        )
        #self.predictor.data.params.pre_proc.run_parallel = False
        #self.predictor.data.params.post_proc.run_parallel = False
        def element_length_fn(x):
            return x["img_len"]
        self.predictor.data.element_length_fn=lambda: element_length_fn

        self.network_input_channels = self.predictor.data.params.input_channels
        for preproc in self.predictor.data.params.pre_proc.processors:
            self.logger.info("preprocessor: %s", str(preproc))

    def predict_raw(self, images, lines, page_id=""):
        # for instrumentation, reimplement raw data pipeline:
        from tfaip import PipelineMode, Sample
        from tfaip.data.databaseparams import DataGeneratorParams
        from tfaip.data.pipeline.datapipeline import RawDataPipeline
        # from tfaip.data.pipeline.datapipeline import DataPipeline
        # from tfaip.data.pipeline.datagenerator import DataGenerator
        # from tfaip.data.pipeline.runningdatapipeline import InputSamplesGenerator, _wrap_dataset
        # from PIL import Image
        # class MyInputSamplesGenerator(InputSamplesGenerator):
        #     def as_dataset(self, tf_dataset_generator):
        #         def generator():
        #             with self as samples:
        #                 # now instrument the processors in the running pipeline
        #                 # for pipeline in self.running_pipeline.pipeline:
        #                 #     print("pipeline: %s" % str(pipeline))
        #                 #     proc = pipeline.create_processor_fn()
        #                 #     for processor in proc.processors:
        #                 #         print("next processor: %s" % repr(processor))
        #                 for s in samples:
        #                     #Image.fromarray(s.inputs["img"].T.squeeze(), mode="L").save(s.meta["id"] + ".png")
        #                     yield s
        #         dataset = tf_dataset_generator.create(generator, self.data_generator.yields_batches())
        #         def print_fn(*x):
        #             import tensorflow as tf
        #             tf.print(tf.shape(x[0]["img"]))
        #             return x
        #         #dataset = dataset.map(print_fn)
        #         dataset = _wrap_dataset(
        #             self.mode, dataset, self.pipeline_params, self.data, self.data_generator.yields_batches())
        #         #dataset = dataset.map(print_fn)
        #         return dataset
        # class RawDataGenerator(DataGenerator):
        #     def __len__(self):
        #         return len(images)
        #     def generate(self):
        #         #return map(lambda x: Sample(inputs=x, meta={}), images)
        #         def to_sample(x):
        #             image, line = x
        #             return Sample(inputs=image, meta={"id": line.id})
        #         return map(to_sample, zip(images, lines))
        # class RawDataPipeline(DataPipeline):
        #     def create_data_generator(self):
        #         return RawDataGenerator(mode=self.mode, params=self.generator_params)
        #     def input_dataset_with_len(self, auto_repeat=None):
        #         #gen = self.generate_input_samples(auto_repeat=auto_repeat)
        #         gen = MyInputSamplesGenerator(
        #             self._input_processors,
        #             self.data,
        #             self.create_data_generator(),
        #             self.pipeline_params,
        #             self.mode,
        #             auto_repeat
        #         )
        #         return gen.as_dataset(self._create_tf_dataset_generator()), len(gen)
        # pipeline = RawDataPipeline(self.predictor.params.pipeline, self.predictor._data, DataGeneratorParams())
        # use tfaip's RawDataPipeline without instrumentation
        assert len(lines) == len(images)
        self.logger.debug("predicting %d images for page '%s'", len(images), page_id)
        pipeline = RawDataPipeline(
            [Sample(inputs=image, meta={"id": line.id})
                   for image, line in zip(images, lines)],
            self.predictor.params.pipeline,
            self.predictor._data,
            DataGeneratorParams(),
        )
        # list() - exhaust result generator to stay thread-safe:
        predictions = list(self.predictor.predict_pipeline(pipeline))
        self.logger.debug("predicted %d images for page '%s'", len(predictions), page_id)
        assert len(predictions) == len(images)
        return predictions

    def setup(self):
        """
        Set up the model prior to processing.
        """
        # not used:
        # binarization = any(isinstance(preproc, calamari_ocr.ocr.dataset.imageprocessors.center_normalizer.CenterNormalizerProcessorParams) for preproc in self.predictor.data.params.pre_proc.processors)
        # self.features = ('' if self.network_input_channels != 1 else
        #                  'binarized' if binarization != 'GRAY' else
        #                  'grayscale_normalized')
        self.features = ""

        # run in a background thread so GPU parts can be interleaved with CPU pre-/post-processing across pages
        self.executor = ThreadPoolExecutor(
            # only 1 (exclusive Tensoflow session)
            max_workers=1,
            thread_name_prefix='bgtask_calamari',
            # cannot just run initializer in parallel to processing,
            # because pages will need to know self.network_input_channels already
            #initializer=self.setup_calamari
        )
        self.executor.submit(self.setup_calamari).result()

    def shutdown(self):
        if getattr(self, 'predictor', None):
            del self.predictor
        if getattr(self, 'executor', None):
            self.executor.shutdown()
            del self.executor

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """
        Perform text recognition with Calamari.

        If ``texequiv_level`` is ``word`` or ``glyph``, then additionally create word /
        glyph level segments by splitting at white space characters / glyph boundaries.
        In the case of ``glyph``, add all alternative character hypotheses down to
        ``glyph_conf_cutoff`` confidence threshold.
        """
        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        page_image, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id, feature_selector=self.features
        )

        lines = []
        maxw = 0
        for region in page.get_AllRegions(classes=["Text"]):
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_coords, feature_selector=self.features
            )

            textlines = region.get_TextLine()
            self.logger.info(
                "About to recognize %i lines of region '%s'",
                len(textlines),
                region.id,
            )
            for line in textlines:
                self.logger.debug(
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
                    self.logger.warning(
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
                    self.logger.warning(
                        "Skipping empty line '%s' in region '%s'",
                        line.id,
                        region.id,
                    )
                    continue
                lines.append((line, line_coords, np.array(line_image, dtype=np.uint8)))

        if not len(lines):
            self.logger.warning("No text lines on page '%s'", page_id)
            return OcrdPageResult(pcgts)

        lines, coords, images = zip(*lines)
        # We cannot directly use the predictor, because all page threads must be synchronised
        # on a single GPU-bound thread.
        # predictions = self.predictor.predict_raw(images)
        # Also, we cannot directly use predict_raw, because our pipeline params use bucket batching,
        # i.e. reordering, so we have to pass in additional metadata for re-identification.
        # predictions = self.executor.submit(self.predictor.predict_raw, images).result()
        # See our predict_raw() implementation above.
        predictions = self.executor.submit(self.predict_raw, images, lines, page_id=page_id).result()
        # Map back predictions to lines via sample metadata
        predict = {prediction.meta["id"]: prediction.outputs for prediction in predictions}
        self.logger.info("Received %d line results for page '%s'", len(predict.keys()), page_id)

        #for line, line_coords, prediction in zip(lines, coords, predictions):
        for line, line_coords in zip(lines, coords):
            #raw_results, prediction = prediction.outputs
            raw_results, prediction = predict[line.id]

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
                self.logger.warning(
                    f"Our own line text is not the same as Calamari's:"
                    f"'{line_text}' != '{prediction.sentence}'"
                )

            # Delete existing results
            if line.get_TextEquiv():
                self.logger.warning("Line '%s' already contained text results", line.id)
            line.set_TextEquiv([])
            if line.get_Word():
                self.logger.warning(
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
            #
            # FIXME: use calamari#282 for this

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
        return OcrdPageResult(pcgts)

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
