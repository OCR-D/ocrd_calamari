from __future__ import absolute_import

from typing import Optional
from functools import cached_property
import itertools
from glob import glob
import atexit
import queue
import multiprocessing as mp
from threading import Thread
import logging

import numpy as np
import cv2 as cv
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
    initLogging,
    config
)

# ruff: isort: on

BATCH_SIZE = 3
GROUP_BOUNDS = [100, 200, 400, 800, 1600, 3200]
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

        # Run in a background thread so GPU parts can be interleaved with CPU pre-/post-processing across pages.
        # We cannot use a ProcessPoolExecutor (or even ThreadPoolExecutor) for this,
        # because that relies on threads to set up IPC, but when process_workspace
        # starts forking/spawning subprocesses, these threads will break.
        # (And we cannot use multithreading for process_workspace either, because
        # Python's GIL would not allow true multiscalar compuation in the first place.)
        # So instead, here we setup our own subprocess+queueing solution.
        self.predictor = CalamariPredictor(
                self.parameter['device'],
                self.parameter["voter"],
                self.resolve_resource(self.parameter["checkpoint_dir"])
        )
        self.logger.debug("model's network_input_channels is %d", self.network_input_channels)

    @cached_property
    def network_input_channels(self):
        # as a special case, this information from the model is needed prior to
        # prediction, but must be retrieved from the background process as soon as
        # the model is loaded, so this will block upon first invocation
        input_channels = self.predictor.network_input_channels
        return input_channels

    def shutdown(self):
        if getattr(self, 'predictor', None):
            self.predictor.shutdown()
            del self.predictor

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

        tasks = []
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

                line_img = load_image(
                    np.array(line_image, dtype=np.uint8),
                    self.network_input_channels
                )
                if (
                    not all(line_image.size)
                    or line_image.height <= 8
                    or line_image.width <= 8
                    or "binarized" in line_coords["features"]
                    and line_img.min() == 255
                ):
                    # empty size or too tiny or no foreground at all: skip
                    self.logger.warning(
                        "Skipping empty line '%s' in region '%s'",
                        line.id,
                        region.id,
                    )
                    continue

                tasks.append(Thread(target=self._process_line,
                                    args=(line, line_coords, line_img, page_id),
                                    name="LinePredictor-%s-%s" % (page_id, line.id)))
                tasks[-1].start()

        if not len(tasks):
            self.logger.warning("No text lines on page '%s'", page_id)
            return OcrdPageResult(pcgts)

        # ids, lines, coords, images = zip(*lines)
        # We cannot delegate to predictor.predict_raw directly...
        #    predictions = self.predictor.predict_raw(images)
        # ...because for efficiency, all page tasks must be synchronised
        # on a single GPU-bound subprocess (no more than 1 simulatneous call).
        # Moreover, we also cannot use predict_raw indirectly...
        #    taskq.put((page_id, images))
        #                                                page_id, images = taskq.get()
        #                                                result = predictor.predict_raw(images)
        #                                                resultq.put((page_id, result))
        #    predictions = resultq.get(page_id)
        # ...because our pipeline params use bucket batching, i.e. reordering,
        # so we have to pass in additional metadata for re-identification.
        # Hence our predict_raw() re-implementation in CalamariPredictor.
        # predictions = self.predictor(images, ids, page_id=page_id)
        # # Map back predictions to lines via sample metadata
        # predict = {prediction.meta["id"]: prediction.outputs for prediction in predictions}
        # self.logger.info("Received %d line results for page '%s'", len(predict.keys()), page_id)
        # for line, line_coords in zip(lines, coords):
        #     raw_results, prediction = predict[line.id]
        for task in tasks:
            task.join()

        _page_update_higher_textequiv_levels("line", pcgts)
        return OcrdPageResult(pcgts)

    def _process_line(self, line, line_coords, line_image, page_id):
        result = self.predictor(line_image, line.id, page_id)
        self.logger.info("Received line result for page '%s' line '%s'", page_id, line.id)
        self._post_process_line(line, line_coords, result)

    def _post_process_line(self, line, line_coords, result):
        _, prediction = result

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

class CalamariPredictor:
    class QueueStop:
        pass

    class PredictWorker(mp.Process):
        def __init__(self, logger, device, voter, checkpoint_dir, taskq, resultq, terminate):
            self.logger = logger # FIXME: synchronize loggers, too
            self.device = device
            self.voter = voter
            self.checkpoint_dir = checkpoint_dir
            self.taskq = taskq
            self.resultq = resultq
            self.terminate = terminate
            super().__init__()
        def put(self, result):
            while not self.terminate.is_set():
                try:
                    self.resultq.put(result, timeout=9)
                    return
                except queue.Full:
                    continue
            self.logger.warning("dropping result for page '%s'", result[0])
        def run(self):
            initLogging()
            tf_disable_interactive_logs()
            try:
                predictor = self.setup_predictor()
                generator = self.setup_pipelines(predictor)
                generator = iter(generator())
                self.put(("input_channels", predictor.data.params.input_channels))
            except Exception as e:
                self.put(("input_channels", e))
                # unrecoverable
                self.terminate.set()
            while not self.terminate.is_set():
                try:
                    prediction = next(generator) # StopIteration?
                    page_id, line_id = prediction.meta["id"]
                    result = prediction.outputs
                    self.put((page_id, line_id, result))
                    self.logger.debug("sent result for page '%s' line '%s'", page_id, line_id)
                except Exception as e:
                    # full traceback gets shown when base Processor handles exception
                    #self.logger.error("prediction failed: %s", e.__class__.__name__)
                    self.logger.exception("prediction failed")
                    self.put(("", "", e)) # for which page/line??
                    # FIXME: or do we have to re-initialize Tensorflow here?
            self.resultq.close()
            self.resultq.cancel_join_thread()
        def setup_predictor(self):
            """
            Set up the model prior to processing.
            """
            from calamari_ocr.ocr.predict.predictor import MultiPredictor, PredictorParams
            from calamari_ocr.ocr.voting import VoterParams, VoterType
            from tfaip.data.databaseparams import DataPipelineParams
            from tfaip import DeviceConfigParams
            from tfaip.device.device_config import DistributionStrategy
            import tensorflow as tf
            # unfortunately, tfaip device selector is mandatory and does not provide auto-detection
            if self.device < 0:
                gpus = []
                self.logger.debug("running on CPU")
            elif self.device < len(tf.config.list_physical_devices("GPU")):
                gpus = [self.device]
                self.logger.info("running on selected GPU device cuda:%d", self.device)
            else:
                gpus = []
                self.logger.warning("running on CPU because selected GPU device cuda:%d is not available", self.device)
            # load model
            pred_params = PredictorParams(
                silent=True,
                progress_bar=False,
                device=DeviceConfigParams(
                    gpus=gpus,
                    soft_device_placement=False,
                    #gpu_memory=7000, # limit to 7GB (logical, no dynamic growth)
                    #dist_strategy=DistributionStrategy.CENTRAL_STORAGE,
                ),
                pipeline=DataPipelineParams(
                    batch_size=BATCH_SIZE,
                    # Number of processes for data loading.
                    num_processes=4,
                    use_shared_memory=True,
                    # group lines with similar lengths to reduce need for padding
                    # and optimally utilise batch size;
                    bucket_boundaries=GROUP_BOUNDS,
                    bucket_batch_sizes=BATCH_GROUPS,
                )
            )
            voter_params = VoterParams()
            voter_params.type = VoterType(self.voter)
            #
            checkpoints = glob("%s/*.ckpt.json" % self.checkpoint_dir)
            self.logger.info("loading %d checkpoints", len(checkpoints))
            predictor = MultiPredictor.from_paths(
                checkpoints,
                voter_params=voter_params,
                predictor_params=pred_params,
            )
            #predictor.data.params.pre_proc.run_parallel = False
            #predictor.data.params.post_proc.run_parallel = False
            def element_length_fn(x):
                return x["img_len"]
            predictor.data.element_length_fn=lambda: element_length_fn
            # rewrap voter JoinedModel and compile (to avoid repeating for each page):
            class WrappedModel(tf.keras.models.Model):
                def call(self, inputs, training=None, mask=None):
                    inputs, meta = inputs
                    return inputs, predictor._keras_model(inputs), meta
            predictor.model = WrappedModel()
            # for preproc in predictor.data.params.pre_proc.processors:
            #     self.logger.info("preprocessor: %s", str(preproc))
            predictor.voter = predictor.create_voter(predictor.data.params)
            return predictor
        def setup_pipelines(self, predictor):
            # set up pipeline and generators (as infinite dataset)
            from dataclasses import field, dataclass
            from paiargparse import pai_dataclass
            from tfaip import Sample
            from tfaip.data.databaseparams import DataGeneratorParams
            from tfaip.data.pipeline.datapipeline import DataPipeline
            from tfaip.data.pipeline.datagenerator import DataGenerator
            from tfaip.data.pipeline.runningdatapipeline import _wrap_dataset
            @pai_dataclass
            @dataclass
            class QueueDataGeneratorParams(DataGeneratorParams):
                terminate : mp.Event = field(default=None)
                taskq : mp.Queue = field(default=None)
                @staticmethod
                def cls():
                    return QueueDataGenerator
            class QueueDataGenerator(DataGenerator[QueueDataGeneratorParams]):
                def __len__(self):
                    raise NotImplementedError()
                def generate(self):
                    while not self.params.terminate.is_set():
                        try:
                            page_id, line_id, image = self.params.taskq.get(timeout=11)
                        except queue.Empty:
                            continue
                        yield Sample(inputs=image, meta={"id": (page_id, line_id)})
            class QueueDataPipeline(DataPipeline):
                def create_data_generator(self):
                    return QueueDataGenerator(mode=self.mode, params=self.generator_params)
                def input_dataset(self, auto_repeat=None):
                    gen = self.generate_input_samples(auto_repeat=auto_repeat)
                    #return gen.as_dataset(self._create_tf_dataset_generator())
                    gen.running_pipeline = gen.processor_pipeline_params.create(gen.pipeline_params, gen.data_params)
                    def generator():
                        running_samples_generator = gen._generate_input_samples()
                        for sample in running_samples_generator:
                            yield sample
                        running_samples_generator.close()
                    dataset = self._create_tf_dataset_generator().create(generator, False)
                    dataset = _wrap_dataset(
                        self.mode, dataset, self.pipeline_params, self.data, False
                    )
                    return dataset
            self.logger.debug("setting up input pipeline")
            input_pipeline = QueueDataPipeline(
                predictor.params.pipeline, predictor._data,
                QueueDataGeneratorParams(terminate=self.terminate, taskq=self.taskq))
            from tfaip.predict.predictorbase import data_adapter
            from tfaip.util.tftyping import sync_to_numpy_or_python_type
            from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams
            from tfaip.predict.multimodelpostprocessor import MultiModelPostProcessorParams
            self.logger.debug("instantiating input dataset")
            tf_dataset = input_pipeline.input_dataset()
            import tensorflow as tf
            # do we really need that?
            tf_dataset = tf_dataset.apply(
                tf.data.experimental.assert_cardinality(tf.data.INFINITE_CARDINALITY)
            )
            self.logger.debug("setting up output pipeline")
            def predict_dataset(dataset):
                for batch in dataset:
                    r = predictor.model.predict_on_batch(batch)
                    inputs, outputs, meta = sync_to_numpy_or_python_type(r)
                    for sample in predictor._unwrap_batch(inputs, {}, outputs, meta):
                        yield sample
            post_processors = [
                d.get_or_create_pipeline(predictor.params.pipeline, input_pipeline.generator_params).create_output_pipeline()
                for d in predictor.datas
            ]
            post_proc_pipeline = SequentialProcessorPipelineParams(
                processors=[MultiModelPostProcessorParams(voter=predictor.voter, post_processors=post_processors)],
                run_parallel=predictor.data.params.post_proc.run_parallel,
                num_threads=predictor.data.params.post_proc.num_threads,
                max_tasks_per_process=predictor.data.params.post_proc.max_tasks_per_process,
            ).create(input_pipeline.pipeline_params, predictor.data.params)
            def output_generator():
                for sample in post_proc_pipeline.apply(predict_dataset(tf_dataset)):
                    yield predictor.voter.finalize_sample(sample)
            return output_generator

    def __init__(self, device, voter, checkpoint_dir):
        self.logger = logging.getLogger("ocrd.processor.CalamariPredictor")
        ctxt = mp.get_context('spawn') # not necessary to fork, and spawn is safer
        self.taskq = ctxt.Queue(maxsize=3 + config.OCRD_MAX_PARALLEL_PAGES * 200) # 3 + npages * nlines
        self.resultq = ctxt.Queue(maxsize=3 + config.OCRD_MAX_PARALLEL_PAGES * 200)
        self.terminate = ctxt.Event() # will be shared across all page workers forked from this process
        self.workers = [
            CalamariPredictor.PredictWorker(self.logger, device, voter, checkpoint_dir,
                                            self.taskq, self.resultq, self.terminate)
        ]
        atexit.register(self.shutdown) # sets self.terminate (on exception or gc)
        for p in self.workers:
            p.start()
        id_, self.network_input_channels = self.resultq.get() # block
        assert id_ == "input_channels" # sole possible task during setup/init
        if isinstance(self.network_input_channels, Exception):
            raise self.network_input_channels
        self.logger.info("Loaded model")
        # prior to base Processor forking page workers, ensure we can sync
        # multiple CalamariPredictors communicating with the same PredictWorker:
        ctxt = mp.get_context("fork") # base.Processor will fork workers
        self.results = ctxt.Manager().dict() # {}

    def __call__(self, image, line_id, page_id):
        self.taskq.put((page_id, line_id, image))
        self.logger.debug("sent image for page '%s' line '%s'", page_id, line_id)
        result = self.get(page_id, line_id)
        self.logger.debug("received result for page '%s' line '%s'", page_id, line_id)
        return result

    def get(self, page_id, line_id):
        self.logger.debug("requested result for page '%s' line '%s'", page_id, line_id)
        while not self.terminate.is_set():
            if (page_id, line_id) in self.results:
                result = self.results.pop((page_id, line_id))
                # if isinstance(result, Exception):
                #     raise Exception(f"prediction failed for page {page_id}") from result
                return result
            try:
                page_id, line_id, result = self.resultq.get(timeout=11)
            except queue.Empty:
                continue
            # FIXME what if page_id == line_id == "" and result is an exception??
            self.logger.debug("storing results for page '%s' line '%s'", page_id, line_id)
            self.results[(page_id, line_id)] = result
        for page_id, line_id in self.results.keys():
            self.logger.warning("dropping results for page '%s'", page_id)
        return None

    def shutdown(self):
        self.terminate.set()
        # while not self.taskq.empty():
        #     page_id, _, _ = self.taskq.get()
        #     self.logger.warning("dropped task for page %s", page_id)
        self.taskq.close()
        self.taskq.cancel_join_thread()


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

# from calamari_ocr.utils.image.ImageLoader (but for PIL.Image objects)
# (Calamari2 does not tolerate wrong input shape anymore -
#  common preprocessors do not change last dimension)
def load_image(img: np.ndarray, channels: int, to_gray_method : str = "cv") -> np.ndarray:
    if len(img.shape) == 2:
        img_channels = 1
    elif len(img.shape) == 3:
        img_channels = img.shape[-1]
    else:
        raise ValueError(f"Unknown image format. Must bei either WxH or WxHxC, but got {img.shape}.")

    if img_channels == channels:
        pass  # good
    elif img_channels == 2 and channels == 1:
        img = img[:, :, 0]
    elif img_channels == 3 and channels == 1:
        if to_gray_method == "avg":
            img = np.mean(img.astype("float32"), axis=-1).astype(dtype=img.dtype)
        elif to_gray_method == "cv":
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported image conversion method {to_gray_method}")
    elif img_channels == 4 and channels == 1:
        if to_gray_method == "avg":
            img = np.mean(img[:, :, :3].astype("float32"), axis=-1).astype(dtype=img.dtype)
        elif to_gray_method == "cv":
            img = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
        else:
            raise ValueError(f"Unsupported image conversion method {to_gray_method}")
    elif img_channels == 1 and channels == 3:
        img = np.stack([img] * 3, axis=-1)
    else:
        raise ValueError(
            f"Unsupported image format. Trying to convert from {img_channels} channels to "
            f"{channels} channels."
        )
    return img
