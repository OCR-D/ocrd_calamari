import os
from os.path import join, exists
import shutil

from test.base import TestCase, main, assets, skip

from ocrd.resolver import Resolver

from ocrd_tesserocr import TesserocrSegmentRegion
from ocrd_tesserocr import TesserocrSegmentLine

from ocrd_calamari import CalamariRecognize

#METS_HEROLD_SMALL = assets.url_of('SBB0000F29300010000/data/mets_one_file.xml')
# as long as #96 remains, we cannot use workspaces which have local relative files:
METS_HEROLD_SMALL = assets.url_of('kant_aufklaerung_1784-binarized/data/mets.xml')

WORKSPACE_DIR = '/tmp/test-ocrd-calamari'

class TestCalamariRecognize(TestCase):

    def setUp(self):
        if exists(WORKSPACE_DIR):
            shutil.rmtree(WORKSPACE_DIR)
        os.makedirs(WORKSPACE_DIR)

    #skip("Takes too long")
    def runTest(self):
        resolver = Resolver()
        workspace = resolver.workspace_from_url(METS_HEROLD_SMALL, dst_dir=WORKSPACE_DIR)

        TesserocrSegmentRegion(
            workspace,
            input_file_grp="OCR-D-IMG",
            output_file_grp="OCR-D-SEG-BLOCK"
        ).process()
        workspace.save_mets()

        TesserocrSegmentLine(
            workspace,
            input_file_grp="OCR-D-SEG-BLOCK",
            output_file_grp="OCR-D-SEG-LINE"
        ).process()
        workspace.save_mets()

        CalamariRecognize(
            workspace,
            input_file_grp="OCR-D-SEG-LINE",
            output_file_grp="OCR-D-OCR-CALAMARI",
            parameter={
                'checkpoint': 'calamari_models/fraktur_historical/*.ckpt.json'
            }
        ).process()
        workspace.save_mets()

        page1 = join(workspace.directory, 'OCR-D-OCR-CALAMARI/OCR-D-OCR-CALAMARI_0001.xml')
        self.assertTrue(exists(page1))
        with open(page1, 'r') as f:
            self.assertIn('verſchuldeten', f.read())

if __name__ == '__main__':
    main()
