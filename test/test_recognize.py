import os
import shutil
import urllib.request

from test.base import TestCase, main, assets, skip

from ocrd.resolver import Resolver

from ocrd_calamari import CalamariRecognize

METS_KANT = assets.url_of('kant_aufklaerung_1784-page-block-line-word_glyph/data/mets.xml')

WORKSPACE_DIR = '/tmp/test-ocrd-calamari'

class TestCalamariRecognize(TestCase):

    def setUp(self):
        if os.path.exists(WORKSPACE_DIR):
            shutil.rmtree(WORKSPACE_DIR)
        os.makedirs(WORKSPACE_DIR)

    def runTest(self):
        resolver = Resolver()
        workspace = resolver.workspace_from_url(METS_KANT, dst_dir=WORKSPACE_DIR)

        # XXX Work around data bug(?):
        #     PAGE-XML links to OCR-D-IMG/INPUT_0017.tif, but this is nothing core can download
        os.makedirs(os.path.join(WORKSPACE_DIR, 'OCR-D-IMG'))
        for f in ['INPUT_0017.tif', 'INPUT_0020.tif']:
            urllib.request.urlretrieve(
                    "https://github.com/OCR-D/assets/raw/master/data/kant_aufklaerung_1784/data/OCR-D-IMG/" + f,
                    os.path.join(WORKSPACE_DIR, 'OCR-D-IMG', f))

        # XXX Should remove GT text to really test this

        CalamariRecognize(
            workspace,
            input_file_grp="OCR-D-GT-SEG-LINE",
            output_file_grp="OCR-D-OCR-CALAMARI",
            parameter={
                'checkpoint': os.path.join(os.getcwd(), 'calamari_models/fraktur_19th_century/*.ckpt.json')
            }
        ).process()
        workspace.save_mets()

        page1 = os.path.join(workspace.directory, 'OCR-D-OCR-CALAMARI/OCR-D-OCR-CALAMARI_0001.xml')
        self.assertTrue(os.path.exists(page1))
        with open(page1, 'r', encoding='utf-8') as f:
            self.assertIn('verſchuldeten', f.read())

if __name__ == '__main__':
    main()
