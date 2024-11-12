from multiprocessing import Process
from time import sleep
import gc
import pytest

from ocrd import Resolver, Workspace, OcrdMetsServer
from ocrd_utils import pushd_popd, initLogging, disableLogging, setOverrideLogLevel, getLogger, config

from .assets import assets

CONFIGS = ['', 'metsserver+metscache', 'pageparallel', 'pageparallel+metscache']

@pytest.fixture(params=CONFIGS)
def workspace(tmpdir, pytestconfig, request):
    def _make_workspace(workspace_path):
        initLogging()
        if pytestconfig.getoption('verbose') > 0:
            setOverrideLogLevel('DEBUG')
        with pushd_popd(tmpdir):
            directory = str(tmpdir)
            resolver = Resolver()
            workspace = resolver.workspace_from_url(workspace_path, dst_dir=directory, download=True)
            config.OCRD_MISSING_OUTPUT = "ABORT"
            if 'metscache' in request.param:
                config.OCRD_METS_CACHING = True
                print("enabled METS caching")
            if 'pageparallel' in request.param:
                config.OCRD_MAX_PARALLEL_PAGES = 4
                print("enabled page-parallel processing")
            if 'pageparallel' in request.param or 'metsserver' in request.param:
                def _start_mets_server(*args, **kwargs):
                    print("running with METS server")
                    server = OcrdMetsServer(*args, **kwargs)
                    server.startup()
                process = Process(target=_start_mets_server,
                                  kwargs={'workspace': workspace, 'url': 'mets.sock'})
                process.start()
                sleep(1)
                workspace = Workspace(resolver, directory, mets_server_url='mets.sock')
                yield {'workspace': workspace, 'mets_server_url': 'mets.sock'}
                process.terminate()
                process.join()
            else:
                yield {'workspace': workspace}
        disableLogging()
        config.reset_defaults()
        gc.collect()
    return _make_workspace


@pytest.fixture
def workspace_manifesto(workspace):
    yield from workspace(assets.path_to('communist_manifesto/data/mets.xml'))

@pytest.fixture
def workspace_aufklaerung(workspace):
    yield from workspace(assets.path_to('kant_aufklaerung_1784/data/mets.xml'))

@pytest.fixture
def workspace_aufklaerung_binarized(workspace):
    yield from workspace(assets.path_to('kant_aufklaerung_1784-binarized/data/mets.xml'))

@pytest.fixture
def workspace_aufklaerung_glyph(workspace):
    yield from workspace(assets.path_to('kant_aufklaerung_1784-page-region-line-word_glyph/data/mets.xml'))

@pytest.fixture
def workspace_sbb(workspace):
    yield from workspace(assets.url_of('SBB0000F29300010000/data/mets_one_file.xml'))
