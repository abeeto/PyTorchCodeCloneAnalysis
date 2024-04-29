import logging
from pytest import fixture
import seedbank

_log = logging.getLogger('lenskit.tests')
logging.getLogger('numba').setLevel(logging.INFO)


@fixture(autouse=True)
def init_rng(request):
    seedbank.initialize(42)


@fixture(autouse=True)
def log_test(request):
    modname = request.module.__name__ if request.module else '<unknown>'
    funcname = request.function.__name__ if request.function else '<unknown>'
    _log.info('running test %s:%s', modname, funcname)
