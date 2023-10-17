# pylint: disable=unused-import

import os
import sys

from test.assets import assets
from ocrd_utils import initLogging

PWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PWD + "/../ocrd")

initLogging()
