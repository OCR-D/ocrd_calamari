import json
from pkg_resources import resource_string

OCRD_TOOL = json.loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))
TF_CPP_MIN_LOG_LEVEL = '3'  # '3' == ERROR
