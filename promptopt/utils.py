import os

HOME_DIR = os.path.expanduser("~")
PROMPTOPT_DIR = os.path.join(HOME_DIR, "promptopt")
DATA_DIR = os.path.join(PROMPTOPT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

TEMPLATE_DIR = os.path.join(PROMPTOPT_DIR, "templates")
FLASK_PORT = 8000
