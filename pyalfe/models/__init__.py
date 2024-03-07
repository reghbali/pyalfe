import os
from pathlib import Path

models_url = 'reghbali/pyalfe-models'

MODELS_PATH = Path(os.path.expanduser(os.path.join('~', '.cache', 'pyalfe')))

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
