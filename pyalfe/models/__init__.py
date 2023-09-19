import os
from pathlib import Path

models_url = 'https://ucsf.box.com/shared/static/a93ruk3m26mso38jm8gk3qs5iod0h90h.gz'

MODELS_PATH = Path(os.path.expanduser(os.path.join('~', '.cache', 'pyalfe')))

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
