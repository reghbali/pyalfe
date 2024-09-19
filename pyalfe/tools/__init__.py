import os
from collections import defaultdict

greedy_url = defaultdict(
    lambda: defaultdict(str),
    {
        'Linux': defaultdict(
            str,
            {
                'x86_64': (
                    'https://sourceforge.net/projects/greedy-reg/files/Nightly/'
                    'greedy-nightly-Linux-gcc64.tar.gz'
                )
            },
        ),
        'Darwin': defaultdict(
            str,
            {
                'x86_64': (
                    'https://sourceforge.net/projects/greedy-reg/files/Nightly/'
                    'greedy-nightly-MacOS-x86_64.dmg'
                ),
                'arm64': (
                    'https://sourceforge.net/projects/greedy-reg/files/Nightly/'
                    'greedy-nightly-MacOS-arm64.dmg'
                ),
            },
        ),
    },
)[os.uname()[0]][os.uname()[-1]]


c3d_url = defaultdict(
    lambda: defaultdict(str),
    {
        'Linux': defaultdict(
            str,
            {
                'x86_64': (
                    'https://sourceforge.net/projects/c3d/files/c3d/Nightly/'
                    'c3d-nightly-Linux-gcc64.tar.gz'
                )
            },
        ),
        'Darwin': defaultdict(
            str,
            {
                'x86_64': (
                    'https://sourceforge.net/projects/c3d/files/c3d/Nightly/'
                    'c3d-nightly-MacOS-x86_64.dmg'
                ),
                'arm64': (
                    'https://sourceforge.net/projects/c3d/files/c3d/Nightly/'
                    'c3d-nightly-MacOS-arm64.dmg'
                ),
            },
        ),
    },
)[os.uname()[0]][os.uname()[-1]]

GREEDY_PATH = 'greedy'  # importlib.resources.files('pyalfe.tools').joinpath('greedy')
C3D_PATH = 'c3d'  # importlib.resources.files('pyalfe.tools').joinpath('c3d')
