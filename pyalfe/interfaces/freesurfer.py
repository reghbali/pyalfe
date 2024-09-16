import subprocess
from shutil import which


class FreeSurfer:
    def __init__(self):
        self.cmd = []

    def mri_synthseg(self, input, output):
        synthseg_cmd = 'mri_synthseg'
        if which(synthseg_cmd) is None:
            raise RuntimeError(
                f'{synthseg_cmd} executable was not found in your system. '
                'To download and install FreeSurfer version 7, visit:\n '
                'https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall \n'
                'To configure FreeSurfer, visit:\n'
                'https://surfer.nmr.mgh.harvard.edu/fswiki/SetupConfiguration_Linux \n'
                'https://surfer.nmr.mgh.harvard.edu/fswiki/SetupConfiguration_Mac'
            )
        self.cmd += [synthseg_cmd, '--i', input, '--o', output]
        return self

    def run(self):
        return subprocess.run(self.cmd, capture_output=True).stdout.decode("utf-8")
