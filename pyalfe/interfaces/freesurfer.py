import subprocess


class FreeSurfer:
    def __init__(self):
        self.cmd = []

    def mri_synthseg(self, input, output):
        self.cmd += ['mri_synthseg', '--i', input, '--o', output]
        return self

    def run(self):
        return subprocess.run(self.cmd, capture_output=True).stdout.decode("utf-8")
