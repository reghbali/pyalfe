import subprocess

from pyalfe.tools import C3D_PATH


class C3D:
    def __init__(self, c3d_path=C3D_PATH):
        self.cmd = [c3d_path]

    def push(self, var):
        self.cmd.append('-push')
        self.cmd.append(var)
        return self

    def popas(self, var):
        self.cmd.append('-popas')
        self.cmd.append(var)
        return self

    def thresh(self, u1, u2, vin, vout):
        self.cmd += ['-thresh', str(u1), str(u2), str(vin), str(vout)]
        return self

    def clip(self, imin, imax):
        self.cmd += ['-clip', str(imin), str(imax)]
        return self

    def comp(self):
        self.cmd.append('-comp')
        return self

    def dup(self):
        self.cmd.append('-dup')
        return self

    def scale(self, factor):
        self.cmd += ['-scale', str(factor)]
        return self

    def binarize(self):
        self.cmd.append('-binarize')
        return self

    def multiply(self):
        self.cmd.append('-multiply')
        return self

    def reslice_identity(self):
        self.cmd.append('-reslice-identity')
        return self

    def resample(self):
        self.cmd.append('-resample')
        return self

    def trim(self, m1, m2, m3):
        self.cmd += ['-trim', f'{m1}x{m2}x{m3}mm']
        return self

    def dilate(self, label, r1, r2, r3):
        self.cmd += ['-dilate', str(label), f'{r1}x{r2}x{r3}vox']
        return self

    def holefill(self, v, c):
        self.cmd += ['-holefill', str(v), str(c)]
        return self

    def info(self):
        self.cmd.append('-info')
        return self

    def out(self, output):
        self.cmd += ['-o', output]
        return self

    def operand(self, *op):
        self.cmd += op
        return self

    def add(self):
        self.cmd.append('-add')
        return self

    def sdt(self):
        self.cmd.append('-sdt')
        return self

    def run(self):
        return subprocess.run(self.cmd, capture_output=True).stdout.decode("utf-8")

    def check_output(self):
        return subprocess.check_output(self.cmd).decode("utf-8")
