from picsl_greedy import Greedy3D


class Greedy:
    def __init__(self):
        self.cmd = []
        self.g = Greedy3D()

    def __repr__(self):
        return ' '.join([str(part) for part in self.cmd])

    def dim(self, d):
        self.cmd += ['-d', str(d)]
        return self

    def threads(self, t):
        self.cmd += ['-threads', str(t)]
        return self

    def affine(self):
        self.cmd.append('-a')
        return self

    def reslice(self, *tran_specs):
        self.cmd += ['-r', *tran_specs]
        return self

    def interpolation(self, mode):
        self.cmd += ['-ri', mode]
        return self

    def initialize_affine(self, affine_transform):
        self.cmd += ['-ia', affine_transform]
        return self

    def transforms(self, *transforms):
        self.cmd += ['-it', *transforms]
        return self

    def epsilon(self, eps):
        self.cmd += ['-e', str(eps)]
        return self

    def reference(self, reference_image):
        self.cmd += ['-rf', reference_image]
        return self

    def input_output(self, input, output):
        self.cmd += ['-rm', input, output]
        return self

    def out(self, transform_output):
        self.cmd += ['-o', transform_output]
        return self

    def dof(self, degrees):
        self.cmd += ['-dof', str(degrees)]
        return self

    def metric(self, metric_name, radius):
        self.cmd += ['-m', metric_name, f'{radius}x{radius}x{radius}']
        return self

    def image_centers(self):
        self.cmd.append('-ia-image-centers')
        return self

    def num_iter(self, low_res, int_res, high_res):
        self.cmd += ['-n', f'{low_res}x{int_res}x{high_res}']
        return self

    def input(self, fixed, moving):
        self.cmd += ['-i', fixed, moving]
        return self

    def run(self):
        return self.g.execute(self.__repr__())
