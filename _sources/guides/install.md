# Installation

PyALFE supports Linux x86-64, Mac x86-64, and Mac arm64 and requires python >= 3.9.

### Image registration and processing
PyALFE can be configured to use either [Greedy](https://greedy.readthedocs.io/en/latest/) or [AntsPy](https://antspy.readthedocs.io/en/latest/registration.html) registration tools.
Similarly, PyALFE can be configured to use [Convert3D](https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md) or python native library [Nilearn](https://nilearn.github.io/stable/index.html) for image processing tasks.
To use Greedy and Convert3d, these command line tools should be downloaded using the [download command](#download-models-and-tools).


You can either instal pyalfe in [development mode](#development-mode-installation) or [build and install](#build-and-install).
### Option 1: Development mode installation

First update the setuptools
```bash
pip install --upgrade setuptools
```

Run the following command in the parent pyalfe directory:

```bash
pip install -e .
```

### Option 2: Build and install

First update the build tool
```bash
pip install --upgrade build
```

Run the following commands in the parent pyalfe directory to build the whl file and install pyalfe
```bash
python -m build
pip install dist/pyalfe-0.0.1-py3-none-any.whl
```

### Download models
To download deep learning models, run
```bash
pyalfe download models
```
### Pyradiomics support
To install pyalfe with pyradiomics support, run
```bash
pip install -e  '.[radiomics]'
```
for development installation or
```bash
pip install 'dist/pyalfe-0.0.1-py3-none-any.whl[radiomics]'
```
when performing a build and install.

### Ants support
If you want to use Ants registration, you need to install pyalfe with ants support, run
```bash
pip install -e  '.[ants]'
```
for development installation or
```bash
pip install 'dist/pyalfe-0.0.1-py3-none-any.whl[ants]'
```
when performing a build and install.
