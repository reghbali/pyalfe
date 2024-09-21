# Installation

PyALFE supports Linux x86-64, Mac x86-64, and Mac arm64 and requires python >= 3.9.


### Option 1: PyPI

To install, run
```bash
pip install pyalfe
```

### Option 2: Development mode
If you want to have the latest development version, you can clone the repo
```bash
git clone https://github.com/reghbali/pyalfe.git
cd pyalfe
```

update the setuptools
```bash
pip install --upgrade setuptools
```

run the following command in the parent pyalfe directory:

```bash
pip install -e .
```

### Option 3: Build and install
Similar to the previous option, you have to first clone the repo

```bash
git clone https://github.com/reghbali/pyalfe.git
cd pyalfe
```

update the build tool
```bash
pip install --upgrade build
```

Run the following commands in the parent pyalfe directory to build the whl file and install pyalfe
```bash
python -m build
pip install dist/pyalfe-0.1.1-py3-none-any.whl
```

### Download models
To download deep learning models, run
```bash
pyalfe download models
```

### Extras
If you want pyalfe to generate pyradiomics features alongside its default features
you can install pyalfe with pyradiomics support. To do so, run:
```bash
pip install 'pyalfe[radiomics]'
```
for development installation
```bash
pip install -e  '.[radiomics]'
```
when performing a build and install
```bash
pip install 'dist/pyalfe-0.1.1-py3-none-any.whl[radiomics]'
```

If you want to use ants registration tool, you can install pyalfe with ants support:
```bash
pip install 'pyalfe[ants]'
```

If you want to build the docs, install pyalfe with docs support:
```bash
pip install 'pyalfe[docs]'
```
