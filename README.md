# PyALFE

Python implementation of Automated Lesion and Feature Extraction (ALFE) pipeline. 

## Requirements
PyALFE supports Linux x86-64, Mac x86-64, and Mac arm64 and requires python 

### Greedy
PyALFE can work with `Greedy` or `Ants` for registration tasks. 

To enable `Greedy`, the [latest version](https://sourceforge.net/projects/greedy-reg/files/latest/download) should be installed on your system and the binary should be on on your path. 


If `Greedy` is not installed on your system, you can install PyALFE with Ants support. Ants registration does not perform as well as greedy.

### Convert3D
For image processing task PyALFE can work with `Convert3D` or python native library `Nilearn`. To enable `Convert3D`, the [latest version](https://sourceforge.net/projects/c3d/files/latest/download) should be installed on your system and the binary should be on on your path. 

## Installation

First run

```bash
pip install --upgrade pip
```

### Development mode installation

First update the setuptools
```bash
pip install --upgrade setuptools
```

Run the following command in the parent pyalfe directory:

```bash
pip install -e .
```

### Build and install

First update the build
```bash
pip install --upgrade build
```

Run the following commands in the parent pyalfe directory to build the whl file and install pyalfe
```bash
python -m build
pip install dist/pyalfe-0.0.1-py3-none-any.whl
```

### Download models & command line tools
To download deep learning models alongside the binaries for c3d and greedy, run
```bash
pylafe download models c3d greedy
```
## Usage

To configrue the PyALFE pipeline you should run:
```bash
pyalfe configure
```
This 
PyALFE reads its input data from `classified_dir` and writes its output to `processed_dir`. These can be given as input argument of configure in a `config.ini` file.

To run PyALFE for an accessionL

```bash
pyalfe run [accession] [--config==config.ini][--classified_dir==path/to/classified] [--processed_dir==path/to/processed] 
```

PyALFE expect the following input directory structure in `classfied_dir`:

```
classfied_dir  
│
│───accession1
│   │
│   │───T1
│   │   └── T1.nii.gz
│   │───T1Post
│   │   └── T1Post.nii.gz
│   │───FLAIR
│   │   └── FLAIR.nii.gz  
│   │───ADC
│   │   └── ADC.nii.gz 
│   └───T2
│       └── T2.nii.gz
│
└───accession2
    │
    │───T1
    │   └── T1.nii.gz
    │───T1Post
    │   └── T1Post.nii.gz
    │───FLAIR
    │   └── FLAIR.nii.gz  
    │───ADC
    │   └── ADC.nii.gz 
    └───T2
        └── T2.nii.gz
  .
  .
  .
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Proprietary for now.
