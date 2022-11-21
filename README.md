# PyALFE

Python implementation of Automated Lesion and Feature Extraction (ALFE) pipeline. 

## Requirements
PyALFE supports Linux x86-64, Mac x86-64, and Mac arm64 and requires python >= 3.9.

### Image registration and processing
PyALFE can be configured to use either [Greedy](https://greedy.readthedocs.io/en/latest/) or [AntsPy](https://antspy.readthedocs.io/en/latest/registration.html) registration tools.
Similarly, PyALFE can can be configured to use [Convert3D](https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md) or python native library [Nilearn](https://nilearn.github.io/stable/index.html) for image processing tasks. 
The Greedy and Convert3d command line tools can be downloaded using the [download command](#download-models-and-tools).
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

First update the build tool
```bash
pip install --upgrade build
```

Run the following commands in the parent pyalfe directory to build the whl file and install pyalfe
```bash
python -m build
pip install dist/pyalfe-0.0.1-py3-none-any.whl
```

### Download models and tools
To download deep learning models alongside the binaries for c3d and greedy, run
```bash
pylafe download models c3d greedy
```

## Usage

### Configuration
To configrue the PyALFE pipeline you should run:
```bash
pyalfe configure
```

PyALFE reads its input data from `classified_dir` and writes its output to `processed_dir`. These should be set during configuration. 
The input directory structure in `classfied_dir` should be organized as follows:

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
.   │
.   │───T1
.   │   └── T1.nii.gz
    │───T1Post
    │   └── T1Post.nii.gz
    │───FLAIR
    │   └── FLAIR.nii.gz  
    │───ADC
    │   └── ADC.nii.gz 
    └───T2
        └── T2.nii.gz
```

### Running the pipeline
To run PyALFE for an accession

```bash
pyalfe run ACCESSION
```

If you chose to save the configuration file in a non-standard location you can run

```bash
pyalfe run -c path/to/conf.ini ACCESSION
```

In general, all the config option can be overwritten by command line options. To see a list of command line options, run:
```bash
pyalfe run --help
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Proprietary for now.
