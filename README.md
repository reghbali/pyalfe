# PyALFE

Python implementation of Automated Lesion and Feature Extraction (ALFE) pipeline.
We developed this pipeline for analysis of brain MRIs of patients suffering from conditions that cause brain lesions. It utilizes image processing tools, image registration tools, and deep learning segmentation models to produce a set of features that describe the lesion in the brain.

## Requirements

PyALFE supports Linux x86-64, Mac x86-64, and Mac arm64 and requires python 3.9+.

### Image registration and processing
PyALFE can be configured to use either [Greedy](https://greedy.readthedocs.io/en/latest/) or [AntsPy](https://antspy.readthedocs.io/en/latest/registration.html) registration tools.
Similarly, PyALFE can can be configured to use [Convert3D](https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md) or python native library [Nilearn](https://nilearn.github.io/stable/index.html) for image processing tasks.
To use Greedy and Convert3d, these command line tools should be installed on your system.

## Installation

Clone the repo
```bash
git clone https://github.com/reghbali/pyalfe.git
cd pyalfe
```

Then run (we recommend using a python virtual environment)

```bash
pip install --upgrade pip
```

You can either install pyalfe in [development mode](#development-mode-installation) or [build and install](#build-and-install).
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
pip install dist/pyalfe-0.1.0-py3-none-any.whl
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
## Usage

### Configuration
To configrue the PyALFE pipeline you should run:
```bash
pyalfe configure
```
which prompt the you to enter the following required configurations:

#### Input directory
```bash
Enter input image directory: /path/to/my_mri_data
```
The input directory (`input_dir`) contains the images that will be processed by PyALFE and should be organized by accessions (or session ids). Inside the directory for each accession there should be a directory for each available modality.
Here is an example that follows ALFE default structure:
```
my_mri_data
│
│───12345
│   │
│   │───T1
│   │   └── T1.nii.gz
│   │───T1Post
│   │   └── T1Post.nii.gz
│   │───FLAIR
│   │   └── FLAIR.nii.gz
│   │───ADC
│   │   └── ADC.nii.gz
│   │───T2
│   │   └── T2.nii.gz
│   └───CBF
│       └── CBF.nii.gz
└───12356
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
To use this directory the user should provide `path/to/my_mri_data` as the input directory. This config value can be overwritten when calling `pyalfe run` via `-id` or `--input-dir` option.

pyALFE also supports BIDS directories. Here is an example of input dir organized in BIDS format:

```
my_mri_data
│
│───sub-01
│   │───anat
│   │   │───sub-01_T1w.nii.gz
│   │   │───sub-01_ce-gadolinium_T1w.nii.gz
│   │   │───sub-01_T2w.nii.gz
│   │   └───sub-01_FLAIR.nii.gz
│   │───dwi
│   │    │───sub-01_dwi.nii.gz
│   │    └───sub-01_md.nii.gz
│   │───swi
│   │    └───sub-01_swi.nii.gz
│   └───perf
│       └───sub-01_cbf.nii.gz
│
└───sub-02
.   │───anat
.   │   │───sub-02_T1w.nii.gz
.   │   │───sub-02_ce-gadolinium_T1w.nii.gz
    │   │───sub-02_T2w.nii.gz
    │   └───sub-02_FLAIR.nii.gz
    │───dwi
    │    │───sub-02_dwi.nii.gz
    │    └───sub-02_md.nii.gz
    │───swi
    │    └───sub-02_swi.nii.gz
    └───perf
        └───sub-02_cbf.nii.gz

```

#### Output directory
```bash
Enter output image directory: /path/to/output_dir
```
The output image directory (`output_dir`) is where pyALFE writes all its output to.
It can be any valid path in filesystem that user have write access to.
This config value can be overwritten when calling `pyalfe run` via `-od` or `--output-dir` option.


#### Modalities
```bash
Enter modalities separated by comma [T1,T1Post,FLAIR,T2,ADC]: T1,T1Post,ADC
```
All the modalities that should be processed by ALFE.
Modalities should be separated by comma.
To use the default value of `T1,T1Post,T2,FLAIR,ADC`, simply press enter.
This config value can be overwritten when calling `pyalfe run` via `-m` or `--modalities` option.

#### Target modalities
```bash
Enter target modalities separated by comma [T1Post,FLAIR]:
```
The target modalities are used to define the abnormalities which are then used to extract features.
Currently, only `T1Post`, `FLAIR`, or both (default) can be target modality.
This config value can be overwritten when calling `pyalfe run` via `-t` or `--targets` option.

#### Dominant Tissue
```bash
Enter the dominant tissue for the lesions (white_matter, gray_matter, auto) [white_matter]:
```
The dominant tissue where the tumor or lesion is expected to be located at.
This information is use in relative signal feature calculations.
If you choose `auto`, pyalfe automatically detect the dominant tissue after segmentation.
This config value can be overwritten when calling `pyalfe run` via `-dt` or `--dominant_tissue` option.

#### Image processor
```bash
image processor to use (c3d, nilearn) [c3d]:
```
Currently, pyalfe can be configures to use either Convert3D (a.k.a. c3d) or Nilearn for image processing tasks.
The default is Convert3d aka c3d. In other to use c3d,
you have to download it using the [download command](#download-models).
To use Nilearn, you do not need to run any extra command since it is already installed when you install pyalfe.
This config value can be overwritten when calling `pyalfe run` via `-ip` or `--image_processing` option.

#### Image Registration
```bash
image registration to use (greedy, ants) [greedy]:
```
Currently, pyalfe can be configures to use either greedy or ants for image registration tasks. The default is greedy.
In other to use greedy, you have to download it using the [download command](#download-models). To use ants,
install pyalfe with ants support ``pip install pyalfe[ants]``.
This config value can be overwritten when calling `pyalfe run` via `-ir` or `--image-registration` option.

#### Dierctory Data Structure
```bash
data directory structure (press enter for default) (alfe, bids) [alfe]:
```

The directory structure that pyALFE expects in the input directory and will follow when creating the output. See [Inupt directory](#input-directory) for information on ALFE and BIDS.
This config value can be overwritten when calling `payalfe run` via `-dds` or `--data-dir-structure` option.

### Running the pipeline
To run PyALFE for an accession

```bash
pyalfe run ACCESSION
```

If you chose to save the configuration file in a non-standard location you can run

```bash
pyalfe run -c path/to/config.ini ACCESSION
```

In general, all the config option can be overwritten by command line options. To see a list of command line options, run:
```bash
pyalfe run --help
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
BSD 3-Clause
