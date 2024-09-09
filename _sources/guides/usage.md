## Usage

We recommend running the configure command first to create a configuration file for the pipeline.
However, you can skip to [Running the pipeline](#running-the-pipeline) section if you prefer to provide all the required configuration through command line
options.
### Configuration
To configure the PyALFE pipeline, you should run:
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
image processor to use (nilearn, c3d) [nilearn]:
```
Currently, pyalfe can be configured to use either Nilearn, Convert3D (a.k.a. c3d) for image processing tasks.
The default is Nilearn which is a python native library and is installed during installation of pyalfe.
To use c3d, you have to download and install it (https://sourceforge.net/projects/c3d/) on your machine.

This config value can be overwritten when calling `pyalfe run` via `-ip` or `--image_processing` option.

#### Image Registration
```bash
image registration to use (greedy, ants) [greedy]:
```
Currently, pyalfe can be configured to use either greedy or ants for image registration tasks. The default is greedy.
In other to use greedy, you have to download and install greedy (https://sourceforge.net/projects/greedy-reg/). To use ants,
install pyalfe with ants support ``pip install pyalfe[ants]``.

This config value can be overwritten when calling `pyalfe run` via `-ir` or `--image-registration` option.

#### Directory Data Structure
```bash
data directory structure (press enter for default) (alfe, bids) [alfe]:
```

The directory structure that pyALFE expects in the input directory and will follow when creating the output. See [Inupt directory](#input-directory) for information on ALFE and BIDS.
This config value can be overwritten when calling `payalfe run` via `-dds` or `--data-dir-structure` option.

#### Tissue Segmentation
```bash
tissue segmentation method (press enter for default) (prior, synthseg) [prior]:
```

The tissue segmentation method that will be used by pyALFE. The default is based on the method described in this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8346689/), a UNet
that receives a template-based **prior** map tissue alongside the image . pyALFE
also supports [SynthSeg](https://www.sciencedirect.com/science/article/pii/S1361841523000506) via [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg). To use SynthSeg, you have to have FreeSurfer >= v7
installed.

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
