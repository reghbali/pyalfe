# Tutorial

In this tutorial, you will download sample data, install pyalfe, configure it, and run it on this data.
This tutorial requires `git` and `python>=3.9`.

## Data download and directory setup

1. Download the tutorial data
```bash
git clone https://github.com/reghbali/pyalfe-test-data.git
```
This will create a directory named `pyalfe-test-data` and downloads the tutorial
data inside it. The data is the MRI scan of a glioblastoma patient and is
taken from the [UPenn-GBM dataset](https://www.nature.com/articles/s41597-022-01560-7).

2. Create the output directory
```bash
mkdir pyalfe-output
```

## Installation

Create a python virtual environment and activate it
```bash
python3 -m venv venv
source venv/bin/activate
```

Install pyalfe
```bash
pip install pyalfe
```

Download models
```bash
pyalfe download models
```

Configure pyalfe by running
```bash
pyalfe configure
```
and simply pressing enter for all the prompts for default values.

## Run
```bash
pyalfe run UPENNGBM0000511 --input-dir pyalfe-test-data  --output-dir pyalfe-output
```

## Pipeline output
You can now inspect the pipeline output by changing directory to

```bash
cd pyalfe-output/UPENNGBM0000511
```
and exploring the `FLAIR`, `T1`, `T1Post`, `T2`, `ADC` subdirectories.

For instance, the individual FLAIR lesion measures can be found at:
```bash
FLAIR/quantification/UPENNGBM0000511_SummaryLesionMeasures.csv
```

FLAIR lesion segmentation can be found at:
```bash
FLAIR/abnormalmap/UPENNGBM0000511_FLAIR_abnormal_seg.nii.gz
```

Summary T1Post (enhancing) lesion measures can be found at:
```bash
T1Post/quantification/UPENNGBM0000511_IndividualLesionMeasures.csv
```

Brain volumetric measures can be found at:
```bash
T1/quantification/UPENNGBM0000511_volumeMeasures.csv
```

The tissue segmentation can be found at:
```bash
T1/UPENNGBM0000511_T1_tissue_seg.nii.gz
```
