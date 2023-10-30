# Tutorial

In this tutorial, you will download sample data, install pyalfe, and run it on this data.
This tutorial requires `git` and `python>=3.9`.

## Data download and directory setup

1. Download the tutorial data
```bash
git clone https://github.com/reghbali/pyalfe-test-data.git
```
This will create a directory named `pyalfe-test-data` and downloads the tutorial
data inside it. The data is the MRI scan of a glioblastoma patient and is 
taken from the [UPenn-GBM dataset](https://www.nature.com/articles/s41597-022-01560-7).

2. Create an output directory where you have write access
```bash
mkdir -p pyalfe-output
```

## Installation

Create a python virtual environment and activate
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

## Run
```bash
pyalfe run UPENNGBM0000511 --input-dir alfe/input  --output-dir alfe/output
```



