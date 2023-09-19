from pathlib import Path

import nibabel
import numpy as np


def create_nifti(nifti_path, data, affine=np.eye(4)):
    Path(nifti_path).parent.expanduser().mkdir(parents=True, exist_ok=True)
    image = nibabel.Nifti1Image(data, affine)
    nibabel.save(image, nifti_path)


def get_nifti_data(nifti_path):
    return nibabel.load(nifti_path).get_fdata()


def get_nifti(nifti_path):
    return nibabel.load(nifti_path)
