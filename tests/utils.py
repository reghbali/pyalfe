import os
from pathlib import Path
import requests
import shutil

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


def download_and_extract(url: str, dest_dir: str, archive_name: str = None):
    response = requests.get(url)

    if not archive_name:
        archive_name = url.split('/')[-1]

    archive_file_path = os.path.join(dest_dir, archive_name)
    with open(archive_file_path, 'wb') as file:
        file.write(response.content)

    shutil.unpack_archive(archive_file_path, dest_dir)
