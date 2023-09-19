import os
import shutil
from unittest import TestCase

import numpy as np
import scipy

from pyalfe.image_registration import (
    ImageRegistration,
    GreedyRegistration,
    AntsRegistration,
)
from tests.utils import create_nifti, get_nifti_data


def get_image_registration_test(
    image_registration: ImageRegistration,
) -> type[TestCase]:
    class TestImageRegistration(TestCase):
        def setUp(self) -> None:
            self.test_dir = os.path.join('/tmp', 'image_registration_Test')
            os.makedirs(self.test_dir)

        def tearDown(self) -> None:
            shutil.rmtree(self.test_dir)

        def get_image_path(self, image_name):
            return os.path.join(self.test_dir, image_name)

        def test_register_rigid(self):
            fixed_path = self.get_image_path('fixed.nii.gz')
            length = 33
            c = length // 2
            fixed_data = np.zeros((length, length, length))
            fixed_data[c - 2 : c + 3, c - 6 : c + 7, c - 11 : c + 12] = 1
            fixed_data[c - 2 : c + 3, c - 2 : c + 3, c - 4 : c + 5] = 0
            create_nifti(fixed_path, fixed_data)
            moving_path = self.get_image_path('moving.nii.gz')
            moving_data = scipy.ndimage.shift(fixed_data, (2, 0, 0))
            create_nifti(moving_path, moving_data)

            transform = self.get_image_path('rigid.mat')
            image_registration.register_rigid(fixed_path, moving_path, transform)

            output = self.get_image_path('output.nii.gz')
            image_registration.reslice(fixed_path, moving_path, output, transform)

            np.testing.assert_array_less(
                np.linalg.norm(get_nifti_data(output) - fixed_data)
                / np.linalg.norm(fixed_data),
                0.15,
            )
            moving2_path = self.get_image_path('moving2.nii.gz')
            moving2_data = scipy.ndimage.rotate(
                fixed_data, 15, axes=(2, 0), reshape=False
            )

            create_nifti(moving2_path, moving2_data)

            transform2 = self.get_image_path('rigid2.mat')
            image_registration.register_rigid(fixed_path, moving2_path, transform2)

            output2 = self.get_image_path('output2.nii.gz')
            image_registration.reslice(fixed_path, moving2_path, output2, transform2)

            np.testing.assert_array_less(
                np.linalg.norm(get_nifti_data(output2) - fixed_data)
                / np.linalg.norm(get_nifti_data(output2) - moving2_data),
                0.25,
            )

        def test_register_affine(self):
            fixed_path = self.get_image_path('fixed.nii.gz')
            length = 33
            c = length // 2
            fixed_data = np.zeros((length, length, length))
            fixed_data[c - 2 : c + 3, c - 6 : c + 7, c - 11 : c + 12] = 1
            fixed_data[c - 2 : c + 3, c - 2 : c + 3, c - 4 : c + 5] = 0
            create_nifti(fixed_path, fixed_data)

            moving_path = self.get_image_path('moving.nii.gz')
            moving_data = scipy.ndimage.shift(fixed_data, (2, 0, 0))
            create_nifti(moving_path, moving_data)

            transform = self.get_image_path('affine.mat')
            image_registration.register_affine(
                fixed_path, moving_path, transform, fast=False
            )

            output = self.get_image_path('output.nii.gz')
            image_registration.reslice(fixed_path, moving_path, output, transform)

            np.testing.assert_array_less(
                np.linalg.norm(get_nifti_data(output) - fixed_data)
                / np.linalg.norm(get_nifti_data(output) - moving_data),
                0.47,
            )

            moving2_path = self.get_image_path('moving2.nii.gz')
            moving2_data = scipy.ndimage.rotate(
                fixed_data, 25, axes=(2, 0), reshape=False
            )
            create_nifti(moving2_path, moving2_data)

            transform2 = self.get_image_path('affine2.mat')
            image_registration.register_affine(
                fixed_path, moving2_path, transform2, fast=False
            )

            output2 = self.get_image_path('output2.nii.gz')
            image_registration.reslice(fixed_path, moving2_path, output2, transform2)

            np.testing.assert_array_less(
                np.linalg.norm(get_nifti_data(output2) - fixed_data)
                / np.linalg.norm(get_nifti_data(output2) - moving2_data),
                0.30,
            )

            moving3_path = self.get_image_path('moving3.nii.gz')
            moving3_data = scipy.ndimage.zoom(fixed_data, 1.2)
            create_nifti(moving3_path, moving3_data)

            transform3 = self.get_image_path('affine3.mat')
            image_registration.register_affine(
                fixed_path, moving3_path, transform3, fast=False
            )

            output3 = self.get_image_path('output3.nii.gz')
            image_registration.reslice(fixed_path, moving3_path, output3, transform3)

            np.testing.assert_array_less(
                np.linalg.norm(get_nifti_data(output3) - fixed_data)
                / np.linalg.norm(get_nifti_data(output3) - moving2_data),
                0.55,
            )

        def test_register_deformable(self):
            fixed_path = self.get_image_path('fixed.nii.gz')
            length = 37
            c = length // 2
            fixed_data = np.zeros((length, length, length))
            fixed_data[c - 2 : c + 3, c - 6 : c + 7, c - 11 : c + 12] = 1
            create_nifti(fixed_path, fixed_data)

            moving_path = self.get_image_path('moving.nii.gz')
            moving_data = np.zeros((length, length, length))
            moving_data[c - 2 : c + 3, c - 7 : c + 8, c - 10 : c + 11] = 1
            create_nifti(moving_path, moving_data)

            affine_transform = self.get_image_path('affine.mat')
            warp_transform = self.get_image_path('warp.nii.gz')
            image_registration.register_deformable(
                fixed_path,
                moving_path,
                transform_output=warp_transform,
                affine_transform=affine_transform,
            )

            output = self.get_image_path('output.nii.gz')
            image_registration.reslice(
                fixed_path, moving_path, output, warp_transform, affine_transform
            )

            np.testing.assert_array_less(
                np.linalg.norm(get_nifti_data(output) - fixed_data)
                / np.linalg.norm(get_nifti_data(output) - moving_data),
                0.5,
            )

    return TestImageRegistration


class TestGreedyRegistration(get_image_registration_test(GreedyRegistration())):
    pass


class TestAntsRegistration(get_image_registration_test(AntsRegistration())):
    pass
