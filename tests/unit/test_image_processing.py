import os
import shutil
from pathlib import Path
from unittest import TestCase

import numpy as np

from pyalfe.image_processing import ImageProcessor, Convert3DProcessor, NilearnProcessor
from tests.utils import create_nifti, get_nifti_data, get_nifti


def get_image_processor_test(image_processor: ImageProcessor) -> type[TestCase]:
    class TestImageProcessor(TestCase):
        def setUp(self) -> None:
            self.test_dir = os.path.join('/tmp', 'image_processing_Test')
            os.makedirs(self.test_dir)

        def tearDown(self) -> None:
            shutil.rmtree(self.test_dir)

        def get_image_path(self, image_name):
            return os.path.join(self.test_dir, image_name)

        def test_threshold(self) -> None:
            data = np.array([[[0, 1, 2, 3]]]).astype(np.int16)
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data)

            threshold1 = self.get_image_path('threshold1.nii.gz')
            image_processor.threshold(Path(input_image_path), threshold1, 0, 1, 0, 1)
            np.testing.assert_array_equal(
                get_nifti_data(threshold1), np.array([[[0, 0, 1, 1]]])
            )

            threshold2 = self.get_image_path('threshold2.nii.gz')
            image_processor.threshold(input_image_path, threshold2, 0, 1, 1, 0)
            np.testing.assert_array_equal(
                get_nifti_data(threshold2), np.array([[[1, 1, 0, 0]]])
            )

            threshold3 = self.get_image_path('threshold3.nii.gz')
            image_processor.threshold(input_image_path, threshold3, 0, 2, 0, 1)
            np.testing.assert_array_equal(
                get_nifti_data(threshold3), np.array([[[0, 0, 0, 1]]])
            )

            threshold4 = self.get_image_path('threshold4.nii.gz')
            image_processor.threshold(input_image_path, threshold4, 0, 3, 0, 1)
            np.testing.assert_array_equal(
                get_nifti_data(threshold4), np.array([[[0, 0, 0, 0]]])
            )

        def test_binarize(self):
            data = np.array([[[0, 1, 2.2, -1, 0]]])
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data)

            binary1 = self.get_image_path('binary1.nii.gz')
            image_processor.binarize(input_image_path, binary1)
            np.testing.assert_array_equal(
                get_nifti_data(binary1), np.array([[[0, 1, 1, 1, 0]]])
            )

        def test_mask(self):
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, np.array([[[0, 1, 2, 3]]]).astype(np.int16))

            mask_image_path = self.get_image_path('mask.nii.gz')
            create_nifti(mask_image_path, np.array([[[0, 1, 1, 0]]]).astype(np.int16))
            output = self.get_image_path('output.nii.gz')
            image_processor.mask(input_image_path, mask_image_path, output)
            np.testing.assert_allclose(
                get_nifti_data(output), np.array([[[0, 1, 2, 0]]]), rtol=1e-03
            )

            mask2_image_path = self.get_image_path('mask2.nii.gz')
            create_nifti(mask2_image_path, np.array([[[0, 0, 0, 0]]]).astype(np.int16))
            output2 = self.get_image_path('output2.nii.gz')
            image_processor.mask(input_image_path, mask2_image_path, output2)
            np.testing.assert_array_equal(
                get_nifti_data(output2), np.array([[[0, 0, 0, 0]]]).astype(np.int16)
            )

            mask3_image_path = self.get_image_path('mask3.nii.gz')
            create_nifti(mask3_image_path, np.array([[[1, 1, 1, 1]]]).astype(np.int16))
            output3 = self.get_image_path('output3.nii.gz')
            image_processor.mask(input_image_path, mask3_image_path, output3)
            np.testing.assert_allclose(
                get_nifti_data(output3), np.array([[[0, 1, 2, 3]]])
            )

        def test_largest_mask_comp(self):
            mask_path = self.get_image_path('input.nii.gz')
            create_nifti(mask_path, np.array([[[0, 1, 1, 0, 1]]]).astype(np.int16))
            output = self.get_image_path('output.nii.gz')
            image_processor.largest_mask_comp(mask_path, output)
            np.testing.assert_array_equal(
                get_nifti_data(output), np.array([[[0, 1, 1, 0, 0]]])
            )

            mask_path2 = self.get_image_path('input.nii.gz')
            create_nifti(mask_path2, np.array([[[0, 0, 0, 0, 0]]]).astype(np.int16))
            output2 = self.get_image_path('output2.nii.gz')
            image_processor.largest_mask_comp(mask_path2, output2)
            np.testing.assert_array_equal(
                get_nifti_data(output2), np.array([[[0, 0, 0, 0, 0]]])
            )

        def test_holefill(self):
            mask_path = self.get_image_path('input.nii.gz')
            data = np.ones([3, 3, 3])
            data[1, 1, 1] = 0
            create_nifti(mask_path, data.astype(np.int16))
            output = self.get_image_path('output.nii.gz')
            image_processor.holefill(mask_path, output)
            np.testing.assert_array_equal(get_nifti_data(output), np.ones([3, 3, 3]))

        def test_reslice_to_ref(self):
            fixed_path = self.get_image_path('fixed.nii.gz')
            fixed_data = np.random.rand(32, 32, 15)
            affine = np.array(
                [
                    [3.0, 0.0, 0.0, -78.0],
                    [0.0, 2.866, -0.887, -76.0],
                    [0.0, 0.887, 2.866, -64.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            create_nifti(fixed_path, fixed_data, affine)

            moving_path = self.get_image_path('moving.nii.gz')
            moving_data = np.random.rand(23, 23, 10)
            create_nifti(moving_path, moving_data)

            output = self.get_image_path('output.nii.gz')

            image_processor.reslice_to_ref(fixed_path, moving_path, output)
            output_nifti = get_nifti(output)
            np.testing.assert_array_equal(
                output_nifti.get_fdata().shape, fixed_data.shape
            )
            np.testing.assert_allclose(output_nifti.affine, affine)

        def test_resample_new_dim(self):
            data = np.concatenate(
                [np.ones((16, 16, 8)), np.zeros((16, 16, 8))], axis=-1
            )
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data)

            output = self.get_image_path('output.nii.gz')
            image_processor.resample_new_dim(input_image_path, output, 100, 100, 200)
            np.testing.assert_array_equal(get_nifti_data(output).shape, (16, 16, 32))

            output2 = self.get_image_path('output2.nii.gz')
            image_processor.resample_new_dim(input_image_path, output2, 100, 50, 100)
            np.testing.assert_array_equal(get_nifti_data(output2).shape, (16, 8, 16))

            output3 = self.get_image_path('output3.nii.gz')
            image_processor.resample_new_dim(input_image_path, output3, 400, 100, 100)
            np.testing.assert_array_equal(get_nifti_data(output3).shape, (64, 16, 16))

            output4 = self.get_image_path('output4.nii.gz')
            image_processor.resample_new_dim(
                input_image_path, output4, 32, 23, 22, percent=False
            )
            np.testing.assert_array_equal(get_nifti_data(output4).shape, (32, 23, 22))

        def test_get_dims(self):
            data = np.random.rand(10, 20, 29)
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data)

            np.testing.assert_array_equal(
                image_processor.get_dims(input_image_path), (10, 20, 29)
            )

        def test_trim_largest_comp(self):
            data = np.array([[[0, 0, 1, 2, 3, 0, 0, 1, 0]]])
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data.astype(np.int16))

            output = self.get_image_path('output.nii.gz')
            image_processor.trim_largest_comp(input_image_path, output, (0, 0, 0))
            np.testing.assert_array_equal(
                get_nifti_data(output), np.array([[[1, 2, 3]]])
            )

            output2 = self.get_image_path('output2.nii.gz')
            image_processor.trim_largest_comp(input_image_path, output2, (0, 0, 1))
            np.testing.assert_array_equal(
                get_nifti_data(output2), np.array([[[0, 1, 2, 3, 0]]])
            )

            output3 = self.get_image_path('output3.nii.gz')
            image_processor.trim_largest_comp(input_image_path, output3, (0, 0, 2))
            np.testing.assert_array_equal(
                get_nifti_data(output3), np.array([[[0, 0, 1, 2, 3, 0, 0]]])
            )

            output4 = self.get_image_path('output4.nii.gz')
            image_processor.trim_largest_comp(input_image_path, output4, (0, 0, 3))
            np.testing.assert_array_equal(
                get_nifti_data(output4), np.array([[[0, 0, 1, 2, 3, 0, 0, 0]]])
            )

        def test_set_subtract(self):
            minuend_data = np.array([[[0, 1, 1, 1, 1, 1]]])
            minuend_path = self.get_image_path('minuend.nii.gz')
            create_nifti(minuend_path, minuend_data.astype(np.int16))

            subtrahend_data = np.array([[[1, 1, 1, 0, 0, 1]]])
            subtrahend_path = self.get_image_path('subtrahend.nii.gz')
            create_nifti(subtrahend_path, subtrahend_data.astype(np.int16))

            output = self.get_image_path('output.nii.gz')
            image_processor.set_subtract(minuend_path, subtrahend_path, output)
            np.testing.assert_array_equal(
                get_nifti_data(output), np.array([[[0, 0, 0, 1, 1, 0]]])
            )

        def test_dilate(self):
            data = np.array([[[0, 1, 1, 0, 0, 0, 1]]])
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data.astype(np.int16))

            output = self.get_image_path('output.nii.gz')

            image_processor.dilate(input_image_path, 1, output)
            np.testing.assert_array_equal(
                get_nifti_data(output), np.array([[[1, 1, 1, 1, 0, 1, 1]]])
            )

            output2 = self.get_image_path('output2.nii.gz')

            image_processor.dilate(input_image_path, 2, output2)
            np.testing.assert_array_equal(
                get_nifti_data(output2), np.array([[[1, 1, 1, 1, 1, 1, 1]]])
            )

        def test_union(self):
            data1 = np.array([[[0, 1, 1, 0, 0, 0, 1]]])
            image1_path = self.get_image_path('input1.nii.gz')
            create_nifti(image1_path, data1.astype(np.int16))

            data2 = np.array([[[1, 0, 1, 0, 0, 1, 0]]])
            image2_path = self.get_image_path('input2.nii.gz')
            create_nifti(image2_path, data2.astype(np.int16))

            output = self.get_image_path('output.nii.gz')

            image_processor.union(image1_path, image2_path, output)
            np.testing.assert_array_equal(
                get_nifti_data(output), np.array([[[1, 1, 1, 0, 0, 1, 1]]])
            )

        def test_distance_transform(self):
            data = np.array([[[0, 1, 1, 0, 0, 0, 1]]])
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data.astype(np.int16))

            output = self.get_image_path('output.nii.gz')
            image_processor.distance_transform(input_image_path, output)
            np.testing.assert_array_equal(
                get_nifti_data(output), np.array([[[1, 0, 0, 1, 2, 1, 0]]])
            )

        def test_label_mask_comp(self):
            data = np.array([[[1, 0, 1, 1, 0, 1, 1, 1, 0]]])
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data.astype(np.int16))

            output = self.get_image_path('output.nii.gz')
            image_processor.label_mask_comp(input_image_path, output)
            np.testing.assert_array_equal(
                get_nifti_data(output), np.array([[[3, 0, 2, 2, 0, 1, 1, 1, 0]]])
            )

        def test_remap_labels(self):
            data = np.array([[[1, 0, 2, 1, 0, 0, 10, 3, 2]]])
            input_image_path = self.get_image_path('input.nii.gz')
            create_nifti(input_image_path, data.astype(np.int16))

            output_1 = self.get_image_path('output_1.nii.gz')
            image_processor.remap_labels(input_image_path, {2: 1, 10: 1}, output_1)
            np.testing.assert_array_equal(
                get_nifti_data(output_1), np.array([[[0, 0, 1, 0, 0, 0, 1, 0, 1]]])
            )

            output_2 = self.get_image_path('output_2.nii.gz')
            image_processor.remap_labels(input_image_path, {2: 2, 10: 2}, output_2)
            np.testing.assert_array_equal(
                get_nifti_data(output_2), np.array([[[0, 0, 2, 0, 0, 0, 2, 0, 2]]])
            )

    return TestImageProcessor


class TestConvert3DProcessor(get_image_processor_test(Convert3DProcessor())):
    pass


class TestNilearnProcessor(get_image_processor_test(NilearnProcessor())):
    pass
