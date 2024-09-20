import os
import pathlib
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from pyalfe.data_structure import (
    DefaultALFEDataDir,
    PatientDicomDataDir,
    Modality,
    Orientation,
    Tissue,
)
from pyalfe.image_processing import Convert3DProcessor
from pyalfe.image_registration import GreedyRegistration
from pyalfe.inference import InferenceModel
from pyalfe.roi import roi_dict
from pyalfe.tasks.dicom_processing import DicomProcessing
from pyalfe.tasks.initialization import Initialization
from pyalfe.tasks.quantification import Quantification
from pyalfe.tasks.registration import (
    CrossModalityRegistration,
    Resampling,
    T1Registration,
)
from pyalfe.tasks.segmentation import (
    TissueWithPriorSegementation,
    SingleModalitySegmentation,
    MultiModalitySegmentation,
)
from pyalfe.tasks.skullstripping import Skullstripping
from pyalfe.tasks.t1_postprocessing import T1Postprocessing
from pyalfe.tasks.t1_preprocessing import T1Preprocessing
from pyalfe.utils.dicom import ImageMeta
from tests.utils import create_nifti


class MockInferenceModel(InferenceModel):
    def __init__(self, number_of_inputs=1):
        self.number_of_inputs = number_of_inputs

    def predict_cases(self, input_image_tuple_list, output_list):
        for output, input_images in zip(output_list, input_image_tuple_list):
            shutil.copy(input_images[-1], output)


class TestTask(TestCase):
    """Parent class for all task tests"""

    def setUp(self) -> None:
        self.test_dir = os.path.join('/tmp', 'tasks_tests')

        output_dir = os.path.join(self.test_dir, 'output')
        input_dir = os.path.join(self.test_dir, 'input')

        os.makedirs(output_dir)
        os.mkdir(input_dir)

        self.pipeline_dir = DefaultALFEDataDir(
            output_dir=output_dir, input_dir=input_dir
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)


class TestInitialization(TestTask):
    """Test Initialization task"""

    def test_run(self):
        modalities = [Modality.T1, Modality.T2, Modality.T1Post, Modality.FLAIR]
        task = Initialization(self.pipeline_dir, modalities, overwrite=False)

        accession = '12345'
        for modality in modalities:
            classified_image = self.pipeline_dir.get_input_image(accession, modality)
            pathlib.Path(classified_image).parent.mkdir(parents=True, exist_ok=True)
            with open(classified_image, 'wb') as _:
                pass

        task.run(accession)

        for modality in modalities:
            modality_image = self.pipeline_dir.get_output_image(accession, modality)
            self.assertTrue(os.path.exists(modality_image))

    def test_run2(self):
        modalities_existing = [Modality.T1, Modality.T2, Modality.T1Post]
        modalities_missing = [Modality.FLAIR]
        modalities = modalities_missing + modalities_existing
        task = Initialization(self.pipeline_dir, modalities, overwrite=False)

        accession = '12345'
        for modality in modalities_existing:
            classified_image = self.pipeline_dir.get_input_image(accession, modality)
            pathlib.Path(classified_image).parent.mkdir(parents=True, exist_ok=True)
            with open(classified_image, 'wb') as _:
                pass

        task.run(accession)

        for modality in modalities_existing:
            modality_image = self.pipeline_dir.get_output_image(accession, modality)
            self.assertTrue(os.path.exists(modality_image))


class TestSkullstripping(TestTask):
    """Test Skullstripping task"""

    def test_run(self):
        accession = 'brainomics02'
        modalities = [Modality.T1]
        task = Skullstripping(
            MockInferenceModel(), Convert3DProcessor(), self.pipeline_dir, modalities
        )

        for modality in modalities:
            self.pipeline_dir.create_dir('output', accession, modality)
            input_image = self.pipeline_dir.get_output_image(accession, modality)
            shutil.copy(
                os.path.join(
                    'tests', 'data', 'brainomics02', f'anat_{modality.lower()}.nii.gz'
                ),
                input_image,
            )
        task.run(accession)
        for modality in modalities:
            ss_image_path = self.pipeline_dir.get_output_image(
                accession, modality, image_type='skullstripped'
            )
            self.assertTrue(os.path.exists(ss_image_path))


class TestT1Preprocessing(TestTask):
    def test_run(self):
        accession = 'brainomics02'
        task = T1Preprocessing(Convert3DProcessor, self.pipeline_dir)

        self.pipeline_dir.create_dir('output', accession, Modality.T1)
        input_image = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='skullstripped'
        )
        shutil.copy(
            os.path.join('tests', 'data', 'brainomics02', 'anat_t1.nii.gz'), input_image
        )

        task.run(accession)
        output = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='trim_upsampled'
        )
        self.assertTrue(os.path.exists(output))


class TestCrossModalityRegistration(TestTask):
    def test_run(self):
        accession = 'brats10'
        modalities = [Modality.T1, Modality.T2, Modality.T1Post, Modality.FLAIR]
        modalities_target = [Modality.T1Post, Modality.FLAIR]
        task = CrossModalityRegistration(
            GreedyRegistration(), self.pipeline_dir, modalities, modalities_target
        )
        for modality in modalities:
            self.pipeline_dir.create_dir('output', accession, modality)
            shutil.copy(
                os.path.join(
                    'tests',
                    'data',
                    'brats10',
                    f'BraTS19_2013_10_1_{modality.lower()}.nii.gz',
                ),
                self.pipeline_dir.get_output_image(
                    accession, modality, task.image_type
                ),
            )
        task.run(accession)
        for target in modalities_target:
            for modality in modalities:
                print(modality, target)
                output = self.pipeline_dir.get_output_image(
                    accession, modality, f'to_{target}_{task.image_type}'
                )
                self.assertTrue(os.path.exists(output), f'{output} is missing.')


class TestTissueWithPriorSegementation(TestTask):
    def test_run(self):
        accession = '10000'
        model = MockInferenceModel()
        task = TissueWithPriorSegementation(
            model, Convert3DProcessor(), self.pipeline_dir
        )

        self.pipeline_dir.create_dir('output', accession, Modality.T1)
        input_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type=task.image_type_input
        )
        prior_path = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin=task.template_name,
            resampling_target=Modality.T1,
            sub_dir_name=roi_dict[task.template_name]['sub_dir'],
        )
        output_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type=task.image_type_output
        )
        shutil.copy(
            os.path.join('tests', 'data', 'brats10', 'BraTS19_2013_10_1_t1.nii.gz'),
            input_path,
        )
        shutil.copy(
            os.path.join('tests', 'data', 'brats10', 'BraTS19_2013_10_1_t1.nii.gz'),
            prior_path,
        )
        task.run(accession)

        self.assertTrue(os.path.exists(output_path))


class TestSingleModalitySegmentation(TestTask):
    def test_run(self):
        accession = '10000'
        modality = Modality.FLAIR
        model = MockInferenceModel()
        task = SingleModalitySegmentation(
            model, Convert3DProcessor(), self.pipeline_dir, Modality.FLAIR
        )

        self.pipeline_dir.create_dir('output', accession, modality)
        input_path = self.pipeline_dir.get_output_image(
            accession, modality, image_type=task.image_type_input
        )
        output_path = self.pipeline_dir.get_output_image(
            accession,
            modality,
            image_type=task.image_type_output,
            sub_dir_name=task.segmentation_dir,
        )
        shutil.copy(
            os.path.join('tests', 'data', 'brats10', 'BraTS19_2013_10_1_flair.nii.gz'),
            input_path,
        )
        task.run(accession)

        self.assertTrue(os.path.exists(output_path))


class TestMultiModalitySegmentation(TestTask):
    def test_run(self):
        accession = '10000'
        modality_list = [Modality.T1, Modality.T1Post]
        output_modality = Modality.T1Post
        model = MockInferenceModel(2)
        task = MultiModalitySegmentation(
            model,
            Convert3DProcessor(),
            self.pipeline_dir,
            modality_list,
            output_modality,
        )

        for modality in modality_list:
            self.pipeline_dir.create_dir('output', accession, modality)
            if modality != output_modality:
                resampling_target = output_modality
            else:
                resampling_target = None

            input_path = self.pipeline_dir.get_output_image(
                accession,
                modality,
                image_type=task.image_type_input,
                resampling_target=resampling_target,
            )
            shutil.copy(
                os.path.join(
                    'tests',
                    'data',
                    'brats10',
                    f'BraTS19_2013_10_1_{modality.lower()}.nii.gz',
                ),
                input_path,
            )
        output_path = self.pipeline_dir.get_output_image(
            accession,
            output_modality,
            image_type=task.image_type_output,
            sub_dir_name=task.segmentation_dir,
        )
        task.run(accession)
        self.assertTrue(os.path.exists(output_path))


class TestT1Postprocessing(TestTask):
    def test_run(self):
        accession = 'brats10'
        input_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='skullstripped'
        )
        shutil.copy(
            os.path.join(
                'tests',
                'data',
                'brats10',
                f'BraTS19_2013_10_1_{Modality.T1.lower()}.nii.gz',
            ),
            input_path,
        )
        tissue_seg_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='tissue_seg'
        )
        shutil.copy(
            os.path.join(
                'tests',
                'data',
                'brats10',
                f'BraTS19_2013_10_1_{Modality.T1.lower()}.nii.gz',
            ),
            tissue_seg_path,
        )
        output_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='VentriclesSeg'
        )
        task = T1Postprocessing(Convert3DProcessor, self.pipeline_dir)
        task.run(accession)
        self.assertTrue(os.path.exists(output_path))


class TestResampling(TestTask):
    def test_run(self):
        accession = 'brats10'
        modalities = [Modality.T1, Modality.T2, Modality.T1Post, Modality.FLAIR]
        modalities_target = [Modality.T1Post, Modality.FLAIR]

        image_registration = GreedyRegistration()
        task = Resampling(
            Convert3DProcessor, image_registration, self.pipeline_dir, modalities_target
        )

        for modality in modalities:
            self.pipeline_dir.create_dir('output', accession, modality)
            shutil.copy(
                os.path.join(
                    'tests',
                    'data',
                    'brats10',
                    f'BraTS19_2013_10_1_{modality.lower()}.nii.gz',
                ),
                self.pipeline_dir.get_output_image(
                    accession, modality, task.image_type
                ),
            )

        template = roi_dict['template']['source']
        template_reg_sub_dir = roi_dict['template']['sub_dir']

        t1ss = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='skullstripped'
        )

        template_to_t1 = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            sub_dir_name=template_reg_sub_dir,
        )

        affine_transform = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            image_type='affine',
            sub_dir_name=template_reg_sub_dir,
            extension='.mat',
        )

        image_registration.register_affine(t1ss, template, affine_transform, fast=True)

        for modality in modalities_target:
            shutil.copy(
                affine_transform,
                self.pipeline_dir.get_output_image(
                    accession,
                    Modality.T1,
                    image_type=task.image_type,
                    resampling_target=modality,
                    resampling_origin=Modality.T1,
                    extension='.mat',
                ),
            )
        warp_transform = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            image_type='warp',
            sub_dir_name=template_reg_sub_dir,
        )

        image_registration.register_deformable(
            t1ss,
            template,
            transform_output=warp_transform,
            affine_transform=affine_transform,
        )

        image_registration.reslice(
            t1ss, template, template_to_t1, warp_transform, affine_transform
        )

        for roi_key, roi_properties in roi_dict.items():
            if roi_properties['type'] == 'derived':
                roi_image = self.pipeline_dir.get_output_image(
                    accession,
                    Modality.T1,
                    image_type=roi_key,
                    sub_dir_name=roi_properties['sub_dir'],
                )
            elif roi_properties['type'] == 'template':
                roi_image = self.pipeline_dir.get_output_image(
                    accession,
                    Modality.T1,
                    resampling_origin=roi_key,
                    resampling_target=Modality.T1,
                    sub_dir_name=roi_properties['sub_dir'],
                )
            shutil.copy(t1ss, roi_image)

        task.run(accession)

        for modality in modalities_target:
            for roi_key, roi_properties in roi_dict.items():
                if roi_properties['type'] not in ['derived', 'registered']:
                    continue
                output_path = self.pipeline_dir.get_output_image(
                    accession,
                    modality,
                    image_type=roi_key,
                    resampling_origin=modality.T1,
                    resampling_target=modality,
                    sub_dir_name=roi_properties['sub_dir'],
                )
                self.assertTrue(
                    os.path.exists(output_path), msg=f'{output_path} does not exists.'
                )


class TestT1Registration(TestTask):
    def test_run(self):
        accession = 'brainomics02'
        task = T1Registration(
            image_processor=Convert3DProcessor,
            image_registration=GreedyRegistration(),
            pipeline_dir=self.pipeline_dir,
        )

        self.pipeline_dir.create_dir('output', accession, Modality.T1)
        input_image = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='skullstripped'
        )
        input_mask = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='skullstripping_mask'
        )
        input_image_trim_upsampled = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='trim_upsampled'
        )
        shutil.copy(
            os.path.join('tests', 'data', 'brainomics02', 'anat_t1.nii.gz'), input_image
        )
        Convert3DProcessor.binarize(input_image, input_mask)
        shutil.copy(
            os.path.join('tests', 'data', 'brainomics02', 'anat_t1.nii.gz'),
            input_image_trim_upsampled,
        )
        task.run(accession)

        for roi_key in ['template', 'lobes']:
            output_path = self.pipeline_dir.get_output_image(
                accession,
                Modality.T1,
                resampling_origin=roi_key,
                resampling_target=Modality.T1,
                sub_dir_name=roi_dict[roi_key]['sub_dir'],
            )
            self.assertTrue(os.path.exists(output_path))


class TestQuantification(TestTask):
    def test_get_lesion_stats(self):
        modalities = [
            Modality.T1,
            Modality.T2,
            Modality.T1Post,
            Modality.FLAIR,
            Modality.ASL,
        ]
        modalities_target = [Modality.T1Post, Modality.FLAIR]

        lesion_seg = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0])
        tissue_seg = np.array([0, 1, 2, 3, 4, 5, 6, 3, 0])
        ventricles_distance = np.array([3, 2, 1, 0, 0, 1, 2, 3, 4])
        modality_images = {
            Modality.T1: np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 0.0]),
            Modality.T2: np.array([0.0, 2.0, 2.0, 0.0, 4.0, 2.0, 2.0, 2.0, 0.0]),
            Modality.ADC: np.array([0.0, 3.0, 0.0, 1.0, 0.5, 2.0, 2.0, 1.0, 1.0]),
            Modality.T1Post: np.array([0.0, 1.0, 2.0, 2.0, 0.0, 5.0, 3.0, 2.0, 1.0]),
            Modality.FLAIR: np.array([0.0, 2.0, 1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 1.0]),
        }
        template_images = {
            'template': np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0]),
            'lobes': np.array([0, 1, 2, 3, 4, 5, 6, 6, 0]),
            'CorpusCallosum': np.array([0, 1, 2, 3, 4, 5, 4, 3, 0, 0]),
        }
        voxel_volume = 2
        task = Quantification(
            pipeline_dir=self.pipeline_dir,
            modalities_all=modalities,
            modalities_target=modalities_target,
            dominant_tissue=Tissue.WHITE_MATTER,
        )
        lesion_stats = task.get_lesion_stats(
            lesion_seg=lesion_seg,
            tissue_seg=tissue_seg,
            ventricles_distance=ventricles_distance,
            modality_images=modality_images,
            template_images=template_images,
            voxel_volume=voxel_volume,
        )
        self.assertEqual(8.0, lesion_stats['total_lesion_volume'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_background'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_csf'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_cortical_gray_matter'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_white_matter'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_deep_gray_matter'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_brain_stem'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_cerebellum'])
        self.assertEqual(0.5, lesion_stats['relative_T1_signal'])
        self.assertEqual(0.75, lesion_stats['relative_T2_signal'])
        self.assertEqual(1.25, lesion_stats['relative_ADC_signal'])
        self.assertEqual(1.25, lesion_stats['mean_ADC_signal'])
        self.assertEqual(0.0, lesion_stats['min_ADC_signal'])
        self.assertEqual(1.5, lesion_stats['median_ADC_signal'])
        np.testing.assert_almost_equal(0.15, lesion_stats['five_percentile_ADC_signal'])
        np.testing.assert_almost_equal(
            2.0, lesion_stats['ninety_five_percentile_ADC_signal']
        )
        self.assertEqual(1.5, lesion_stats['relative_T1Post_signal'])
        self.assertEqual(1.0, lesion_stats['relative_FLAIR_signal'])
        self.assertEqual(2.0, lesion_stats['enhancement'])
        self.assertEqual(1.0, lesion_stats['average_dist_to_ventricles_(voxels)'])
        self.assertEqual(0.0, lesion_stats['minimum_dist_to_Ventricles_(voxels)'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_Frontal'])
        self.assertEqual(0.0, lesion_stats['percentage_volume_in_Frontal'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_Parietal'], 2)
        self.assertEqual(25.0, lesion_stats['percentage_volume_in_Parietal'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_Occipital'])
        self.assertEqual(25.0, lesion_stats['percentage_volume_in_Occipital'])
        self.assertEqual(4.0, lesion_stats['lesion_volume_in_Temporal'], 4)
        self.assertEqual(50.0, lesion_stats['percentage_volume_in_Temporal'])
        self.assertEqual(8.0, lesion_stats['lesion_volume_in_CorpusCallosum'])
        self.assertEqual(100.0, lesion_stats['percentage_volume_in_CorpusCallosum'])

    def test_get_lesion_stats_with_label(self):
        modalities = [
            Modality.T1,
            Modality.T2,
            Modality.T1Post,
            Modality.FLAIR,
            Modality.ASL,
        ]
        modalities_target = [Modality.T1Post, Modality.FLAIR]

        lesion_seg_comp = np.array([0, 0, 1, 2, 0, 2, 2, 0, 0])
        tissue_seg = np.array([0, 1, 2, 3, 4, 5, 6, 3, 0])
        ventricles_distance = np.array([3, 2, 1, 0, 0, 1, 2, 3, 4])
        modality_images = {
            Modality.T1: np.array([0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 4.0, 0.0]),
            Modality.T2: np.array([0.0, 2.0, 2.0, 0.0, 4.0, 9.0, 0.0, 4.0, 0.0]),
            Modality.ADC: np.array([0.0, 3.0, 0.0, 4.0, 0.5, 6.0, 5.0, 4.0, 1.0]),
            Modality.T1Post: np.array([0.0, 1.0, 2.0, 2.0, 0.0, 5.0, 2.0, 2.0, 1.0]),
            Modality.FLAIR: np.array([0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0]),
        }
        template_images = {
            'template': np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0]),
            'lobes': np.array([0, 1, 2, 3, 4, 5, 6, 6, 0]),
            'CorpusCallosum': np.array([0, 1, 2, 3, 4, 5, 4, 3, 0, 0]),
        }
        voxel_volume = 2
        task = Quantification(
            pipeline_dir=self.pipeline_dir,
            modalities_all=modalities,
            modalities_target=modalities_target,
            dominant_tissue=Tissue.WHITE_MATTER,
        )
        lesion_stats = task.get_lesion_stats(
            lesion_seg=lesion_seg_comp,
            tissue_seg=tissue_seg,
            ventricles_distance=ventricles_distance,
            modality_images=modality_images,
            template_images=template_images,
            voxel_volume=voxel_volume,
            lesion_label=2.0,
        )
        self.assertEqual(6.0, lesion_stats['total_lesion_volume'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_background'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_csf'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_cortical_gray_matter'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_white_matter'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_deep_gray_matter'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_brain_stem'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_cerebellum'])
        self.assertEqual(0.5, lesion_stats['relative_T1_signal'])
        self.assertEqual(0.75, lesion_stats['relative_T2_signal'])
        self.assertEqual(1.25, lesion_stats['relative_ADC_signal'])
        self.assertEqual(5.0, lesion_stats['mean_ADC_signal'])
        self.assertEqual(4.0, lesion_stats['min_ADC_signal'])
        self.assertEqual(5.0, lesion_stats['median_ADC_signal'])
        self.assertEqual(1.5, lesion_stats['relative_T1Post_signal'])
        self.assertEqual(1.0, lesion_stats['relative_FLAIR_signal'])
        self.assertEqual(1.5, lesion_stats['enhancement'])
        self.assertEqual(1.0, lesion_stats['average_dist_to_ventricles_(voxels)'])
        self.assertEqual(0.0, lesion_stats['minimum_dist_to_Ventricles_(voxels)'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_Frontal'])
        self.assertEqual(0.0, lesion_stats['percentage_volume_in_Frontal'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_Parietal'], 2)
        self.assertEqual(0.0, lesion_stats['percentage_volume_in_Parietal'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_Occipital'])
        self.assertEqual(100.0 / 3, lesion_stats['percentage_volume_in_Occipital'])
        self.assertEqual(4.0, lesion_stats['lesion_volume_in_Temporal'], 4)
        self.assertEqual(200 / 3.0, lesion_stats['percentage_volume_in_Temporal'])
        self.assertEqual(6.0, lesion_stats['lesion_volume_in_CorpusCallosum'])
        self.assertEqual(100.0, lesion_stats['percentage_volume_in_CorpusCallosum'])

    def test_get_brain_volume_stats(self):
        brain_seg = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        tissue_seg = np.array([0, 0, 1, 2, 3, 4, 5, 6, 1, 0, 0])
        ventricles_seg = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        template_images = {
            'template': np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0]),
            'lobes': np.array([0, 1, 2, 3, 4, 5, 6, 6, 2, 3]),
            'CorpusCallosum': np.array([0, 1, 2, 3, 4, 5, 4, 3, 0, 0]),
        }

        task = Quantification(
            pipeline_dir=self.pipeline_dir,
            modalities_all=[Modality.T1],
            modalities_target=[Modality.T1Post],
            dominant_tissue=Tissue.WHITE_MATTER,
        )

        volume_stats = task.get_brain_volume_stats(
            brain_seg, tissue_seg, ventricles_seg, template_images, voxel_volume=2.0
        )

        self.assertEqual(14.0, volume_stats['total_brain_volume'])
        self.assertEqual(2.0, volume_stats['total_ventricles_volume'])
        self.assertEqual(8.0, volume_stats['volume_of_background'])
        self.assertEqual(4.0, volume_stats['volume_of_csf'])
        self.assertEqual(2.0, volume_stats['volume_of_cortical_gray_matter'])
        self.assertEqual(2.0, volume_stats['volume_of_white_matter'])
        self.assertEqual(2.0, volume_stats['volume_of_deep_gray_matter'])
        self.assertEqual(2.0, volume_stats['volume_of_brain_stem'])
        self.assertEqual(2.0, volume_stats['volume_of_cerebellum'])
        self.assertEqual(2.0, volume_stats['volume_of_Frontal'])
        self.assertEqual(4.0, volume_stats['volume_of_Parietal'])
        self.assertEqual(4.0, volume_stats['volume_of_Occipital'])
        self.assertEqual(2.0, volume_stats['volume_of_AnteriorTemporal'])
        self.assertEqual(2.0, volume_stats['volume_of_MiddleTemporal'])
        self.assertEqual(4.0, volume_stats['volume_of_PosteriorTemporal'])
        self.assertEqual(8.0, volume_stats['volume_of_Parietal_Occipital'])
        self.assertEqual(14.0, volume_stats['volume_of_CorpusCallosum'])
        self.assertEqual(2.0, volume_stats['volume_of_CorpusCallosum_Rostrum'])
        self.assertEqual(2.0, volume_stats['volume_of_CorpusCallosum_Genu'])
        self.assertEqual(4.0, volume_stats['volume_of_CorpusCallosum_Body'])
        self.assertEqual(4.0, volume_stats['volume_of_CorpusCallosum_Isthmus'])
        self.assertEqual(2.0, volume_stats['volume_of_CorpusCallosum_Splenium'])

    def test_run(self):
        accession = '001'
        modalities = [
            Modality.T1,
            Modality.T2,
            Modality.T1Post,
            Modality.FLAIR,
            Modality.ADC,
        ]
        affine_map = np.eye(4)
        affine_map[0] = 2.0
        modalities_target = [Modality.FLAIR]

        modality_images = {
            Modality.T1: np.array([0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 4.0, 0.0]),
            Modality.T2: np.array([0.0, 2.0, 2.0, 0.0, 4.0, 9.0, 0.0, 4.0, 0.0]),
            Modality.ADC: np.array([0.0, 3.0, 0.0, 4.0, 0.5, 6.0, 5.0, 4.0, 1.0]),
            Modality.T1Post: np.array([0.0, 1.0, 2.0, 2.0, 0.0, 5.0, 2.0, 2.0, 1.0]),
            Modality.FLAIR: np.array([0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0]),
        }

        for modality, modality_image in modality_images.items():
            create_nifti(
                self.pipeline_dir.get_output_image(
                    accession=accession, modality=modality, image_type='skullstripped'
                ),
                modality_image.astype(np.int16),
                affine=affine_map,
            )
            create_nifti(
                self.pipeline_dir.get_output_image(
                    accession=accession,
                    modality=modality,
                    image_type='skullstripped',
                    resampling_target=Modality.FLAIR,
                ),
                modality_image.astype(np.int16),
                affine=affine_map,
            )
            create_nifti(
                self.pipeline_dir.get_output_image(
                    accession=accession,
                    modality=modality,
                    image_type='skullstripping_mask',
                ),
                np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
                affine=affine_map,
            )

        lesion_seg = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0])
        create_nifti(
            self.pipeline_dir.get_output_image(
                accession=accession,
                modality=Modality.FLAIR,
                image_type='abnormal_seg',
                sub_dir_name='abnormalmap',
            ),
            lesion_seg.astype(np.int16),
            affine=affine_map,
        )
        lesion_seg_comp = np.array([0, 0, 1, 2, 0, 2, 2, 0, 0])
        create_nifti(
            self.pipeline_dir.get_output_image(
                accession=accession,
                modality=modality.FLAIR,
                image_type='abnormal_seg_comp',
                sub_dir_name='abnormalmap',
            ),
            lesion_seg_comp.astype(np.int16),
            affine=affine_map,
        )
        tissue_seg = np.array([0, 1, 2, 3, 4, 5, 6, 3, 0])
        create_nifti(
            self.pipeline_dir.get_output_image(
                accession=accession, modality=modality.T1, image_type='tissue_seg'
            ),
            tissue_seg.astype(np.int16),
            affine=affine_map,
        )

        create_nifti(
            self.pipeline_dir.get_output_image(
                accession,
                Modality.FLAIR,
                image_type='tissue_seg',
                resampling_origin='T1',
                resampling_target=Modality.FLAIR,
            ),
            tissue_seg.astype(np.int16),
            affine=affine_map,
        )

        ventricles_seg = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        create_nifti(
            self.pipeline_dir.get_output_image(accession, Modality.T1, 'VentriclesSeg'),
            ventricles_seg.astype(np.int16),
            affine=affine_map,
        )
        ventricles_distance = np.array([3, 2, 1, 0, 0, 1, 2, 3, 4])
        create_nifti(
            self.pipeline_dir.get_output_image(
                accession, Modality.T1, 'VentriclesDist'
            ),
            ventricles_distance.astype(np.int16),
            affine=affine_map,
        )

        create_nifti(
            self.pipeline_dir.get_output_image(
                accession,
                modality=Modality.FLAIR,
                image_type='VentriclesDist',
                resampling_target=Modality.FLAIR,
                resampling_origin=Modality.T1,
            ),
            ventricles_distance.astype(np.int16),
            affine=affine_map,
        )

        template_images = {
            'template': np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0]),
            'lobes': np.array([0, 1, 2, 3, 4, 5, 6, 6, 0]),
            'CorpusCallosum': np.array([0, 1, 2, 3, 4, 5, 4, 3, 0, 0]),
        }

        for template_key, template_image in template_images.items():
            if 'regions' not in roi_dict[template_key]:
                continue
            template_image_to_T1_file = self.pipeline_dir.get_output_image(
                accession,
                modality=Modality.T1,
                resampling_target=Modality.T1,
                resampling_origin=template_key,
                sub_dir_name=roi_dict[template_key]['sub_dir'],
            )
            create_nifti(
                template_image_to_T1_file,
                template_image.astype(np.int16),
                affine=affine_map,
            )
            template_image_to_target_file = self.pipeline_dir.get_output_image(
                accession,
                modality=Modality.FLAIR,
                image_type=template_key,
                resampling_target=Modality.FLAIR,
                resampling_origin=Modality.T1,
                sub_dir_name=roi_dict[template_key]['sub_dir'],
            )
            create_nifti(
                template_image_to_target_file,
                template_image.astype(np.int16),
            )

        dominant_tissue = 'white_matter'
        task = Quantification(
            self.pipeline_dir,
            modalities,
            modalities_target,
            dominant_tissue=dominant_tissue,
        )

        task.run(accession)

        volumetric_quant_file = self.pipeline_dir.get_quantification_file(
            accession, Modality.T1, 'volumeMeasures'
        )
        volume_stats = pd.read_csv(volumetric_quant_file, index_col=0).squeeze(
            "columns"
        )

        self.assertEqual(14.0, volume_stats['total_brain_volume'])
        self.assertEqual(2.0, volume_stats['total_ventricles_volume'])
        self.assertEqual(4.0, volume_stats['volume_of_background'])
        self.assertEqual(2.0, volume_stats['volume_of_csf'])
        self.assertEqual(2.0, volume_stats['volume_of_cortical_gray_matter'])
        self.assertEqual(4.0, volume_stats['volume_of_white_matter'])
        self.assertEqual(2.0, volume_stats['volume_of_deep_gray_matter'])
        self.assertEqual(2.0, volume_stats['volume_of_brain_stem'])
        self.assertEqual(2.0, volume_stats['volume_of_cerebellum'])
        self.assertEqual(2.0, volume_stats['volume_of_Frontal'])
        self.assertEqual(2.0, volume_stats['volume_of_Parietal'])
        self.assertEqual(2.0, volume_stats['volume_of_Occipital'])
        self.assertEqual(2.0, volume_stats['volume_of_AnteriorTemporal'])
        self.assertEqual(2.0, volume_stats['volume_of_MiddleTemporal'])
        self.assertEqual(4.0, volume_stats['volume_of_PosteriorTemporal'])
        self.assertEqual(4.0, volume_stats['volume_of_Parietal_Occipital'])
        self.assertEqual(14.0, volume_stats['volume_of_CorpusCallosum'])
        self.assertEqual(2.0, volume_stats['volume_of_CorpusCallosum_Rostrum'])
        self.assertEqual(2.0, volume_stats['volume_of_CorpusCallosum_Genu'])
        self.assertEqual(4.0, volume_stats['volume_of_CorpusCallosum_Body'])
        self.assertEqual(4.0, volume_stats['volume_of_CorpusCallosum_Isthmus'])
        self.assertEqual(2.0, volume_stats['volume_of_CorpusCallosum_Splenium'])

        summary_quantification_file = self.pipeline_dir.get_quantification_file(
            accession, Modality.FLAIR, 'SummaryLesionMeasures'
        )
        lesion_stats = pd.read_csv(summary_quantification_file, index_col=0).squeeze(
            "columns"
        )

        self.assertEqual(8.0, lesion_stats['total_lesion_volume'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_background'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_csf'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_cortical_gray_matter'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_white_matter'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_deep_gray_matter'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_brain_stem'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_cerebellum'])
        self.assertEqual(0.4375, lesion_stats['relative_T1_signal'])
        self.assertEqual(0.6875, lesion_stats['relative_T2_signal'])
        self.assertEqual(0.9375, lesion_stats['relative_ADC_signal'])
        self.assertEqual(3.75, lesion_stats['mean_ADC_signal'])
        self.assertEqual(0.0, lesion_stats['min_ADC_signal'])
        self.assertEqual(4.5, lesion_stats['median_ADC_signal'])
        np.testing.assert_almost_equal(0.6, lesion_stats['five_percentile_ADC_signal'])
        np.testing.assert_almost_equal(
            lesion_stats['ninety_five_percentile_ADC_signal'], 5.85
        )
        self.assertEqual(1.375, lesion_stats['relative_T1Post_signal'])
        self.assertEqual(0.875, lesion_stats['relative_FLAIR_signal'])
        np.testing.assert_almost_equal(lesion_stats['enhancement'], 1.57, decimal=2)
        self.assertEqual(1.0, lesion_stats['average_dist_to_ventricles_(voxels)'])
        self.assertEqual(0.0, lesion_stats['minimum_dist_to_Ventricles_(voxels)'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_Frontal'])
        self.assertEqual(0.0, lesion_stats['percentage_volume_in_Frontal'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_Parietal'], 2)
        self.assertEqual(25.0, lesion_stats['percentage_volume_in_Parietal'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_Occipital'])
        self.assertEqual(25.0, lesion_stats['percentage_volume_in_Occipital'])
        self.assertEqual(4.0, lesion_stats['lesion_volume_in_Temporal'], 4)
        self.assertEqual(50.0, lesion_stats['percentage_volume_in_Temporal'])
        self.assertEqual(8.0, lesion_stats['lesion_volume_in_CorpusCallosum'])
        self.assertEqual(100.0, lesion_stats['percentage_volume_in_CorpusCallosum'])

        individual_quantification_file = self.pipeline_dir.get_quantification_file(
            accession, Modality.FLAIR, 'IndividualLesionMeasures'
        )
        individual_lesion_stats = pd.read_csv(
            individual_quantification_file, index_col=0
        ).squeeze("columns")
        lesion_stats = individual_lesion_stats.iloc[1]

        self.assertEqual(6.0, lesion_stats['total_lesion_volume'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_background'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_csf'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_cortical_gray_matter'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_white_matter'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_deep_gray_matter'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_brain_stem'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_cerebellum'])
        self.assertEqual(0.5, lesion_stats['relative_T1_signal'])
        self.assertEqual(0.75, lesion_stats['relative_T2_signal'])
        self.assertEqual(1.25, lesion_stats['relative_ADC_signal'])
        self.assertEqual(5.0, lesion_stats['mean_ADC_signal'])
        self.assertEqual(4.0, lesion_stats['min_ADC_signal'])
        self.assertEqual(5.0, lesion_stats['median_ADC_signal'])
        self.assertEqual(1.5, lesion_stats['relative_T1Post_signal'])
        self.assertEqual(1.0, lesion_stats['relative_FLAIR_signal'])
        self.assertEqual(1.5, lesion_stats['enhancement'])
        self.assertEqual(1.0, lesion_stats['average_dist_to_ventricles_(voxels)'])
        self.assertEqual(0.0, lesion_stats['minimum_dist_to_Ventricles_(voxels)'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_Frontal'])
        self.assertEqual(0.0, lesion_stats['percentage_volume_in_Frontal'])
        self.assertEqual(0.0, lesion_stats['lesion_volume_in_Parietal'], 2)
        self.assertEqual(0.0, lesion_stats['percentage_volume_in_Parietal'])
        self.assertEqual(2.0, lesion_stats['lesion_volume_in_Occipital'])
        self.assertEqual(100.0 / 3, lesion_stats['percentage_volume_in_Occipital'])
        self.assertEqual(4.0, lesion_stats['lesion_volume_in_Temporal'], 4)
        self.assertEqual(200 / 3.0, lesion_stats['percentage_volume_in_Temporal'])
        self.assertEqual(6.0, lesion_stats['lesion_volume_in_CorpusCallosum'])
        self.assertEqual(100.0, lesion_stats['percentage_volume_in_CorpusCallosum'])


class TestDicomProcessing(TestTask):
    def setUp(self) -> None:
        super().setUp()
        self.dicom_input_dir = os.path.join(self.test_dir, 'dicom')
        os.mkdir(self.dicom_input_dir)

        self.dicom_data_dir = PatientDicomDataDir(self.dicom_input_dir)

    def test_get_best(self):
        image_1 = ImageMeta(
            path='/dicom/123/456/1-01.dcm',
            series_uid='1.3.6.1.4.1.14519.5.2.1.23720270058',
            manufacturer='SIEMENS',
            seq='SE\\IR ',
            series_desc='T2',
            tr=9420,
            te=141,
            flip_angle=170,
            contrast_agent='9ML MULTIHANCE',
            patient_orientation_vector=[
                0.9999984769134,
                -0.0017453283007,
                0,
                0.00174532830068,
                0.9999984769134,
                0,
            ],
            slice_thickness=None,
            echo_number=1,
            date='2010-05-15',
        )

        self.assertEqual(DicomProcessing.get_best([image_1]), image_1)

        image_2 = ImageMeta(
            path='/dicom/123/456/1-01.dcm',
            series_uid='1.3.6.1.4.1.14519.5.2.1.23720270058',
            manufacturer='SIEMENS',
            seq='SE\\IR ',
            series_desc='T2',
            tr=9420,
            te=141,
            flip_angle=170,
            contrast_agent='9ML MULTIHANCE',
            patient_orientation_vector=[
                0.9999984769134,
                -0.0017453283007,
                0,
                0.00174532830068,
                0.9999984769134,
                0,
            ],
            slice_thickness=1,
            echo_number=1,
            date='2010-05-15',
        )
        self.assertEqual(DicomProcessing.get_best([image_1, image_2]), image_2)

        image_3 = ImageMeta(
            path='/dicom/123/456/1-01.dcm',
            series_uid='1.3.6.1.4.1.14519.5.2.1.23720270058',
            manufacturer='SIEMENS',
            seq='SE\\IR ',
            series_desc='T2',
            tr=9420,
            te=141,
            flip_angle=170,
            contrast_agent='9ML MULTIHANCE',
            patient_orientation_vector=[
                0.9999984769134,
                -0.0017453283007,
                0,
                0.00174532830068,
                0.9999984769134,
                0,
            ],
            slice_thickness=1.5,
            echo_number=1,
            date='2010-05-15',
        )
        self.assertEqual(DicomProcessing.get_best([image_1, image_2, image_3]), image_2)

    def test_select_orientation(self):
        image_axial = ImageMeta(
            path='/dicom/123/456/1-01.dcm',
            series_uid='1.3.6.1.4.1.14519.5.2.1.23720270058',
            manufacturer='SIEMENS',
            seq='SE\\IR ',
            series_desc='T2',
            tr=9420,
            te=141,
            flip_angle=170,
            contrast_agent='9ML MULTIHANCE',
            patient_orientation_vector=[
                0.9999984769134,
                -0.0017453283007,
                0,
                0.00174532830068,
                0.9999984769134,
                0,
            ],
            slice_thickness=1,
            echo_number=1,
            date='2010-05-15',
        )
        orientation_dict = {Orientation.AXIAL: image_axial}
        selected_orientation, selected_image = DicomProcessing.select_orientation(
            orientation_dict
        )
        self.assertEqual(selected_orientation, Orientation.AXIAL)
        self.assertEqual(selected_image, image_axial)

        image_sagital = ImageMeta(
            path='/dicom/123/457/1-01.dcm',
            series_uid='1.3.6.1.4.1.14519.5.2.1.23720270058',
            manufacturer='SIEMENS',
            seq='SE\\IR ',
            series_desc='T2',
            tr=9420,
            te=141,
            flip_angle=170,
            contrast_agent='9ML MULTIHANCE',
            patient_orientation_vector=[
                0,
                0.9999984769134,
                0,
                0.00174532830068,
                0,
                0.9999984769134,
            ],
            slice_thickness=2,
            echo_number=1,
            date='2010-05-15',
        )

        orientation_dict = {
            Orientation.AXIAL: image_axial,
            Orientation.SAGITTAL: image_sagital,
        }
        selected_orientation, selected_image = DicomProcessing.select_orientation(
            orientation_dict
        )
        self.assertEqual(selected_orientation, Orientation.AXIAL)
        self.assertEqual(selected_image, image_axial)

        orientation_dict = {Orientation.SAGITTAL: image_sagital}
        selected_orientation, selected_image = DicomProcessing.select_orientation(
            orientation_dict
        )
        self.assertEqual(selected_orientation, Orientation.SAGITTAL)
        self.assertEqual(selected_image, image_sagital)

        image_coronal = ImageMeta(
            path='/dicom/123/457/1-01.dcm',
            series_uid='1.3.6.1.4.1.14519.5.2.1.23720270058',
            manufacturer='SIEMENS',
            seq='SE\\IR ',
            series_desc='T2',
            tr=9420,
            te=141,
            flip_angle=170,
            contrast_agent='9ML MULTIHANCE',
            patient_orientation_vector=[
                0.9999984769134,
                0,
                0,
                0.00174532830068,
                0,
                0.9999984769134,
            ],
            slice_thickness=0.5,
            echo_number=1,
            date='2010-05-15',
        )

        orientation_dict = {
            Orientation.SAGITTAL: image_sagital,
            Orientation.CORONAL: image_coronal,
        }
        selected_orientation, selected_image = DicomProcessing.select_orientation(
            orientation_dict
        )
        self.assertEqual(selected_orientation, Orientation.CORONAL)
        self.assertEqual(selected_image, image_coronal)

    def test_run(self):
        accession = '05152010'
        shutil.copytree(
            os.path.join('tests', 'data', 'upenn_gbm', 'UPENN-GBM-00621', accession),
            os.path.join(self.dicom_input_dir, accession),
        )
        task = DicomProcessing(
            pipeline_dir=self.pipeline_dir, dicom_dir=self.dicom_data_dir
        )
        task.run(accession)

        for modality in [Modality.T1, Modality.T2, Modality.FLAIR, Modality.T1Post]:
            nifti_image = self.pipeline_dir.get_input_image(accession, modality)
            self.assertTrue(os.path.exists(nifti_image))
