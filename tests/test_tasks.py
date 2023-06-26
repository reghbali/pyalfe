import os
import pathlib
import shutil
from unittest import TestCase

import pandas as pd
import numpy as np

from pyalfe.data_structure import DefaultALFEDataDir, Modality, Tissue
from pyalfe.image_processing import Convert3DProcessor
from pyalfe.image_registration import GreedyRegistration
from pyalfe.inference import InferenceModel
from pyalfe.roi import roi_dict
from pyalfe.tasks.initialization import Initialization
from pyalfe.tasks.quantification import Quantification
from pyalfe.tasks.registration import (
    CrossModalityRegistration,
    Resampling,
    T1Registration,
)
from pyalfe.tasks.segmentation import (
    SingleModalitySegmentation,
    MultiModalitySegmentation,
)
from pyalfe.tasks.skullstripping import Skullstripping
from pyalfe.tasks.t1_postprocessing import T1Postprocessing
from pyalfe.tasks.t1_preprocessing import T1Preprocessing


class MockInferenceModel(InferenceModel):
    def __init__(self, number_of_inputs=1):
        self.number_of_inputs = number_of_inputs

    def predict_cases(self, image_tuple_list, output_list):
        for image_tuple, output in zip(image_tuple_list, output_list):
            assert len(image_tuple) == self.number_of_inputs
            shutil.copy(image_tuple[-1], output)


class TestTask(TestCase):
    """Parent class for all task tests"""

    def setUp(self) -> None:
        self.test_dir = os.path.join('/tmp', 'tasks_tests')

        processed_dir = os.path.join(self.test_dir, 'processed')
        classified_dir = os.path.join(self.test_dir, 'classified')

        os.makedirs(processed_dir)
        os.mkdir(classified_dir)

        self.pipeline_dir = DefaultALFEDataDir(
            processed=processed_dir, classified=classified_dir
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
            classified_image = self.pipeline_dir.get_classified_image(
                accession, modality
            )
            pathlib.Path(classified_image).parent.mkdir(parents=True, exist_ok=True)
            with open(classified_image, 'wb') as _:
                pass

        task.run(accession)

        for modality in modalities:
            modality_image = self.pipeline_dir.get_processed_image(accession, modality)
            self.assertTrue(os.path.exists(modality_image))

    def test_run2(self):
        modalities_existing = [Modality.T1, Modality.T2, Modality.T1Post]
        modalities_missing = [Modality.FLAIR]
        modalities = modalities_missing + modalities_existing
        task = Initialization(self.pipeline_dir, modalities, overwrite=False)

        accession = '12345'
        for modality in modalities_existing:
            classified_image = self.pipeline_dir.get_classified_image(
                accession, modality
            )
            pathlib.Path(classified_image).parent.mkdir(parents=True, exist_ok=True)
            with open(classified_image, 'wb') as _:
                pass

        task.run(accession)

        for modality in modalities_existing:
            modality_image = self.pipeline_dir.get_processed_image(accession, modality)
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
            self.pipeline_dir.create_dir('processed', accession, modality)
            input_image = self.pipeline_dir.get_processed_image(accession, modality)
            shutil.copy(
                os.path.join(
                    'tests', 'data', 'brainomics02', f'anat_{modality.lower()}.nii.gz'
                ),
                input_image,
            )
        task.run(accession)
        for modality in modalities:
            ss_image_path = self.pipeline_dir.get_processed_image(
                accession, modality, image_type='skullstripped'
            )
            self.assertTrue(os.path.exists(ss_image_path))


class TestT1Preprocessing(TestTask):
    def test_run(self):
        accession = 'brainomics02'
        task = T1Preprocessing(Convert3DProcessor, self.pipeline_dir)

        self.pipeline_dir.create_dir('processed', accession, Modality.T1)
        input_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='skullstripped'
        )
        shutil.copy(
            os.path.join('tests', 'data', 'brainomics02', 'anat_t1.nii.gz'), input_image
        )

        task.run(accession)
        output = self.pipeline_dir.get_processed_image(
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
            self.pipeline_dir.create_dir('processed', accession, modality)
            shutil.copy(
                os.path.join(
                    'tests',
                    'data',
                    'brats10',
                    f'BraTS19_2013_10_1_{modality.lower()}.nii.gz',
                ),
                self.pipeline_dir.get_processed_image(
                    accession, modality, task.image_type
                ),
            )
        task.run(accession)
        for target in modalities_target:
            for modality in modalities:

                print(modality, target)
                output = self.pipeline_dir.get_processed_image(
                    accession, modality, f'to_{target}_{task.image_type}'
                )
                self.assertTrue(os.path.exists(output), f'{output} is missing.')


class TestSingleModalitySegmentation(TestTask):
    def test_run(self):
        accession = '10000'
        modality = Modality.FLAIR
        model = MockInferenceModel()
        task = SingleModalitySegmentation(
            model, Convert3DProcessor(), self.pipeline_dir, Modality.FLAIR
        )

        self.pipeline_dir.create_dir('processed', accession, modality)
        input_path = self.pipeline_dir.get_processed_image(
            accession, modality, image_type=task.image_type_input
        )
        output_path = self.pipeline_dir.get_processed_image(
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
            self.pipeline_dir.create_dir('processed', accession, modality)
            if modality != output_modality:
                resampling_target = output_modality
            else:
                resampling_target = None

            input_path = self.pipeline_dir.get_processed_image(
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
        output_path = self.pipeline_dir.get_processed_image(
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
        input_path = self.pipeline_dir.get_processed_image(
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
        tissue_seg_path = self.pipeline_dir.get_processed_image(
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
        output_path = self.pipeline_dir.get_processed_image(
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
            self.pipeline_dir.create_dir('processed', accession, modality)
            shutil.copy(
                os.path.join(
                    'tests',
                    'data',
                    'brats10',
                    f'BraTS19_2013_10_1_{modality.lower()}.nii.gz',
                ),
                self.pipeline_dir.get_processed_image(
                    accession, modality, task.image_type
                ),
            )

        template = roi_dict['template']['source']
        template_reg_sub_dir = roi_dict['template']['sub_dir']

        t1ss = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='skullstripped'
        )

        template_to_t1 = self.pipeline_dir.get_processed_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            sub_dir_name=template_reg_sub_dir,
        )

        affine_transform = self.pipeline_dir.get_processed_image(
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
                self.pipeline_dir.get_processed_image(
                    accession,
                    Modality.T1,
                    image_type=task.image_type,
                    resampling_target=modality,
                    resampling_origin=Modality.T1,
                    extension='.mat',
                ),
            )
        warp_transform = self.pipeline_dir.get_processed_image(
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
                roi_image = self.pipeline_dir.get_processed_image(
                    accession,
                    Modality.T1,
                    image_type=roi_key,
                    sub_dir_name=roi_properties['sub_dir'],
                )
            elif roi_properties['type'] == 'registered':
                roi_image = self.pipeline_dir.get_processed_image(
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
                output_path = self.pipeline_dir.get_processed_image(
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

        self.pipeline_dir.create_dir('processed', accession, Modality.T1)
        input_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='skullstripped'
        )
        shutil.copy(
            os.path.join(
                'tests', 'data', 'brainomics02', 'anat_t1.nii.gz'),
            input_image
        )
        task.run(accession)

        for roi_key in ['template', 'lobes']:
            output_path = self.pipeline_dir.get_processed_image(
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
            Modality.ASL]
        modalities_target = [Modality.T1Post, Modality.FLAIR]

        lesion_seg = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0])
        lesion_seg_comp = np.array([0, 0, 1, 1, 0, 2, 2, 0, 0])
        tissue_seg = np.array([0, 1, 2, 3, 4, 5, 6, 3, 0])
        ventricles_distance = np.array([3, 2, 1, 0, 0, 1, 2, 3, 4])
        modality_images = {
            Modality.T1: np.array([0, 1, 1, 1, 1, 1, 3, 3, 0]),
            Modality.T2: np.array([0, 2, 2, 0, 4, 2, 2, 2, 0]),
            Modality.ADC: np.array([0, 3, 0, 1, 0.5, 2, 2, 1, 1]),
            Modality.T1Post: np.array([0, 1, 2, 2, 0, 5, 3, 2, 1]),
            Modality.FLAIR: np.array([0, 2, 1, 2, 1, 3, 2, 2, 1])}
        template_images = {
            'template': np.array([0, 1, 1, 1, 1, 1, 2, 1, 0]),
            'lobes': np.array([0, 1, 2, 3, 4, 5, 6, 6, 0]),
            'CorpusCallosum': np.array([0, 1, 2, 3, 4, 5, 4, 3, 0, 0])}
        voxel_volume = 2
        task = Quantification(
            pipeline_dir=self.pipeline_dir,
            modalities_all=modalities,
            modalities_target=modalities_target,
            dominant_tissue=Tissue.WHITE_MATTER)
        lesion_stats = task.get_lesion_stats(
            lesion_seg=lesion_seg,
            tissue_seg=tissue_seg,
            ventricles_distance=ventricles_distance,
            modality_images=modality_images,
            template_images=template_images,
            voxel_volume=voxel_volume)
        print(lesion_stats)
        self.assertEqual(lesion_stats['total_lesion_volume'], 8)
        self.assertEqual(lesion_stats['lesion_volume_in_background'], 0)
        self.assertEqual(lesion_stats['lesion_volume_in_csf'], 0)
        self.assertEqual(
            lesion_stats['lesion_volume_in_cortical_gray_matter'], 2)
        self.assertEqual(lesion_stats['lesion_volume_in_white_matter'], 2)
        self.assertEqual(lesion_stats['lesion_volume_in_deep_gray_matter'], 0)
        self.assertEqual(lesion_stats['lesion_volume_in_brain_stem'], 2)
        self.assertEqual(lesion_stats['lesion_volume_in_cerebellum'], 2)
        self.assertEqual(lesion_stats['relative_T1_signal'], 0.5)
        self.assertEqual(lesion_stats['relative_T2_signal'], 0.75)
        self.assertEqual(lesion_stats['relative_ADC_signal'], 1.25)
        self.assertEqual(lesion_stats['mean_adc_signal'], 1.25)
        self.assertEqual(lesion_stats['min_adc_signal'], 0.0)
        self.assertEqual(lesion_stats['median_adc_signal'], 1.5)
        np.testing.assert_almost_equal(
            lesion_stats['five_percentile_adc_signal'], 0.15)
        np.testing.assert_almost_equal(
            lesion_stats['ninety_five_percentile_adc_signal'], 2.0)
        self.assertEqual(
            lesion_stats['relative_T1Post_signal'], 1.5)
        self.assertEqual(
            lesion_stats['relative_FLAIR_signal'], 1.0)
        self.assertEqual(
            lesion_stats['enhancement'], 2.0)
        self.assertEqual(
            lesion_stats['average_dist_to_ventricles_(voxels)'], 1)
        self.assertEqual(
            lesion_stats['minimum_dist_to_Ventricles_(voxels)'], 0)
        self.assertEqual(lesion_stats['lesion_volume_in_Frontal'], 0)
        self.assertEqual(lesion_stats['percentage_volume_in_Frontal'], 0.0)
        self.assertEqual(lesion_stats['lesion_volume_in_Parietal'], 2)
        self.assertEqual(lesion_stats['percentage_volume_in_Parietal'], 25.0)
        self.assertEqual(lesion_stats['lesion_volume_in_Occipital'], 2)
        self.assertEqual(lesion_stats['percentage_volume_in_Occipital'], 25.0)
        self.assertEqual(lesion_stats['lesion_volume_in_Temporal'], 4)
        self.assertEqual(lesion_stats['percentage_volume_in_Temporal'], 50.0)
        self.assertEqual(lesion_stats['lesion_volume_in_CorpusCallosum'], 8)
        self.assertEqual(
            lesion_stats['percentage_volume_in_CorpusCallosum'], 100.0)
