import os
import pathlib
import shutil
from unittest import TestCase

from pyalfe.data_structure import DefaultALFEDataDir, Modality
from pyalfe.image_processing import Convert3DProcessor
from pyalfe.image_registration import GreedyRegistration
from pyalfe.inference import InferenceModel
from pyalfe.tasks.initialization import Initialization
from pyalfe.tasks.registration import CrossModalityRegistration
from pyalfe.tasks.skullstripping import Skullstripping


class MockInferenceModel(InferenceModel):
    def predict_cases(self, image_tuple_list, output_list):
        for image_tuple, output in zip(image_tuple_list, output_list):
            shutil.copy(image_tuple[-1], output)


class TestTask(TestCase):
    """ Parent class for all task tests """

    def setUp(self) -> None:
        self.test_dir = os.path.join('/tmp', 'tasks_tests')

        processed_dir = os.path.join(self.test_dir, 'processed')
        classified_dir = os.path.join(self.test_dir, 'classified')

        os.makedirs(processed_dir)
        os.mkdir(classified_dir)

        self.pipeline_dir = DefaultALFEDataDir(
            processed=processed_dir, classified=classified_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)


class TestInitialization(TestTask):
    """ Test Initialization task """

    def test_run(self):
        modalities = [Modality.T1, Modality.T2, Modality.T1Post, Modality.FLAIR]
        task = Initialization(self.pipeline_dir, modalities, overwrite=False)

        accession = '12345'
        for modality in modalities:
            classified_image = self.pipeline_dir.get_classified_image(
                accession, modality)
            pathlib.Path(classified_image).parent.mkdir(
                parents=True, exist_ok=True)
            with open(classified_image, 'wb') as _:
                pass

        task.run(accession)

        for modality in modalities:
            modality_image = self.pipeline_dir.get_processed_image(
                accession, modality)
            self.assertTrue(os.path.exists(modality_image))

    def test_run2(self):
        modalities_existing = [Modality.T1, Modality.T2, Modality.T1Post]
        modalities_missing = [Modality.FLAIR]
        modalities = modalities_missing + modalities_existing
        task = Initialization(self.pipeline_dir, modalities, overwrite=False)

        accession = '12345'
        for modality in modalities_existing:
            classified_image = self.pipeline_dir.get_classified_image(
                accession, modality)
            pathlib.Path(classified_image).parent.mkdir(
                parents=True, exist_ok=True)
            with open(classified_image, 'wb') as _:
                pass

        task.run(accession)

        for modality in modalities_existing:
            modality_image = self.pipeline_dir.get_processed_image(
                accession, modality)
            self.assertTrue(os.path.exists(modality_image))


class TestSkullstripping(TestTask):
    """ Test Skullstripping task """

    def test_run(self):
        accession = 'brainomics02'
        modalities = [Modality.T1]
        task = Skullstripping(
            MockInferenceModel(),
            Convert3DProcessor(),
            self.pipeline_dir,
            modalities)

        for modality in modalities:
            self.pipeline_dir.create_dir('processed', accession, modality)
            input_image = self.pipeline_dir.get_processed_image(
                accession, modality)
            shutil.copy(
                os.path.join(
                    'tests', 'data',
                    'brainomics02', f'anat_{modality.lower()}.nii.gz'),
                input_image
            )
        task.run(accession)
        for modality in modalities:
            ss_image_path = self.pipeline_dir.get_processed_image(
                accession, modality, image_type='skullstripped'
            )
            self.assertTrue(os.path.exists(ss_image_path))


class TestCrossModalityRegistration(TestTask):

    def test_run(self):
        accession = 'brats10'
        modalities = [
            Modality.T1, Modality.T2, Modality.T1Post, Modality.FLAIR]
        modalities_target = [Modality.T1Post, Modality.FLAIR]
        task = CrossModalityRegistration(
            self.pipeline_dir, modalities,
            modalities_target, GreedyRegistration())
        for modality in modalities:
            self.pipeline_dir.create_dir(
                'processed', accession, modality)
            shutil.copy(
                os.path.join(
                    'tests', 'data', 'brats10',
                    f'BraTS19_2013_10_1_{modality.lower()}.nii.gz'),
                self.pipeline_dir.get_processed_image(
                    accession, modality, task.image_type)
            )
        task.run(accession)
        for target in modalities_target:
            for modality in modalities:
                output = self.pipeline_dir.get_processed_image(
                    accession, modality, f'to_{target}_{task.image_type}')
                self.assertTrue(output)


class TestResampling(TestCase):
    def test_run(self):
        self.fail()
