import importlib.resources
import os
import shutil
from unittest import TestCase

import pandas as pd
from click.testing import CliRunner

from pyalfe.data_structure import DefaultALFEDataDir, Modality
from pyalfe.main import main, run


class TestIntegration(TestCase):

    def setUp(self) -> None:
        self.test_dir = os.path.join('/tmp', 'integration_test')

        self.processed_dir = os.path.join(self.test_dir, 'processed')
        self.classified_dir = os.path.join(self.test_dir, 'classified')

        os.makedirs(self.processed_dir)
        os.mkdir(self.classified_dir)

    def tearDown(self) -> None:
        pass
        #shutil.rmtree(self.test_dir)

    def test_run(self):
        accession = 'brats10'
        modalities = [
                Modality.T1, Modality.T2, Modality.T1Post, Modality.FLAIR]
        pipeline_dir = DefaultALFEDataDir(
            processed=self.processed_dir, classified=self.classified_dir)
        for modality in modalities:
            pipeline_dir.create_dir(
                'classified', accession, modality)
            shutil.copy(
                os.path.join(
                    'tests', 'data', 'brats10',
                    f'BraTS19_2013_10_1_{modality.lower()}.nii.gz'),
                pipeline_dir.get_classified_image(accession, modality)
            )
        runner = CliRunner()
        config_file = importlib.resources.files('pyalfe').joinpath('config.ini')
        targets = [Modality.T1Post, Modality.FLAIR]
        args =  [accession,
                 '-c', config_file,
                 '--classified_dir', self.classified_dir,
                 '--processed_dir', self.processed_dir,
                 '--targets', ','.join(targets)]
        result = runner.invoke(run, args, catch_exceptions=False)
        print(result)
        self.assertEqual(result.exit_code, 0)

        for modality in modalities:
            processed_image_path = pipeline_dir.get_processed_image(
                accession, modality)
            self.assertTrue(os.path.exists(processed_image_path))
            image_path = pipeline_dir.get_processed_image(
                accession, modality)
            ss_image_path = pipeline_dir.get_processed_image(
                accession, modality, image_type='skullstripped'
            )
            self.assertTrue(
                os.path.exists(ss_image_path),
                msg=f'{ss_image_path} does not exist.')
            self.assertTrue(
                os.path.exists(image_path),
                msg=f'{image_path} does not exist.')

        for modality in targets:
            quantification_path = pipeline_dir.get_quantification_file(
                accession, modality, 'SummaryLesionMeasures')
            self.assertTrue(
                os.path.exists(quantification_path),
                msg=f'{quantification_path} does not exist.')
            quantification = pd.read_csv(quantification_path)
            self.assertEqual(quantification.dropna().shape, (14, 2))
