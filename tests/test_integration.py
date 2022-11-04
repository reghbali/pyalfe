import os
import shutil
from unittest import TestCase

from click.testing import CliRunner

from pyalfe.data_structure import DefaultALFEDataDir, Modality
from pyalfe.main import main


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
        args =  [accession,
                 '--classified_dir', self.classified_dir,
                 '--processed_dir', self.processed_dir]
        result = runner.invoke(main, args, catch_exceptions=False)
        print(result)
        self.assertEqual(result.exit_code, 0)

        for modality in modalities:
            processed_image_path = pipeline_dir.get_processed_image(
                accession, modality)
            self.assertTrue(os.path.exists(processed_image_path))
            ss_image_path = self.pipeline_dir.get_processed_image(
                accession, modality, suffix='skullstripped'
            )
            self.assertTrue(os.path.exists(ss_image_path))