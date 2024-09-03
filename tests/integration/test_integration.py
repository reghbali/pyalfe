import importlib.resources
import os
import shutil
from unittest import TestCase

import pandas as pd
from click.testing import CliRunner

from pyalfe.data_structure import DefaultALFEDataDir, Modality
from pyalfe.main import run_command
from tests.utils import download_and_extract


class TestIntegration(TestCase):
    def setUp(self) -> None:
        self.test_dir = os.path.join('/tmp', 'integration_test')

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_run(self):

        test_data_url = (
            'https://github.com/reghbali/pyalfe-test-data/archive/master.zip'
        )
        test_data_dir_name = 'pyalfe-test-data-main'
        accession = 'UPENNGBM0000511'

        output_dir = os.path.join(self.test_dir, 'output')
        input_dir = os.path.join(self.test_dir, test_data_dir_name)
        os.makedirs(output_dir)
        os.mkdir(input_dir)

        modalities = [
            Modality.T1,
            Modality.T2,
            Modality.T1Post,
            Modality.FLAIR,
            Modality.ADC,
        ]
        targets = [Modality.T1Post, Modality.FLAIR]
        pipeline_dir = DefaultALFEDataDir(output_dir=output_dir, input_dir=input_dir)

        download_and_extract(test_data_url, self.test_dir)

        runner = CliRunner()
        config_file = importlib.resources.files('pyalfe').joinpath('config.ini')
        args = [
            accession,
            '-c',
            config_file,
            '--input-dir',
            input_dir,
            '--output-dir',
            output_dir,
            '--targets',
            ','.join(targets),
        ]
        result = runner.invoke(run_command, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0, msg=result.stdout)

        for modality in modalities:
            processed_image_path = pipeline_dir.get_output_image(accession, modality)
            self.assertTrue(os.path.exists(processed_image_path))
            image_path = pipeline_dir.get_output_image(accession, modality)
            ss_image_path = pipeline_dir.get_output_image(
                accession, modality, image_type='skullstripped'
            )
            self.assertTrue(
                os.path.exists(ss_image_path), msg=f'{ss_image_path} does not exist.'
            )
            self.assertTrue(
                os.path.exists(image_path), msg=f'{image_path} does not exist.'
            )
        for modality in targets:
            segmentation_path = pipeline_dir.get_output_image(
                accession,
                modality,
                image_type='abnormal_seg',
                sub_dir_name='abnormalmap',
            )
            self.assertTrue(
                os.path.exists(segmentation_path),
                msg=f'{segmentation_path} does not exist.',
            )
            summary_quantification_path = pipeline_dir.get_quantification_file(
                accession, modality, 'SummaryLesionMeasures'
            )
            individual_quantification_path = pipeline_dir.get_quantification_file(
                accession, modality, 'IndividualLesionMeasures'
            )

            self.assertTrue(
                os.path.exists(summary_quantification_path),
                msg=f'{summary_quantification_path} does not exist.',
            )
            summary_quantification = pd.read_csv(summary_quantification_path)
            self.assertEqual(summary_quantification.dropna().shape, (65, 2))

            individual_quantification = pd.read_csv(individual_quantification_path)
            self.assertEqual(individual_quantification.dropna().shape[1], 63)
