import importlib.resources
import os
import pathlib
import shutil
from unittest import TestCase

import pandas as pd
from click.testing import CliRunner

from pyalfe.data_structure import DefaultALFEDataDir, Modality
from pyalfe.main import run


class TestIntegration(TestCase):
    def setUp(self) -> None:
        self.test_dir = os.path.join('/tmp', 'integration_test')

        self.processed_dir = os.path.join(self.test_dir, 'output')
        self.classified_dir = os.path.join(self.test_dir, 'input')

        os.makedirs(self.processed_dir)
        os.mkdir(self.classified_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_run(self):
        accession = 'cnsl_patient'
        modalities = [
            Modality.T1,
            Modality.T2,
            Modality.T1Post,
            Modality.FLAIR,
            Modality.ADC,
        ]
        pipeline_dir = DefaultALFEDataDir(
            output=self.processed_dir, input=self.classified_dir
        )

        for modality in modalities:
            pipeline_dir.create_dir('input', accession, modality)
            shutil.copy(
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    'data',
                    'cnsl_patient',
                    f'{modality}.nii.gz',
                ),
                pipeline_dir.get_input_image(accession, modality),
            )

        runner = CliRunner()
        config_file = importlib.resources.files('pyalfe').joinpath('config.ini')
        targets = [Modality.T1Post, Modality.FLAIR]
        args = [
            accession,
            '-c',
            config_file,
            '--input-dir',
            self.classified_dir,
            '--output-dir',
            self.processed_dir,
            '--targets',
            ','.join(targets),
        ]
        result = runner.invoke(run, args, catch_exceptions=False)
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
                image_type='CNNAbnormalMap_seg',
                sub_dir_name='abnormalmap',
            )
            self.assertTrue(
                os.path.exists(segmentation_path),
                msg=f'{segmentation_path} does not exist.',
            )
            quantification_path = pipeline_dir.get_quantification_file(
                accession, modality, 'SummaryLesionMeasures'
            )
            self.assertTrue(
                os.path.exists(quantification_path),
                msg=f'{quantification_path} does not exist.',
            )
            quantification = pd.read_csv(quantification_path)
            self.assertEqual(quantification.dropna().shape, (14, 2))
