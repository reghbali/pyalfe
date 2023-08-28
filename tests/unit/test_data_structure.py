import json
import os
import shutil
from unittest import TestCase

from pyalfe.data_structure import (
    DefaultALFEDataDir,
    Modality,
    BIDSDataDir,
)


class TestPipelineDataDir(TestCase):
    def setUp(self) -> None:
        self.test_dir = os.path.join('/tmp', 'image_pipeline_data_dir_Test')
        self.output_dir = os.path.join(self.test_dir, 'output')
        self.input_dir = os.path.join(self.test_dir, 'input')

        os.makedirs(self.output_dir)
        os.mkdir(self.input_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)


class TestDefaultALFEDataDir(TestPipelineDataDir):
    def test_get_output_image(self):
        pipeline_dir = DefaultALFEDataDir(
            output_dir=self.output_dir, input_dir=self.input_dir
        )
        accession = '123'

        for modality in Modality:
            mod_path = pipeline_dir.get_output_image(
                accession=accession, modality=modality
            )
            self.assertEqual(
                os.path.join(
                    self.output_dir,
                    accession,
                    modality,
                    f'{accession}_{modality}.nii.gz',
                ),
                mod_path,
            )
            skullstripped_path = pipeline_dir.get_output_image(
                accession=accession, modality=modality, image_type='skullstripped'
            )
            self.assertEqual(
                os.path.join(
                    self.output_dir,
                    accession,
                    modality,
                    f'{accession}_{modality}_skullstripped.nii.gz',
                ),
                skullstripped_path,
            )
            template_path = pipeline_dir.get_output_image(
                accession=accession,
                modality=modality,
                resampling_origin='template',
                resampling_target=modality,
            )
            self.assertEqual(
                os.path.join(
                    self.output_dir,
                    accession,
                    modality,
                    f'{accession}_template_to_{modality}.nii.gz',
                ),
                template_path,
            )

    def test_get_input_image(self):
        pipeline_dir = DefaultALFEDataDir(
            output_dir=self.output_dir, input_dir=self.input_dir
        )
        accession = '123'

        for modality in Modality:
            mod_path = pipeline_dir.get_input_image(
                accession=accession, modality=modality
            )
            self.assertEqual(
                os.path.join(self.input_dir, accession, modality, f'{modality}.nii.gz'),
                mod_path,
            )

    def test_get_quantification_file(self):
        pipeline_dir = DefaultALFEDataDir(
            output_dir=self.output_dir, input_dir=self.input_dir
        )
        accession = '123'

        for modality in Modality:
            quant_path = pipeline_dir.get_quantification_file(
                accession=accession,
                modality=modality,
                quantification_file_type='SummaryLesionMeasures',
            )
            self.assertEqual(
                os.path.join(
                    self.output_dir,
                    accession,
                    modality,
                    'quantification',
                    f'{accession}_SummaryLesionMeasures.csv',
                ),
                quant_path,
            )


class TestBIDSDataDir(TestPipelineDataDir):
    def setUp(self) -> None:
        super().setUp()
        with open(
            os.path.join(self.input_dir, 'dataset_description.json'), 'w'
        ) as file:
            json.dump({'Name': 'test dataset', 'BIDSVersion': '1.0.2'}, file)

    def test_get_output_image(self):
        pipeline_dir = BIDSDataDir(output_dir=self.output_dir, input_dir=self.input_dir)
        accession = '123'

        mod_path = pipeline_dir.get_output_image(
            accession=accession, modality=Modality.FLAIR
        )
        self.assertEqual(
            os.path.join(
                self.output_dir,
                f'sub-{accession}',
                'anat',
                f'sub-{accession}_FLAIR.nii.gz',
            ),
            mod_path,
        )
        skullstripped_path = pipeline_dir.get_output_image(
            accession=accession, modality=Modality.FLAIR, image_type='skullstripped'
        )
        self.assertEqual(
            os.path.join(
                self.output_dir,
                f'sub-{accession}',
                'anat',
                f'sub-{accession}_desc-skullstripped_FLAIR.nii.gz',
            ),
            skullstripped_path,
        )
        template_path = pipeline_dir.get_output_image(
            accession=accession,
            modality=Modality.FLAIR,
            resampling_origin='template',
            resampling_target=Modality.FLAIR,
        )
        self.assertEqual(
            os.path.join(
                self.output_dir,
                f'sub-{accession}',
                'anat',
                f'sub-{accession}_space-orig_desc-template_{Modality.FLAIR}.nii.gz',
            ),
            template_path,
        )
        template_path = pipeline_dir.get_output_image(
            accession=accession,
            modality=Modality.FLAIR,
            resampling_origin=Modality.ADC,
            resampling_target=Modality.FLAIR,
        )
        self.assertEqual(
            os.path.join(
                self.output_dir,
                f'sub-{accession}',
                'anat',
                f'sub-{accession}_space-orig_desc-ADC_{Modality.FLAIR}.nii.gz',
            ),
            template_path,
        )

    def test_get_input_image(self):
        pipeline_dir = BIDSDataDir(output_dir=self.output_dir, input_dir=self.input_dir)
        accession = '123'

        mod_path = pipeline_dir.get_input_image(
            accession=accession, modality=Modality.T1
        )
        self.assertEqual(
            os.path.join(
                self.input_dir,
                f'sub-{accession}',
                'anat',
                f'sub-{accession}_T1w.nii.gz',
            ),
            mod_path,
        )
        mod_path = pipeline_dir.get_input_image(
            accession=accession, modality=Modality.T2
        )
        self.assertEqual(
            os.path.join(
                self.input_dir,
                f'sub-{accession}',
                'anat',
                f'sub-{accession}_T2w.nii.gz',
            ),
            mod_path,
        )
        mod_path = pipeline_dir.get_input_image(
            accession=accession, modality=Modality.FLAIR
        )
        self.assertEqual(
            os.path.join(
                self.input_dir,
                f'sub-{accession}',
                'anat',
                f'sub-{accession}_FLAIR.nii.gz',
            ),
            mod_path,
        )
        mod_path = pipeline_dir.get_input_image(
            accession=accession, modality=Modality.SWI
        )
        self.assertEqual(
            os.path.join(
                self.input_dir,
                f'sub-{accession}',
                'derivative',
                'swi',
                f'sub-{accession}_swi.nii.gz',
            ),
            mod_path,
        )
        mod_path = pipeline_dir.get_input_image(
            accession=accession, modality=Modality.ADC
        )
        self.assertEqual(
            os.path.join(
                self.input_dir,
                f'sub-{accession}',
                'derivative',
                'dwi',
                f'sub-{accession}_md.nii.gz',
            ),
            mod_path,
        )
        mod_path = pipeline_dir.get_input_image(
            accession=accession, modality=Modality.CBF
        )
        self.assertEqual(
            os.path.join(
                self.input_dir,
                f'sub-{accession}',
                'derivative',
                'perf',
                f'sub-{accession}_cbf.nii.gz',
            ),
            mod_path,
        )

    def test_get_quantification_file(self):
        pipeline_dir = BIDSDataDir(output_dir=self.output_dir, input_dir=self.input_dir)
        accession = '123'

        quant_path = pipeline_dir.get_quantification_file(
            accession=accession,
            modality=Modality.T1,
            quantification_file_type='SummaryLesionMeasures',
        )
        self.assertEqual(
            os.path.join(
                self.output_dir,
                f'sub-{accession}',
                'anat',
                'quantification',
                f'sub-{accession}_desc-T1_SummaryLesionMeasures.csv',
            ),
            quant_path,
        )

        quant_path = pipeline_dir.get_quantification_file(
            accession=accession,
            modality=Modality.T1Post,
            quantification_file_type='SummaryLesionMeasures',
        )
        self.assertEqual(
            os.path.join(
                self.output_dir,
                f'sub-{accession}',
                'anat',
                'quantification',
                f'sub-{accession}_ce-gadolinium_desc-T1Post_SummaryLesionMeasures.csv',
            ),
            quant_path,
        )

        quant_path = pipeline_dir.get_quantification_file(
            accession=accession,
            modality=Modality.FLAIR,
            quantification_file_type='SummaryLesionMeasures',
        )
        self.assertEqual(
            os.path.join(
                self.output_dir,
                f'sub-{accession}',
                'anat',
                'quantification',
                f'sub-{accession}_desc-FLAIR_SummaryLesionMeasures.csv',
            ),
            quant_path,
        )
