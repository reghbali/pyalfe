import os
from configparser import ConfigParser
from functools import cached_property
from types import SimpleNamespace

from pyalfe.data_structure import (
    DefaultALFEDataDir,
    BIDSDataDir,
    PatientDicomDataDir,
    Modality,
)
from pyalfe.image_processing import Convert3DProcessor, NilearnProcessor
from pyalfe.image_registration import GreedyRegistration, AntsRegistration
from pyalfe.inference import NNUnetV2, SynthSeg
from pyalfe.models import MODELS_PATH
from pyalfe.pipeline import PyALFEPipelineRunner, DicomProcessingPipelineRunner
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
    SynthSegTissueSegmentation,
)
from pyalfe.tasks.skullstripping import Skullstripping
from pyalfe.tasks.t1_postprocessing import T1Postprocessing
from pyalfe.tasks.t1_preprocessing import T1Preprocessing


class Config:
    def from_ini(self, ini_file):
        if not os.path.exists(ini_file):
            raise FileNotFoundError(f'config file {ini_file} does not exist.')
        config_parser = ConfigParser()
        config_parser.read(ini_file)
        self.from_dict(
            {
                section: dict(config_parser.items(section))
                for section in config_parser.sections()
            }
        )

    def from_dict(self, d):
        for section, key_value in d.items():
            if hasattr(self, section):
                current = getattr(self, section)
                for key, value in key_value.items():
                    current.key = value
            else:
                setattr(self, section, SimpleNamespace(**key_value))


class DeclarativeContainer:
    def __init__(self):
        self.config = Config()

    def init_resources(self):
        """This function exists for compatibility reasons"""
        pass


class PipelineContainer(DeclarativeContainer):
    """
    container objects for all the dependencies of the pipeline.
    """

    @cached_property
    def pipeline_dir(self):
        if self.config.options.data_dir_structure == 'alfe':
            return DefaultALFEDataDir(
                output_dir=self.config.options.output_dir,
                input_dir=self.config.options.input_dir,
            )
        elif self.config.options.data_dir_structure == 'bids':
            return BIDSDataDir(
                output_dir=self.config.options.output_dir,
                input_dir=self.config.options.input_dir,
            )
        else:
            raise ValueError(
                f'Invalid data dir structure {self.config.options.data_dir_structure}'
            )

    @cached_property
    def image_processor(self):
        if self.config.options.image_processor == 'c3d':
            return Convert3DProcessor()
        elif self.config.options.image_processor == 'nilearn':
            return NilearnProcessor()
        else:
            raise ValueError(
                f'Invalid image processor {self.config.options.image_processor}'
            )

    @property
    def image_registration(self):
        if self.config.options.image_registration == 'greedy':
            return GreedyRegistration()
        elif self.config.options.image_registration == 'ants':
            return AntsRegistration()
        else:
            raise ValueError(
                f'Invalid image registration {self.config.options.image_registration}'
            )

    @cached_property
    def skullstripping_model(self):
        return NNUnetV2(
            model_dir=str(
                MODELS_PATH.joinpath(
                    'nnunetv2',
                    'Dataset502_SS',
                    'nnUNetTrainer__nnUNetPlans__3d_fullres',
                )
            ),
            folds=(2,),
        )

    @cached_property
    def flair_model(self):
        return NNUnetV2(
            model_dir=str(
                MODELS_PATH.joinpath(
                    'nnunetv2',
                    'Dataset500_FLAIR',
                    'nnUNetTrainer__nnUNetPlans__3d_fullres',
                )
            ),
            folds=(4,),
        )

    @cached_property
    def enhancement_model(self):
        return NNUnetV2(
            model_dir=str(
                MODELS_PATH.joinpath(
                    'nnunetv2',
                    'Dataset503_Enhancement',
                    'nnUNetTrainer__nnUNetPlans__3d_fullres',
                )
            ),
            folds=(0,),
        )

    @cached_property
    def tissue_model(self):
        if self.config.options.tissue_segmentation == 'prior':
            return NNUnetV2(
                model_dir=str(
                    MODELS_PATH.joinpath(
                        'nnunetv2',
                        'Dataset510_Tissue_W_Prior',
                        'nnUNetTrainer__nnUNetPlans__3d_fullres',
                    )
                ),
                folds=(3,),
            )
        elif self.config.options.tissue_segmentation == 'synthseg':
            return SynthSeg()
        else:
            raise ValueError(
                f'Invalid tissue segmentation {self.config.options.tissue_segmentation}'
            )

    @cached_property
    def initialization(self):
        return Initialization(
            pipeline_dir=self.pipeline_dir,
            modalities=self.config.options.modalities.split(','),
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def skullstripping(self):
        return Skullstripping(
            inference_model=self.skullstripping_model,
            image_processor=self.image_processor,
            pipeline_dir=self.pipeline_dir,
            modalities=self.config.options.modalities.split(','),
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def t1_preprocessing(self):
        return T1Preprocessing(
            image_processor=self.image_processor,
            pipeline_dir=self.pipeline_dir,
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def cross_modality_registration(self):
        return CrossModalityRegistration(
            image_registration=self.image_registration,
            pipeline_dir=self.pipeline_dir,
            modalities_all=self.config.options.modalities.split(','),
            modalities_target=self.config.options.targets.split(','),
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def flair_segmentation(self):
        return SingleModalitySegmentation(
            inference_model=self.flair_model,
            image_processor=self.image_processor,
            pipeline_dir=self.pipeline_dir,
            modality=Modality.FLAIR,
            image_type_input='skullstripped',
            image_type_output='abnormal_seg',
            image_type_mask='skullstripping_mask',
            segmentation_dir='abnormalmap',
            components=True,
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def enhancement_segmentation(self):
        return MultiModalitySegmentation(
            inference_model=self.enhancement_model,
            image_processor=self.image_processor,
            pipeline_dir=self.pipeline_dir,
            modality_list=[Modality.T1, Modality.T1Post],
            output_modality=Modality.T1Post,
            image_type_input='skullstripped',
            image_type_output='abnormal_seg',
            image_type_mask='skullstripping_mask',
            segmentation_dir='abnormalmap',
            components=True,
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def tissue_segmentation(self):
        if self.config.options.tissue_segmentation == 'prior':
            return TissueWithPriorSegementation(
                inference_model=self.tissue_model,
                image_processor=self.image_processor,
                pipeline_dir=self.pipeline_dir,
                image_type_input='trim_upsampled',
                image_type_output='tissue_seg',
                template_name='Tissue',
                overwrite=self.config.options.overwrite_images,
            )
        elif self.config.options.tissue_segmentation == 'synthseg':
            return SynthSegTissueSegmentation(
                inference_model=self.tissue_model,
                image_processor=self.image_processor,
                pipeline_dir=self.pipeline_dir,
                image_type_input='trim_upsampled',
                image_type_output='tissue_seg',
                overwrite=self.config.options.overwrite_images,
            )
        else:
            raise ValueError(
                f'Invalid tissue segmentation {self.config.options.tissue_segmentation}'
            )

    @cached_property
    def t1_postprocessing(self):
        return T1Postprocessing(
            image_processor=self.image_processor,
            pipeline_dir=self.pipeline_dir,
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def t1_registration(self):
        return T1Registration(
            image_processor=self.image_processor,
            image_registration=self.image_registration,
            pipeline_dir=self.pipeline_dir,
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def resampling(self):
        return Resampling(
            image_processor=self.image_processor,
            image_registration=self.image_registration,
            pipeline_dir=self.pipeline_dir,
            modalities_target=self.config.options.targets.split(','),
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def quantification(self):
        return Quantification(
            pipeline_dir=self.pipeline_dir,
            modalities_all=self.config.options.modalities.split(','),
            modalities_target=self.config.options.targets.split(','),
            dominant_tissue=self.config.options.dominant_tissue,
        )

    @cached_property
    def pyalfe_pipeline_runner(self):
        return PyALFEPipelineRunner(
            initialization=self.initialization,
            skullstripping=self.skullstripping,
            t1_preprocessing=self.t1_preprocessing,
            cross_modality_registration=self.cross_modality_registration,
            flair_segmentation=self.flair_segmentation,
            enhancement_segmentation=self.enhancement_segmentation,
            tissue_segmentation=self.tissue_segmentation,
            t1_postprocessing=self.t1_postprocessing,
            t1_registration=self.t1_registration,
            resampling=self.resampling,
            quantification=self.quantification,
        )


class DicomProcessingContianer(DeclarativeContainer):
    """Contianer for dicom processing pipeline depedencies"""

    @cached_property
    def pipeline_dir(self):
        if self.config.options.data_dir_structure == 'alfe':
            return DefaultALFEDataDir(
                output_dir=os.devnull, input_dir=self.config.options.nifti_dir
            )
        elif self.config.options.data_dir_structure == 'bids':
            return BIDSDataDir(
                output_dir=os.devnull,
                input_dir=self.config.options.nifti_dir,
            )
        else:
            raise ValueError(
                f'Invalid data dir structure {self.config.options.data_dir_structure}'
            )

    @cached_property
    def dicom_dir(self):
        return PatientDicomDataDir(dicom_dir=self.config.options.dicom_dir)

    @cached_property
    def dicom_processing(self):
        return DicomProcessing(
            pipeline_dir=self.pipeline_dir,
            dicom_dir=self.dicom_dir,
            overwrite=self.config.options.overwrite_images,
        )

    @cached_property
    def dicom_processing_pipeline_runner(self):
        return DicomProcessingPipelineRunner(dicom_processing=self.dicom_processing)
