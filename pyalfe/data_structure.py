from enum import Enum, IntEnum
import os
from pathlib import Path


class Modality(str, Enum):
    T1 = 'T1'
    T1Post = 'T1Post'
    T2 = 'T2'
    FLAIR = 'FLAIR'
    GRE = 'GRE'
    DWI = 'DWI'
    SingleDWI = 'SingleDWI'
    ADC = 'ADC'
    EADC = 'EADC'
    ASL = 'ASL'
    PERFUSION = 'PERFUSION'


class Orientation(str, Enum):
    AXIAL = 'AXIAL'
    SAGITTAL = 'SAGITTAL'
    CORONAL = 'CORONAL'


class Tissue(IntEnum):
    BACKGROUND = 0
    CSF = 1
    CORTICAL_GRAY_MATTER = 2
    WHITE_MATTER = 3
    DEEP_GRAY_MATTER = 4
    BRAIN_STEM = 5
    CEREBELLUM = 6


class PipelineDataDir:

    def __init__(self, **kwargs):
        self.dir_dict = kwargs

    def get_processed_image(
        self,
        accession,
        modality,
        image_type=None,
        resampling_target=None,
        resampling_origin=None,
        sub_dir_name=None,
        extension='.nii.gz'
    ):
        raise NotImplementedError

    def get_classified_image(self, accession, modality, extension='.nii.gz'):
        raise NotImplementedError

    def get_quantification_file(
        self,
        accession,
        modality,
        quantification_file_type,
        extension='.csv'
    ):
        raise NotImplementedError


class DefaultALFEDataDir(PipelineDataDir):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, dir_type, accession, *sub_dir_names):
        return os.path.join(
            self.dir_dict[dir_type], accession, *sub_dir_names)

    def create_dir(self, dir_type, accession, *sub_dir_names, exists=True):
        directory = self(dir_type, accession, *sub_dir_names)
        Path(directory).mkdir(parents=True, exist_ok=exists)
        return directory

    def get_processed_image(
            self,
            accession,
            modality,
            image_type=None,
            resampling_target=None,
            resampling_origin=None,
            sub_dir_name=None,
            extension='.nii.gz'
    ):
        if not resampling_origin:
            file_name = f'{accession}_{modality}'
        else:
            file_name = f'{accession}_{resampling_origin}'

        if resampling_target:
            file_name += '_to_' + resampling_target

        if image_type:
            file_name += '_' + image_type

        file_name += extension

        if sub_dir_name:
            file_dir = self.create_dir(
                'processed', accession, modality, sub_dir_name)
        else:
            file_dir = self.create_dir(
                'processed', accession, modality)

        return os.path.join(file_dir, file_name)

    def get_classified_image(self, accession, modality, extension='.nii.gz'):
        return os.path.join(
            self('classified', accession, modality),
            f'{modality}.nii.gz')

    def get_quantification_file(
            self,
            accession,
            modality,
            quantification_file_type,
            extension='.csv'
    ):
        quantification_dir = self.create_dir('processed', accession,
                                             modality, 'quantification')
        return os.path.join(
            quantification_dir,
            f'{accession}_{quantification_file_type}{extension}'
        )

