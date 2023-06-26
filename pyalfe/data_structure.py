import importlib
import json
import logging
import re
from abc import ABC
from enum import auto, IntEnum
import os
from pathlib import Path

from bids import BIDSLayout

from strenum import StrEnum


class Modality(StrEnum):
    T1 = auto()
    T1Post = auto()
    T2 = auto()
    FLAIR = auto()
    SWI = auto()
    DWI = auto()
    SingleDWI = auto()
    ADC = auto()
    EADC = auto()
    CBF = auto()
    ASL = auto()
    PERFUSION = auto()


class Orientation(StrEnum):
    AXIAL = auto()
    SAGITTAL = auto()
    CORONAL = auto()


class Tissue(IntEnum):
    BACKGROUND = 0
    CSF = 1
    CORTICAL_GRAY_MATTER = 2
    WHITE_MATTER = 3
    DEEP_GRAY_MATTER = 4
    BRAIN_STEM = 5
    CEREBELLUM = 6


class PipelineDataDir(ABC):
    """
    Abstract PipelineDataDir

    Methods
    -------
    get_processed_image
    get_classified_image
    get_quantification_file
    """

    def get_processed_image(
        self,
        accession,
        modality,
        image_type=None,
        resampling_target=None,
        resampling_origin=None,
        sub_dir_name=None,
        extension='.nii.gz',
    ):
        """

        Parameters
        ----------
        accession: str
            The accession or study number.
        modality: Modality or str
            Image modality.
        image_type: str, default=None
            Image type. For example: skullstripped.
        resampling_target: Modality or str, default=None
            The target space that the image has been resampled to. For example,
            for a `T2` image that is resampled to `FLAIR`, resampling
            target is `FLAIR`.
        resampling_origin: Modality, str, default=None
            The origin of image before resampling. For example, if a `template`
            image is resampled to `T1`, the resampling origin is `template`.
        sub_dir_name: str, default=None
            The name of the sub directory inside the modality directory in
            which the image is stored. If None the image is stored directory
            inside the modality directory.
        extension: str, default='.nii.gz'
            The extension of the image.

        Returns
        -------
        str
            The path to processed image.
        """
        raise NotImplementedError

    def get_classified_image(self, accession, modality, extension='.nii.gz'):
        """

        Parameters
        ----------
        accession: str
            The accession or study number.
        modality: Modality or str
            Image modality.
        extension: str, default='.nii.gz'
            The extension of the image.

        Returns
        -------
        str
            The path to classified image.
        """
        raise NotImplementedError

    def get_quantification_file(
        self, accession, modality, quantification_file_type, extension='.csv'
    ):
        """

        Parameters
        ----------
        accession: str
            The accession or study number.
        modality: Modality or str
            Image modality.
        quantification_file_type: str, default=None
            Image type. For example: SummaryLesionMeasures.
        extension: str, default='.csv'
            The extension of the quantification.

        Returns
        -------
        str
            The path to quantification file.
        """
        raise NotImplementedError


class DefaultALFEDataDir(PipelineDataDir):
    """
    Default implementation of PipelineDataDir
    """
    def __init__(self, processed, classified):
        self.dir_dict = {'processed': processed, 'classified': classified}

    def __call__(self, dir_type, accession, *sub_dir_names):
        return os.path.join(self.dir_dict[dir_type], accession, *sub_dir_names)

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
            file_dir = self.create_dir('processed', accession, modality, sub_dir_name)
        else:
            file_dir = self.create_dir('processed', accession, modality)

        return os.path.join(file_dir, file_name)

    def get_classified_image(self, accession, modality, extension='.nii.gz'):
        return os.path.join(
            self('classified', accession, modality), f'{modality}.nii.gz'
        )

    def get_quantification_file(
        self, accession, modality, quantification_file_type, extension='.csv'
    ):
        quantification_dir = self.create_dir(
            'processed', accession, modality, 'quantification'
        )
        return os.path.join(
            quantification_dir, f'{accession}_{quantification_file_type}{extension}'
        )


class BIDSDataDir(PipelineDataDir):
    logger = logging.getLogger('BIDSDataDir')

    def __init__(self, processed, classified):
        self.classified_layout = BIDSLayout(classified)

        with open(os.path.join(processed, 'dataset_description.json'),
                  'w') as file:
            json.dump(
                {'Name': 'ALFE Output',
                 'BIDSVersion': '1.0.2',
                 'DatasetType': 'derivative',
                 'GeneratedBy': [
                     {'Name': 'ALFE',
                      'Version': importlib.metadata.version('pyalfe')}]
                 }, file)
        self.processed_layout = BIDSLayout(processed, is_derivative=True)

    modality_dict = {
        Modality.T1: {'suffix': 'T1w', 'datatype': 'anat'},
        Modality.T1Post: {'suffix': 'T1w', 'ceagent': 'gadolinium',
                          'datatype': 'anat'},
        Modality.T2: {'suffix': 'T2w', 'datatype': 'anat'},
        Modality.FLAIR: {'suffix': 'FLAIR', 'datatype': 'anat'},
        Modality.SWI: {'suffix': 'swi', 'datatype': 'swi'},
        Modality.ADC: {'suffix': 'md', 'datatype': 'dwi'},
        Modality.CBF: {'suffix': 'cbf', 'datatype': 'perf'},
        Modality.ASL: {'suffix': 'asl', 'datatype': 'perf'}}

    def get_classified_image(self, accession, modality, extension='.nii.gz'):
        path = (
            'sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}]'
            '[_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]'
            '[_run-{run}][_part-{part}]_{suffix<T1w|T2w|FLAIR|swi|asl|cbf|md>}'
            '{extension<.nii|.nii.gz>|.nii.gz}')
        entities = {
            'subject': accession,
            **self.modality_dict[modality],
            'extension': extension
        }
        candidates = self.classified_layout.get(*entities)
        if len(candidates) == 0:
            self.logger.warning(
                f'Did not find any file matching {entities} '
                f'when looking for {modality} image for {accession}.')
        elif len(candidates) == 1:
            return candidates[0]
        else:
            self.logger.warning(
                f'Multiple files matching {entities} were found '
                f'when looking for {modality} image for {accession}: '
                f'{candidates}. Choosing {candidates[0]}.')
            return candidates[0]

    def get_processed_image(
            self,
            accession,
            modality,
            image_type=None,
            resampling_target=None,
            resampling_origin=None,
            sub_dir_name=None,
            extension='.nii.gz'):
        pattern = ('sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}'
                   '[_ce-{ceagent}][_space-{space}][_desc-{desc}]'
                   '_{suffix<T1w|T2w|FLAIR|swi|asl|cbf|md|mask|dseg|probseg>}'
                   '{extension<.nii|.nii.gz>|.nii.gz}')
        entities = {
            'subject': accession,
            **self.modality_dict[modality],
            'extension': extension
        }
        if resampling_target == modality:
            entities['space'] = 'orig'
        elif resampling_target:
            entities['space'] = resampling_target

        if resampling_origin:
            entities['desc'] = resampling_origin

        if image_type:
            if re.search(r'(?i)prob', image_type):
                entities['suffix'] = 'probseg'
            elif re.search(r'(?i)seg', image_type):
                entities['suffix'] = 'dseg'
            elif re.search(r'(?i)mask', image_type):
                entities['suffix'] = 'mask'
            elif entities['desc']:
                entities['desc'] += image_type
            else:
                entities['desc'] = image_type

        ret = self.processed_layout.build_path(entities, pattern,
                                               validate=False)
        Path(ret).parent.mkdir(parents=True, exist_ok=True)
        return ret

    def get_quantification_file(
            self, accession, modality, quantification_file_type,
            extension='.csv'
    ):
        pattern = ('sub-{subject}[/ses-{session}]/{datatype}/quantification/'
                   'sub-{subject}[_ses-{session}][_ce-{ceagent}][_desc-{desc}]'
                   '_{file_type}{extension<.tsv|.csv>}')

        entities = {'subject': accession, **self.modality_dict[modality],
                    'extension': extension, 'desc': modality,
                    'file_type': quantification_file_type}

        ret = self.processed_layout.build_path(entities, pattern,
                                               validate=False)
        Path(ret).parent.mkdir(parents=True, exist_ok=True)
        return ret