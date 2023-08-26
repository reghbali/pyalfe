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
    """Abstract PipelineDataDir"""

    def get_output_image(
        self,
        accession,
        modality,
        image_type=None,
        resampling_target=None,
        resampling_origin=None,
        sub_dir_name=None,
        extension='.nii.gz',
    ):
        """Generates the path for an output image. All the implementations
         should make the parent directories if they do not exist.

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
            The path to output image.
        """
        raise NotImplementedError

    def get_input_image(self, accession, modality, extension='.nii.gz'):
        """Generates the path for an input image.

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
            The path to input image.
        """
        raise NotImplementedError

    def get_quantification_file(
        self, accession, modality, quantification_file_type, extension='.csv'
    ):
        """Generates the path for an output quantification file.
        All the implementations should make the parent directories
        if they do not exist.

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

    def __init__(self, output_dir, input_dir):
        self.dir_dict = {'output': output_dir, 'input': input_dir}

    def __call__(self, dir_type, accession, *sub_dir_names):
        return os.path.join(self.dir_dict[dir_type], accession, *sub_dir_names)

    def create_dir(self, dir_type, accession, *sub_dir_names, exists=True):
        directory = self(dir_type, accession, *sub_dir_names)
        Path(directory).expanduser().mkdir(parents=True, exist_ok=exists)
        return directory

    def get_output_image(
        self,
        accession,
        modality,
        image_type=None,
        resampling_target=None,
        resampling_origin=None,
        sub_dir_name=None,
        extension='.nii.gz',
    ):
        """Generate the path for an output image in ALFE default
        dir structure format.

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
            The path to output image.

        Examples
        -------
        >>> from pyalfe.data_structure import DefaultALFEDataDir
        >>> data_dir = DefaultALFEDataDir(
        output_dir='/mri-data/processed', input_dir='/mri-data/input')
        >>> data_dir.get_output_image('34521', 'T1')
        '/mri-data/processed/34521/T1/34521_T1.nii.gz'

        >>> data_dir.get_output_image(
        '34521', 'T2', image_type='skullstrippied')
        '/mri-data/processed/34521/T2/34521_T2_skullstrippied.nii.gz'

        >>> data_dir.get_output_image(
        '34521', 'T1Post', resampling_target='T1Post', resampling_origin='template')
        '/mri-data/processed/34521/T1Post/34521_template_to_T1Post.nii.gz'

        >>> data_dir.get_output_image(
        '34521', 'T2', image_type='abnormal_seg', sub_dir_name='abnormalmap')
        '/mri-data/processed/34521/T2/abnormalmap/34521_T2_abnormal_seg.nii.gz'
        """
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
            file_dir = self.create_dir('output', accession, modality, sub_dir_name)
        else:
            file_dir = self.create_dir('output', accession, modality)

        return os.path.join(file_dir, file_name)

    def get_input_image(self, accession, modality, extension='.nii.gz'):
        """Generate the path for an input image in
        ALFE default dir structure format.

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
            The path to input image.

        Examples
        -------
        >>> from pyalfe.data_structure import DefaultALFEDataDir
        >>> data_dir = DefaultALFEDataDir(
        output_dir='/mri-data/processed', input_dir='/mri-data/input')
        >>> data_dir.get_input_image('34521', 'T1')
        '/mri-data/input/34521/T1/T1.nii.gz'
        """
        return os.path.join(self('input', accession, modality), f'{modality}.nii.gz')

    def get_quantification_file(
        self, accession, modality, quantification_file_type, extension='.csv'
    ):
        """Generate the path for a quantification file in
        ALFE default dir structure format.

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

        Examples
        -------
        >>> from pyalfe.data_structure import DefaultALFEDataDir
        >>> data_dir = DefaultALFEDataDir(
        output_dir='/mri-data/processed', input_dir='/mri-data/input')
        >>> data_dir.get_quantification_file(
        '34521', 'T1Post', quantification_file_type='IndividualLesionMeasures')
        '/mri-data/processed/34521/T1Post/quantification/34521_IndividualLesionMeasures.csv'
        """
        quantification_dir = self.create_dir(
            'output', accession, modality, 'quantification'
        )
        return os.path.join(
            quantification_dir, f'{accession}_{quantification_file_type}{extension}'
        )


class BIDSDataDir(PipelineDataDir):
    """
    BIDS almost-compliant implementation of PipelineDataDir
    """

    logger = logging.getLogger('BIDSDataDir')

    def __init__(self, output_dir, input_dir):
        self.input_layout = BIDSLayout(Path(input_dir).expanduser())

        with open(
            Path(output_dir).expanduser() / 'dataset_description.json', 'w'
        ) as file:
            json.dump(
                {
                    'Name': 'ALFE Output',
                    'BIDSVersion': '1.0.2',
                    'DatasetType': 'derivative',
                    'GeneratedBy': [
                        {
                            'Name': 'ALFE',
                            'Version': importlib.metadata.version('pyalfe'),
                        }
                    ],
                },
                file,
            )
        self.output_layout = BIDSLayout(
            Path(output_dir).expanduser(), is_derivative=True
        )

    modality_dict = {
        Modality.T1: {'suffix': 'T1w', 'datatype': 'anat'},
        Modality.T1Post: {'suffix': 'T1w', 'ceagent': 'gadolinium', 'datatype': 'anat'},
        Modality.T2: {'suffix': 'T2w', 'datatype': 'anat'},
        Modality.FLAIR: {'suffix': 'FLAIR', 'datatype': 'anat'},
        Modality.SWI: {'suffix': 'swi', 'datatype': 'swi'},
        Modality.ADC: {'suffix': 'md', 'datatype': 'dwi'},
        Modality.CBF: {'suffix': 'cbf', 'datatype': 'perf'},
        Modality.ASL: {'suffix': 'asl', 'datatype': 'perf'},
    }

    def get_input_image(self, accession, modality, extension='.nii.gz'):
        """Generates the path for an input image in BIDS dir structure format.

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
            The path to input image.

        Examples
        -------
        >>> from pyalfe.data_structure import BIDSDataDir
        >>> data_dir = BIDSDataDir(
        output_dir='/mri-data/processed', input_dir='/mri-data/input')
        >>> data_dir.get_input_image('34521', 'T1')
        '/mri-data/input/sub-34521/anat/sub-34521_T1w.nii.gz'

        >>> data_dir.get_input_image('34521', 'ADC')
        '/mri-data/input/sub-34521/derivative/dwi/sub-34521_md.nii.gz'

        >>> data_dir.get_input_image('34521', 'CBF')
        '/mri-data/input/sub-34521/derivative/perf/sub-34521_cbf.nii.gz'

        >>> data_dir.get_input_image('34521', 'SWI')
        '/mri-data/input/sub-34521/derivative/swi/sub-34521_swi.nii.gz'

        >>> data_dir.get_input_image('34521', 'T1Post')
        '/mri-data/input/sub-34521/anat/sub-34521_ce-gadolinium_T1w.nii.gz'
        """
        patterns = [
            'sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}]'
            '[_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]'
            '[_run-{run}][_part-{part}]_{suffix<T1w|T2w|FLAIR|dwi|asl>}'
            '{extension<.nii|.nii.gz>|.nii.gz}',
            'sub-{subject}[/ses-{session}]/derivative/{datatype}/sub-{subject}[_ses-{session}]'
            '[_task-{task}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]'
            '[_run-{run}][_part-{part}]_{suffix<swi|cbf|md>}'
            '{extension<.nii|.nii.gz>|.nii.gz}',
        ]
        entities = {
            'subject': accession,
            **self.modality_dict[modality],
            'extension': extension,
        }
        candidates = self.input_layout.get(*entities, patterns)
        if len(candidates) == 0:
            default_path = self.input_layout.build_path(
                entities, patterns, validate=False
            )
            self.logger.warning(
                f'Did not find any file matching {entities} '
                f'when looking for {modality} image for {accession}.'
                f'Returning the default path {default_path}.'
            )
            # I see you watching me, you should watch her.
            # Cause she is sick of your tricks, I am the doctor.
            return default_path
        elif len(candidates) == 1:
            return candidates[0]
        else:
            self.logger.warning(
                f'Multiple files matching {entities} were found '
                f'when looking for {modality} image for {accession}: '
                f'{candidates}. Choosing {candidates[0]}.'
            )
            return candidates[0]

    def get_output_image(
        self,
        accession,
        modality,
        image_type=None,
        resampling_target=None,
        resampling_origin=None,
        sub_dir_name=None,
        extension='.nii.gz',
    ):
        """Generates the path for an output image in BIDS dir structure format.

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
            The path to output image.

        Examples
        -------
        >>> from pyalfe.data_structure import BIDSDataDir
        >>> data_dir = BIDSDataDir(
        output_dir='/mri-data/processed', input_dir='/mri-data/input')
        >>> data_dir.get_output_image('34521', 'T1')
        '/mri-data/processed/sub-34521/anat/sub-34521_T1w.nii.gz'

        >>> data_dir.get_output_image('34521', 'T2', image_type='skullstrippied')
        '/mri-data/processed/sub-34521/anat/sub-34521_desc-skullstrippied_T2w.nii.gz'

        >>> data_dir.get_output_image(
        '34521', 'T1Post', resampling_target='T1Post', resampling_origin='template')
        '/mri-data/processed/sub-34521/anat/sub-34521_ce-gadolinium_space-orig_desc-template_T1w.nii.gz'

        >>> data_dir.get_output_image(
        '34521', 'T2', image_type='abnormal_seg', sub_dir_name='abnormalmap')
        '/mri-data/processed/sub-34521/anat/sub-34521_desc-abnormalseg_dseg.nii.gz'
        """
        pattern = [
            'sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}'
            '[_ce-{ceagent}][_space-{space}][_desc-{desc}]'
            '_{suffix<T1w|T2w|FLAIR|swi|asl|cbf|md|mask|dseg|probseg>}'
            '{extension<.nii|.nii.gz>|.nii.gz}'
        ]
        entities = {
            'subject': accession,
            **self.modality_dict[modality],
            'extension': extension,
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

            if 'desc' in entities:
                entities['desc'] += re.sub('[^A-Za-z0-9]+', '', image_type)
            else:
                entities['desc'] = re.sub('[^A-Za-z0-9]+', '', image_type)

        ret = self.output_layout.build_path(entities, pattern, validate=False)
        Path(ret).parent.mkdir(parents=True, exist_ok=True)
        return ret

    def get_quantification_file(
        self, accession, modality, quantification_file_type, extension='.csv'
    ):
        """Generate the path for a quantification file in
        BIDS dir structure format.

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

        Examples
        -------
        >>> from pyalfe.data_structure import BIDSDataDir
        >>> data_dir = BIDSDataDir(
        output_dir='/mri-data/processed', input_dir='/mri-data/input')
        >>> data_dir.get_quantification_file(
        '34521', 'T1Post', quantification_file_type='IndividualLesionMeasures')
        '/mri-data/processed/sub-34521/anat/quantification/sub-34521_ce-gadolinium_desc-T1Post_IndividualLesionMeasures.csv'

        """
        pattern = (
            'sub-{subject}[/ses-{session}]/{datatype}/quantification/'
            'sub-{subject}[_ses-{session}][_ce-{ceagent}][_desc-{desc}]'
            '_{file_type}{extension<.tsv|.csv>}'
        )

        entities = {
            'subject': accession,
            **self.modality_dict[modality],
            'extension': extension,
            'desc': modality,
            'file_type': quantification_file_type,
        }

        ret = self.output_layout.build_path(entities, pattern, validate=False)
        Path(ret).parent.mkdir(parents=True, exist_ok=True)
        return ret
