from collections import defaultdict
import logging
import math
import os
from pathlib import Path
import subprocess

from pyalfe.data_structure import (
    PipelineDataDir,
    PatientDicomDataDir,
    Modality,
    Orientation
)
from pyalfe.tasks import Task
from pyalfe.utils.dicom import (
    extract_dicom_meta,
    get_max_echo_series_crc,
    ImageMeta
)
from pyalfe.utils.technique import detect_modality, detect_orientation


class DicomProcessing(Task):
    """This task classifies raw dicom images into their modality and converts
    them to NIfTI.  
    """

    logger = logging.getLogger('DicomProcessing')

    def __init__(
        self,
        pipeline_dir: PipelineDataDir,
        dicom_dir: PatientDicomDataDir,
        overwrite: bool = True,
        ) -> None:
        super().__init__()
        self.pipeline_dir = pipeline_dir
        self.dicom_dir = dicom_dir
        self.overwrite = overwrite
    
    @staticmethod
    def dicom2nifti(
        dcm_series_dir: str,
        nifti_path: str,
        series_crc: str = None,
        ):
        """This function is a wrapper around dcm2niix that converts a 
        dicom series to a NIfTI file.

        Parameters
        ----------
        dcm_series_dir: str
            The path to a dicom series directory
        nifti_path: str 
            The path to the output NIfTI file
        series_crc: str, default None
            The Series CRC number that should be converted. This option should
            be set when there are that one echo numbers in one series. 

        Returns
        -------
        """
        nifti_dir = os.path.dirname(nifti_path)
        Path(nifti_dir).mkdir(parents=True, exist_ok=True)
        nifti_name = os.path.basename(nifti_path).split('.')[0]
        cmd_parts = [ 
            'dcm2niix',
            '-z', 'y',
            '-b', 'y',
            '-f', nifti_name,
            '-o', nifti_dir]

        if series_crc is not None:
            cmd_parts += ['-n', series_crc]
            
        return subprocess.run(cmd_parts + [dcm_series_dir])

    @staticmethod
    def get_best(image_list: list[ImageMeta]) -> ImageMeta:
        """This function returns the image with the smallest slice thickness.
        
        Paramters
        ---------
        image_list: list[ImageMeta]
            List of image meta data
        
        Returns
        -------
        ImageMeta
            The image meta for the image with the smallest slice thickness.
        """
        def none_to_inf(val):
            if val is not None:
                return val
            else:
                return math.inf
        return min(
            image_list,
            key=lambda image_meta: none_to_inf(image_meta.slice_thickness)
            )

    
    @staticmethod
    def select_orientation(
            orientation_dict: dict[Orientation, ImageMeta]
    ) -> tuple[Orientation, ImageMeta]:
        selected_orientation = None
        selected_image = None

        if Orientation.AXIAL in orientation_dict:
            selected_orientation = Orientation.AXIAL
            selected_image = orientation_dict[Orientation.AXIAL]
        else:
            min_thickness = float('inf')
            for orientation in [Orientation.SAGITTAL, Orientation.CORONAL]:
                if orientation in orientation_dict:
                    image_meta = orientation_dict[orientation]
                    if min_thickness > image_meta['slice_thickness']:
                        min_thickness = image_meta['slice_thickness']
                        selected_orientation = orientation
                        selected_image = image_meta
        return selected_orientation, selected_image
    
    def run(self, accession: str) -> None:
        classified = defaultdict(defaultdict(list).copy)
        converted, conversion_failed = defaultdict(str), defaultdict(str)
        skipped = []
        meta_series_map = {}
        all_series_instances = self.dicom_dir.get_all_dicom_series_instances(
            accession)

        for series_uid, instances in all_series_instances.items():
            if len(instances) == 0:
                continue

            dicom_meta = extract_dicom_meta(
                instances[0], series_uid=series_uid)
            modality = detect_modality(
                dicom_meta.series_desc,
                dicom_meta.seq,
                dicom_meta.contrast_agent,
                dicom_meta.te)
            orientation = detect_orientation(
                dicom_meta.patient_orientation_vector)

            if modality is None or orientation is None:
                skipped.append(dicom_meta)
                continue

            classified[modality][orientation].append(dicom_meta)

        for modality, orientations in classified.items():
            orientation_best = defaultdict(Modality)
            
            for orientation, image_list in orientations.items():
                orientation_best[orientation] = self.get_best(image_list)

            selected_orientation, selected_image_meta = self.select_orientation(orientation_best)
            nifti_path = self.pipeline_dir.get_input_image(
                accession=accession, modality=modality)
            selected_series_uid = selected_image_meta.series_uid
            dicom_dir = self.dicom_dir.get_series(
                accession, selected_series_uid)
            max_echo_series_crc = get_max_echo_series_crc(
                all_series_instances[selected_series_uid])

            conversion_status = self.dicom2nifti(
                dcm_series_dir=dicom_dir,
                nifti_path=nifti_path,
                series_crc=max_echo_series_crc)
        
            if conversion_status == 0:
                self.logger.warning(f'failed to convert {dicom_dir} to nifti.')
                conversion_failed[modality] = selected_image_meta
                continue
            converted[modality] = selected_image_meta
            


        

            


        

