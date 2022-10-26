import logging
import os

from pyalfe.data_structure import PipelineDataDir, Modality
from pyalfe.image_processing import ImageProcessor
from pyalfe.image_registration import ImageRegistration
from pyalfe.roi import roi_dict


class CrossModalityRegistration():

    logger = logging.getLogger('CrossModalityRegistration')

    def __init__(
        self,
        image_registration: ImageRegistration,
        pipeline_dir: PipelineDataDir,
        modalities_all: list[Modality],
        modalities_target: list[Modality],
        image_type: str='skullstripped',
        overwrite: bool=True
    ):
        self.pipeline_dir = pipeline_dir
        self.modalities_all = modalities_all
        self.modalities_target = modalities_target
        self.image_registration = image_registration
        self.image_type = image_type
        self.overwrite = overwrite

    def run(self, accession):
        for target in self.modalities_target:
            target_image = self.pipeline_dir.get_processed_image(
                accession, target, image_type=self.image_type)
            if not os.path.exists(target_image):
                self.logger.info(
                    f'{target} is missing.'
                    f'Skipping CrossModalityRegistration for target {target}.')
                continue
            for modality in self.modalities_all:
                modality_image = self.pipeline_dir.get_processed_image(
                    accession, modality, image_type=self.image_type
                )
                if not os.path.exists(modality_image):
                    self.logger.info(
                        f'{modality} is missing'
                        f'Skipping CrossModalityRegistration for {modality}.'
                    )
                    continue
                output = self.pipeline_dir.get_processed_image(
                    accession, modality,
                    image_type=self.image_type,
                    resampling_target=target)
                transform = self.pipeline_dir.get_processed_image(
                    accession, modality,
                    image_type=self.image_type,
                    resampling_target=target, extension='.mat'
                )
                if (not self.overwrite and os.path.exists(output)
                    and os.path.exists(transform)):
                    continue
                if modality == target:
                    if not os.path.exists(output):
                        os.symlink(target_image, output)
                else:
                    self.image_registration.register_affine(
                        target_image, modality_image,
                        transform_output=transform, fast=True)
                    self.image_registration.reslice(
                        target_image, modality_image,
                        output, transform)


class Resampling(object):
    logger = logging.getLogger('Resampling')

    def __init__(
            self,
            image_processor: ImageProcessor,
            image_registration: ImageRegistration,
            pipeline_dir: PipelineDataDir,
            modalities_target,
            overwrite=True
    ):
        self.pipeline_dir = pipeline_dir
        self.modalities_target = modalities_target
        self.image_processor = image_processor
        self.image_registration = image_registration
        self.overwrite = overwrite

    def run(self, accession):
        for target in self.modalities_target:
            target_image = self.pipeline_dir.get_processed_image(
                accession, target, image_type='skullstripped')
            if not os.path.exists(target_image):
                self.logger.info(
                    f'{target} is missing.'
                    f'Skipping Resampling for target {target}.')
                continue

            transform = self.pipeline_dir.get_processed_image(
                accession=accession, modality=Modality.T1,
                image_type='skullstripped',
                resampling_target=target,
                extension='.mat'
            )

            for roi_type, roi_properties in roi_dict.items():

                roi_sub_dir = roi_properties['sub_dir']
                roi_image = self.pipeline_dir.get_processed_image(
                    accession=accession, modality=Modality.T1,
                    image_type=roi_type,
                    sub_dir_name=roi_sub_dir)

                if not os.path.exists(roi_image):
                    self.logger.info(
                        f'ROI image {roi_image} does not exist.'
                        ' Skipping resampling for this image.'
                    )

                output = self.pipeline_dir.get_processed_image(
                    accession, modality=target, image_type=roi_type,
                    resampling_target=target, resampling_origin=Modality.T1,
                    sub_dir_name=roi_sub_dir)

                if not self.overwrite and os.path.exists(output):
                    continue

                self.image_registration.reslice(
                    target_image, roi_image, output, transform)
