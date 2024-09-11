import logging
import os

from pyalfe.data_structure import PipelineDataDir, Modality
from pyalfe.image_processing import ImageProcessor
from pyalfe.image_registration import ImageRegistration
from pyalfe.roi import roi_dict


class CrossModalityRegistration:
    """This task registers all the modalities to target modalities.

    Attributes
    ----------
    image_registration: ImageRegistration
        The image registration object.
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    modalities_all: list[Modality]
        All the modalities that should be registered to the target modalities.
    modalities_target: list[Modality]
        Target modalities.
    image_type: str
        The type of image that should be registered. Default is `skullstripped`.
    overwrite: bool
        Whether to overwrite existing registered images. Default is True.

    """

    logger = logging.getLogger('CrossModalityRegistration')

    def __init__(
        self,
        image_registration: ImageRegistration,
        pipeline_dir: PipelineDataDir,
        modalities_all: list[Modality],
        modalities_target: list[Modality],
        image_type: str = 'skullstripped',
        overwrite: bool = True,
    ):
        self.pipeline_dir = pipeline_dir
        self.modalities_all = modalities_all
        self.modalities_target = modalities_target
        self.image_registration = image_registration
        self.image_type = image_type
        self.overwrite = overwrite

    def run(self, accession):
        for target in self.modalities_target:
            target_image = self.pipeline_dir.get_output_image(
                accession, target, image_type=self.image_type
            )
            if not os.path.exists(target_image):
                self.logger.info(
                    f'{target} is missing.'
                    f' Skipping CrossModalityRegistration for target {target}.'
                )
                continue
            for modality in self.modalities_all:
                modality_image = self.pipeline_dir.get_output_image(
                    accession, modality, image_type=self.image_type
                )
                if not os.path.exists(modality_image):
                    self.logger.info(
                        f'{modality} is missing'
                        f' Skipping CrossModalityRegistration for {modality}.'
                    )
                    continue
                output = self.pipeline_dir.get_output_image(
                    accession,
                    modality,
                    image_type=self.image_type,
                    resampling_target=target,
                )
                transform = self.pipeline_dir.get_output_image(
                    accession,
                    modality,
                    image_type=self.image_type,
                    resampling_target=target,
                    extension='.mat',
                )
                if (
                    not self.overwrite
                    and os.path.exists(output)
                    and os.path.exists(transform)
                ):
                    continue
                if modality == target:
                    if not os.path.exists(output):
                        os.symlink(target_image, output)
                else:
                    self.image_registration.register_affine(
                        target_image,
                        modality_image,
                        transform_output=transform,
                        fast=True,
                    )
                    self.image_registration.reslice(
                        target_image, modality_image, output, transform
                    )


class Resampling:
    """This task resamples all the ROIs in the T1 space to
    the target modalities.

    Attributes
    ----------
    image_processor: ImageProcessor
        The image processor object.
    image_registration: ImageRegistration
        Image registration object.
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    modalities_target: list[Modality]
        Target modalities.
    image_type: str
        The type of image that should be registered. Default is `skullstripped`.
    overwrite: bool
        Whether to overwrite existing registered images. Default is True.
    """

    logger = logging.getLogger('Resampling')

    def __init__(
        self,
        image_processor: ImageProcessor,
        image_registration: ImageRegistration,
        pipeline_dir: PipelineDataDir,
        modalities_target,
        image_type: str = 'skullstripped',
        overwrite=True,
    ):
        self.pipeline_dir = pipeline_dir
        self.modalities_target = modalities_target
        self.image_processor = image_processor
        self.image_registration = image_registration
        self.image_type = image_type
        self.overwrite = overwrite

    def run(self, accession):
        for target in self.modalities_target:
            target_image = self.pipeline_dir.get_output_image(
                accession, target, image_type=self.image_type
            )
            if not os.path.exists(target_image):
                self.logger.info(
                    f'{target} is missing. Skipping Resampling for target {target}.'
                )
                continue

            transform = self.pipeline_dir.get_output_image(
                accession=accession,
                modality=Modality.T1,
                image_type=self.image_type,
                resampling_target=target,
                extension='.mat',
            )

            for roi_key, roi_properties in roi_dict.items():

                roi_sub_dir = roi_properties['sub_dir']
                if roi_properties['type'] == 'derived':
                    roi_image = self.pipeline_dir.get_output_image(
                        accession=accession,
                        modality=Modality.T1,
                        image_type=roi_key,
                        sub_dir_name=roi_sub_dir,
                    )
                elif roi_properties['type'] == 'template':
                    roi_image = self.pipeline_dir.get_output_image(
                        accession=accession,
                        modality=Modality.T1,
                        resampling_origin=roi_key,
                        resampling_target=Modality.T1,
                        sub_dir_name=roi_properties['sub_dir'],
                    )
                else:
                    continue

                if not os.path.exists(roi_image):
                    self.logger.info(
                        f'ROI image {roi_image} does not exist.'
                        ' Skipping resampling for this image.'
                    )

                output = self.pipeline_dir.get_output_image(
                    accession,
                    modality=target,
                    image_type=roi_key,
                    resampling_target=target,
                    resampling_origin=Modality.T1,
                    sub_dir_name=roi_sub_dir,
                )

                if not self.overwrite and os.path.exists(output):
                    continue

                self.image_registration.reslice(
                    target_image, roi_image, output, transform
                )


class T1Registration:
    """This task registers anatomical templates to the T1 image.

    Attributes
    ----------
    image_processor: ImageProcessor
        Image processor object.
    image_registration: ImageRegistration
        Image registration object.
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    overwrite: bool
        Whether to overwrite existing registered images. Default is True.
    """

    logger = logging.getLogger('T1Registration')

    def __init__(
        self,
        image_processor: ImageProcessor,
        image_registration: ImageRegistration,
        pipeline_dir: PipelineDataDir,
        overwrite: bool = True,
    ):
        self.pipeline_dir = pipeline_dir
        self.image_processor = image_processor
        self.image_registration = image_registration
        self.overwrite = overwrite

    def run(self, accession):
        t1ss = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='skullstripped'
        )
        t1ss_mask = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='skullstripping_mask'
        )
        t1trim_upsampled = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='trim_upsampled'
        )

        if not os.path.exists(t1ss):
            self.logger.info(
                'T1 skullstripped image is missing. Skipping T1Registration'
            )
            return
        if not os.path.exists(t1trim_upsampled):
            self.logger.info('T1 trim upsampled is missing. Skipping T1Registration')
            return

        template = roi_dict['template']['source']
        template_mask = roi_dict['template_mask']['source']
        template_reg_sub_dir = roi_dict['template']['sub_dir']

        rigid_init_transform = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            image_type='greedy_rigid_init',
            sub_dir_name=template_reg_sub_dir,
            extension='.mat',
        )

        if self.overwrite or not os.path.exists(rigid_init_transform):
            self.image_registration.register_rigid(
                t1ss_mask, template_mask, rigid_init_transform
            )

        self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            sub_dir_name=template_reg_sub_dir,
        )

        affine_transform = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            image_type='affine',
            sub_dir_name=template_reg_sub_dir,
            extension='.mat',
        )

        if self.overwrite or not os.path.exists(affine_transform):
            self.image_registration.register_affine(
                t1trim_upsampled,
                template,
                affine_transform,
                init_transform=rigid_init_transform,
                fast=False,
            )

        warp_transform = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            image_type='warp',
            sub_dir_name=template_reg_sub_dir,
        )

        if self.overwrite or not os.path.exists(warp_transform):
            self.image_registration.register_deformable(
                t1trim_upsampled,
                template,
                transform_output=warp_transform,
                affine_transform=affine_transform,
            )

        for roi_key, roi_properties in roi_dict.items():
            if roi_properties['type'] != 'template':
                continue
            roi_template_to_t1 = self.pipeline_dir.get_output_image(
                accession,
                Modality.T1,
                resampling_origin=roi_key,
                resampling_target=Modality.T1,
                sub_dir_name=roi_properties['sub_dir'],
            )
            roi_template = roi_properties['source']

            if self.overwrite or not os.path.exists(roi_template_to_t1):
                self.image_registration.reslice(
                    t1trim_upsampled,
                    roi_template,
                    roi_template_to_t1,
                    warp_transform,
                    affine_transform,
                )
