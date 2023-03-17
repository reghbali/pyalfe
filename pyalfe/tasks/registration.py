import logging
import os

from pyalfe.data_structure import PipelineDataDir, Modality
from pyalfe.image_processing import ImageProcessor
from pyalfe.image_registration import ImageRegistration
from pyalfe.roi import roi_dict


class CrossModalityRegistration:

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
            target_image = self.pipeline_dir.get_processed_image(
                accession, target, image_type=self.image_type
            )
            if not os.path.exists(target_image):
                self.logger.info(
                    f'{target} is missing.'
                    f'Skipping CrossModalityRegistration for target {target}.'
                )
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
                    accession,
                    modality,
                    image_type=self.image_type,
                    resampling_target=target,
                )
                transform = self.pipeline_dir.get_processed_image(
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
            target_image = self.pipeline_dir.get_processed_image(
                accession, target, image_type=self.image_type
            )
            if not os.path.exists(target_image):
                self.logger.info(
                    f'{target} is missing.' f'Skipping Resampling for target {target}.'
                )
                continue

            transform = self.pipeline_dir.get_processed_image(
                accession=accession,
                modality=Modality.T1,
                image_type=self.image_type,
                resampling_target=target,
                extension='.mat',
            )

            for roi_key, roi_properties in roi_dict.items():

                roi_sub_dir = roi_properties['sub_dir']
                if roi_properties['type'] == 'derived':
                    roi_image = self.pipeline_dir.get_processed_image(
                        accession=accession,
                        modality=Modality.T1,
                        image_type=roi_key,
                        sub_dir_name=roi_sub_dir,
                    )
                elif roi_properties['type'] == 'template':
                    roi_image = self.pipeline_dir.get_processed_image(
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

                output = self.pipeline_dir.get_processed_image(
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
        t1ss = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='skullstripped'
        )
        t1ss_mask = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='skullstripped_mask'
        )

        if not os.path.exists(t1ss):
            self.logger.info(
                'T1 skullstripped image is missing. Skipping T1Registration'
            )
            return

        template = roi_dict['template']['source']
        template_mask = roi_dict['template_mask']['source']
        template_reg_sub_dir = roi_dict['template']['sub_dir']

        rigid_init_transform = self.pipeline_dir.get_processed_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            image_type='greedy_rigid_init',
            sub_dir_name=template_reg_sub_dir,
            extension='.mat',
        )

        self.image_processor.binarize(t1ss, t1ss_mask)

        if self.overwrite or not os.path.exists(rigid_init_transform):
            self.image_registration.register_rigid(
                t1ss_mask, template_mask, rigid_init_transform
            )

        self.pipeline_dir.get_processed_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            sub_dir_name=template_reg_sub_dir,
        )

        affine_transform = self.pipeline_dir.get_processed_image(
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
                t1ss,
                template,
                affine_transform,
                init_transform=rigid_init_transform,
                fast=False,
            )

        warp_transform = self.pipeline_dir.get_processed_image(
            accession,
            Modality.T1,
            resampling_origin='template',
            resampling_target=Modality.T1,
            image_type='warp',
            sub_dir_name=template_reg_sub_dir,
        )

        if self.overwrite or not os.path.exists(warp_transform):
            self.image_registration.register_deformable(
                t1ss,
                template,
                transform_output=warp_transform,
                affine_transform=affine_transform,
            )

        for roi_key in ['template', 'lobes']:
            roi_template_to_t1 = self.pipeline_dir.get_processed_image(
                accession,
                Modality.T1,
                resampling_origin=roi_key,
                resampling_target=Modality.T1,
                sub_dir_name=roi_dict[roi_key]['sub_dir'],
            )
            roi_template = roi_dict[roi_key]['source']

            if self.overwrite or not os.path.exists(roi_template_to_t1):
                self.image_registration.reslice(
                    t1ss,
                    roi_template,
                    roi_template_to_t1,
                    warp_transform,
                    affine_transform,
                )
