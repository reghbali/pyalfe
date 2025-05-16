import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd

from pyalfe.data_structure import PipelineDataDir, Tissue, Modality
from pyalfe.roi import roi_dict
from pyalfe.tasks import Task

try:
    from radiomics import featureextractor
except ImportError:
    featureextractor = None


class Quantification(Task):
    """This task generates the quantification files.
    This is the final task in the pipeline.

    Attributes
    ----------
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    modalities_all: list[Modality]
        All the modalities for processed by the pipeline.
    modalities_target: list[Modality]
        Target modalities that are used to defince lesions.
    dominant_tissue: str
        The dominant tissue where the tumor or lesion is expected to be
        located at. THe options are `white_matter`, `gray_matter`, `auto`.
        If `auto` is chosen, then the dominant tissue is chosen to be the tissue
        with which the lesion most intersects with.
    """

    logger = logging.getLogger('Quantification')

    def __init__(
        self,
        pipeline_dir: PipelineDataDir,
        modalities_all: list[Modality],
        modalities_target: list[Modality],
        dominant_tissue: str = None,
    ):
        self.modalities_all = modalities_all
        self.modalities_target = modalities_target
        self.pipeline_dir = pipeline_dir
        self.dominant_tissue = dominant_tissue

        if featureextractor:
            self.radiomics_extractor = featureextractor.RadiomicsFeatureExtractor()
            self.radiomics_extractor.disableAllFeatures()
            self.radiomics_extractor.enableFeaturesByName(firstorder=[], shape=[])
            self.radiomics_enabled = True
        else:
            self.radiomics_enabled = False

    @staticmethod
    def load_nii_gz(filename):
        return nib.load(filename)

    def load(self, filename):
        nibabel_image = self.load_nii_gz(filename)
        return (
            nibabel_image.get_fdata().flatten(),
            np.prod(nibabel_image.header.get_zooms()),
        )

    def load_modality_images(self, accession, target):
        modality_images = {}

        for modality in self.modalities_all:

            modality_image_to_target_file = self.pipeline_dir.get_output_image(
                accession,
                modality,
                image_type='skullstripped',
                resampling_target=target,
            )

            if os.path.exists(modality_image_to_target_file):
                modality_images[modality], _ = self.load(modality_image_to_target_file)
        return modality_images

    def load_template_images(self, accession, modality):
        target_images = {}

        for roi_key, roi_properties in roi_dict.items():

            roi_sub_dir = roi_properties['sub_dir']

            if roi_properties['type'] == 'template':
                if modality == Modality.T1:
                    template_image_to_target_file = self.pipeline_dir.get_output_image(
                        accession,
                        modality=modality,
                        resampling_target=modality,
                        resampling_origin=roi_key,
                        sub_dir_name=roi_sub_dir,
                    )
                else:
                    template_image_to_target_file = self.pipeline_dir.get_output_image(
                        accession,
                        modality=modality,
                        image_type=roi_key,
                        resampling_target=modality,
                        resampling_origin=Modality.T1,
                        sub_dir_name=roi_sub_dir,
                    )

                if os.path.exists(template_image_to_target_file):
                    target_images[roi_key], _ = self.load(template_image_to_target_file)
                    self.logger.info(
                        'loaded template: ' + template_image_to_target_file
                    )
                    target_images[roi_key] = np.round(target_images[roi_key])
                else:
                    self.logger.info(
                        'missing template: ' + template_image_to_target_file
                    )
        return target_images

    def get_radiomics(self, skullstripped_file, lesion_seg_file):
        try:
            return self.radiomics_extractor.execute(skullstripped_file, lesion_seg_file)
        except ValueError as e:
            self.logger.debug(f'Pyradiomics failed with this error {e}.')
            return {}

    def get_brain_volume_stats(
        self, brain_mask, tissue_seg, ventricles_seg, template_images, voxel_volume
    ):
        stats = {}
        brain_indices = np.where(brain_mask == 1)[0]
        stats['total_brain_volume'] = len(brain_indices) * voxel_volume

        if ventricles_seg is not None:
            stats['total_ventricles_volume'] = (
                len(np.where(ventricles_seg == 1)[0]) * voxel_volume
            )

        if tissue_seg is not None:
            for tissue in Tissue:
                stats[f'volume_of_{tissue.name.lower()}'] = (
                    len(np.where(tissue_seg == tissue)[0]) * voxel_volume
                )

        for template_key, template_image in template_images.items():
            if 'regions' not in roi_dict[template_key]:
                continue
            regions = roi_dict[template_key]['regions']
            for region_key, region_values in regions.items():
                stats[f'volume_of_{region_key}'] = (
                    np.sum(np.isin(template_image, region_values)) * voxel_volume
                )

        return stats

    def get_lesion_stats(
        self,
        lesion_seg,
        tissue_seg,
        ventricles_distance,
        modality_images,
        template_images,
        voxel_volume,
        lesion_label=1,
    ):
        stats = {}
        lesion_indices = np.where(lesion_seg == lesion_label)[0]

        volume = len(lesion_indices) * voxel_volume
        stats['total_lesion_volume'] = volume
        if volume == 0.0:
            return stats

        unique_counts = np.unique(tissue_seg[lesion_indices], return_counts=True)

        if self.dominant_tissue and self.dominant_tissue != 'auto':
            invalid_tissue = True
            for tissue in Tissue:
                tissue_id = int(tissue)
                tissue_name = tissue.name.lower()
                if (
                    self.dominant_tissue == tissue_id
                    or self.dominant_tissue == tissue_name
                ):
                    dominant_tissue_id = tissue_id
                    invalid_tissue = False
                    break
            if invalid_tissue:
                raise ValueError(
                    f'{self.dominant_tissue} is invalid.'
                    ' The valid options are: {list(Tissue)}'
                )
        else:
            dominant_tissue_id = unique_counts[0][np.argmax(unique_counts[1])]

        tissue_freq = dict(zip(*unique_counts))
        for tissue in Tissue:
            if tissue in tissue_freq:
                stats[f'lesion_volume_in_{tissue.name.lower()}'] = (
                    tissue_freq[tissue] * voxel_volume
                )
            else:
                stats[f'lesion_volume_in_{tissue.name.lower()}'] = 0

        for modality_name, modality_image in modality_images.items():
            if modality_image is None:
                stats[f'relative_{modality_name}_signal'] = np.nan
                continue
            healthy_dom_tissue_indices = np.logical_and(
                tissue_seg == dominant_tissue_id, lesion_seg == 0
            )

            if len(healthy_dom_tissue_indices) == 0:
                self.logger.warning('There is np healthy voxel in the dominant tissue')

            mean_signal = np.mean(modality_image[healthy_dom_tissue_indices])
            stats[f'relative_{modality_name}_signal'] = (
                np.mean(modality_image[lesion_indices]) / mean_signal
            )
            if modality_name in [Modality.ADC, Modality.CBF]:
                stats[f'relative_min_{modality_name}_signal'] = (
                    np.min(modality_image[lesion_indices]) / mean_signal
                )
                stats[f'min_{modality_name}_signal'] = np.min(
                    modality_image[lesion_indices]
                )
                stats[f'mean_{modality_name}_signal'] = np.mean(
                    modality_image[lesion_indices]
                )
                stats[f'median_{modality_name}_signal'] = np.median(
                    modality_image[lesion_indices]
                )
                stats[f'five_percentile_{modality_name}_signal'] = np.percentile(
                    modality_image[lesion_indices], 5
                )
                stats[f'ninety_five_percentile_{modality_name}_signal'] = np.percentile(
                    modality_image[lesion_indices], 95
                )

        if Modality.T1 in modality_images and Modality.T1Post in modality_images:
            t1_image = modality_images[Modality.T1]
            t1post_image = modality_images[Modality.T1Post]
            stats['enhancement'] = np.mean(t1post_image[lesion_indices]) / np.mean(
                t1_image[lesion_indices]
            )
        else:
            self.logger.info(
                'Cannot quantify avg enhancement over the leasion because T1'
                ' or T1Post are not registered to target modality'
            )

        if ventricles_distance is not None:
            stats['average_dist_to_ventricles_(voxels)'] = np.mean(
                ventricles_distance[lesion_indices]
            )
            stats['minimum_dist_to_Ventricles_(voxels)'] = np.min(
                ventricles_distance[lesion_indices]
            )

        for template_key, template_image in template_images.items():
            if 'regions' not in roi_dict[template_key]:
                continue
            regions = roi_dict[template_key]['regions']
            for region_key, region_values in regions.items():
                stats[f'lesion_volume_in_{region_key}'] = (
                    np.sum(np.isin(template_image, region_values)[lesion_indices])
                    * voxel_volume
                )
                stats[f'percentage_volume_in_{region_key}'] = (
                    stats[f'lesion_volume_in_{region_key}']
                    * 100
                    / stats['total_lesion_volume']
                )

        return stats

    @staticmethod
    def get_summary_update_from_individual(individual_lesion_df):
        if len(individual_lesion_df) == 0:
            return {
                'number_of_lesions': 0,
                'largest_lesion_volume': 0.0,
                'average_lesion_volume': 0.0,
            }
        return {
            'number_of_lesions': len(individual_lesion_df),
            'largest_lesion_volume': individual_lesion_df['total_lesion_volume'].max(),
            'average_lesion_volume': individual_lesion_df['total_lesion_volume'].mean(),
        }

    def run(self, accession):
        volumetric_quantification_file = self.pipeline_dir.get_quantification_file(
            accession, Modality.T1, 'volumeMeasures'
        )

        brain_mask_file = self.pipeline_dir.get_output_image(
            accession=accession, modality=Modality.T1, image_type='skullstripping_mask'
        )

        if not os.path.exists(brain_mask_file):
            self.logger.info(
                'T1 skullstripping mask is missing.'
                'Skipping brain volume quantification.'
            )
        else:
            brain_mask, voxel_volume = self.load(brain_mask_file)

            tissue_seg_file = self.pipeline_dir.get_output_image(
                accession=accession, modality=Modality.T1, image_type='tissue_seg'
            )

            if not os.path.exists(tissue_seg_file):
                tissue_seg = None
            else:
                tissue_seg, _ = self.load(tissue_seg_file)

            template_images = self.load_template_images(accession, Modality.T1)

            ventricles_seg_file = self.pipeline_dir.get_output_image(
                accession=accession, modality=Modality.T1, image_type='VentriclesSeg'
            )

            if not os.path.exists(ventricles_seg_file):
                ventricles_seg = None
            else:
                ventricles_seg, _ = self.load(ventricles_seg_file)

            volume_stats = self.get_brain_volume_stats(
                brain_mask, tissue_seg, ventricles_seg, template_images, voxel_volume
            )
            pd.Series(volume_stats).to_csv(volumetric_quantification_file)

        for target in self.modalities_target:
            summary_quantification_file = self.pipeline_dir.get_quantification_file(
                accession, target, 'SummaryLesionMeasures'
            )
            individual_quantification_file = self.pipeline_dir.get_quantification_file(
                accession, target, 'IndividualLesionMeasures'
            )
            radiomics_file = self.pipeline_dir.get_quantification_file(
                accession, target, 'radiomics'
            )

            skullstripped_file = self.pipeline_dir.get_output_image(
                accession, target, image_type='skullstripped'
            )
            lesion_seg_file = self.pipeline_dir.get_output_image(
                accession=accession,
                modality=target,
                image_type='abnormal_seg',
                sub_dir_name='abnormalmap',
            )

            lesion_seg_comp_file = self.pipeline_dir.get_output_image(
                accession=accession,
                modality=target,
                image_type='abnormal_seg_comp',
                sub_dir_name='abnormalmap',
            )

            if not os.path.exists(lesion_seg_file):
                self.logger.info(
                    f'Lesion seg file for {target} is missing.'
                    f'Skipping quantification for {target}.'
                )
                continue

            lesion_seg, voxel_volume = self.load(lesion_seg_file)
            tissue_seg_file = self.pipeline_dir.get_output_image(
                accession,
                target,
                image_type='tissue_seg',
                resampling_origin='T1',
                resampling_target=target,
            )

            if not os.path.exists(tissue_seg_file):
                self.logger.info(
                    f'Tissue seg file for {target} is missing.'
                    f'Skipping quantification for {target}.'
                )
                continue
            tissue_seg, _ = self.load(tissue_seg_file)

            ventricles_distance_file = self.pipeline_dir.get_output_image(
                accession=accession,
                modality=target,
                image_type='VentriclesDist',
                resampling_origin='T1',
                resampling_target=target,
            )

            if not os.path.exists(ventricles_distance_file):
                ventricles_distance = None
            else:
                ventricles_distance, _ = self.load(ventricles_distance_file)

            modality_images = self.load_modality_images(accession, target)
            template_images = self.load_template_images(accession, target)

            summary_stats = self.get_lesion_stats(
                lesion_seg,
                tissue_seg,
                ventricles_distance,
                modality_images,
                template_images,
                voxel_volume,
            )

            if self.radiomics_enabled:
                radiomics = self.get_radiomics(skullstripped_file, lesion_seg_file)
                pd.Series(radiomics).to_csv(radiomics_file)

            if not os.path.exists(lesion_seg_comp_file):
                self.logger.info(
                    f'Lesion seg comp file for {target} is missing.'
                    f'Skipping individual lesion quantification for {target}.'
                )
            else:
                lesion_seg_comp, _ = self.load(lesion_seg_comp_file)
                number_of_lesions = int(np.max(lesion_seg_comp))
                individual_lesion_stats = [
                    self.get_lesion_stats(
                        lesion_seg_comp,
                        tissue_seg,
                        ventricles_distance,
                        modality_images,
                        template_images,
                        voxel_volume,
                        lesion_label=label,
                    )
                    for label in range(1, number_of_lesions + 1)
                ]

                individual_lesion_df = pd.DataFrame(individual_lesion_stats)
                individual_lesion_df.to_csv(individual_quantification_file)
                summary_stats.update(
                    self.get_summary_update_from_individual(individual_lesion_df)
                )
            pd.Series(summary_stats).to_csv(summary_quantification_file)
