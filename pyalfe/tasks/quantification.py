import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd

from pyalfe.data_structure import PipelineDataDir, Tissue, Modality


class Quantification(object):

    logger = logging.getLogger('Quantification')

    def __init__(
        self,
        pipeline_dir: PipelineDataDir,
        modalities_all,
        modalities_target,
        dominant_tissue=None,
        overwrite=True,
    ):
        self.modalities_all = modalities_all
        self.modalities_target = modalities_target
        self.pipeline_dir = pipeline_dir
        self.overwrite = overwrite
        self.dominant_tissue = dominant_tissue 

    @staticmethod
    def load_nii_gz(filename):
        return nib.load(filename)

    def load(self, filename):
        nibabel_image = self.load_nii_gz(filename)
        return (
            nibabel_image.get_fdata().flatten(),
            np.prod(nibabel_image.header.get_zooms()))

    def get_lesion_stats(
        self,
        lesion_seg,
        tissue_seg,
        ventricles_distance,
        modality_images,
        voxel_volume
    ):
        stats = {}
        lesion_indices = np.nonzero(lesion_seg)[0]

        volume = len(lesion_indices) * voxel_volume
        stats['total_lesion_volume'] = volume
        if volume == 0.0:
            return stats

        unique_counts = np.unique(
            tissue_seg[lesion_indices], return_counts=True)

        if self.dominant_tissue and self.dominant_tissue != 'auto':
            invalid_tissue = True
            for tissue in Tissue:
                tissue_id = int(tissue)
                tissue_name = tissue.name.lower()
                if (self.dominant_tissue == tissue_id or
                   self.dominant_tissue == tissue_name):
                    self.dominant_tissue_id = tissue_id
                    invalid_tissue = False
                    break
            if invalid_tissue:
                raise ValueError(
                    f'{self.dominant_tissue} is invalid.'
                    ' The valid options are: {list(Tissue)}')
        else:
            self.dominant_tissue_id = unique_counts[0][
                np.argmax(unique_counts[1])]

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
                tissue_seg == self.dominant_tissue_id, lesion_seg == 0)
            mean_signal = np.mean(modality_image[healthy_dom_tissue_indices])
            stats[f'relative_{modality_name}_signal'] = \
                np.mean(modality_image[lesion_indices]) / mean_signal
            if modality_name == Modality.ADC:
                stats['relative_min_adc_signal'] = \
                    np.min(modality_image[lesion_indices]) / mean_signal

        if ventricles_distance is not None:
            stats['average_dist_to_ventricles_(voxels)'] = np.mean(
                ventricles_distance[lesion_indices])
            stats['minimum_dist_to_Ventricles_(voxels)'] = np.min(
                ventricles_distance[lesion_indices])

        return stats

    def run(self, accession):
        for target in self.modalities_target:
            quantification_file = self.pipeline_dir.get_quantification_file(
                accession, target, 'SummaryLesionMeasures')
            if not self.overwrite and os.path.exists(quantification_file):
                continue

            lesion_seg_file = self.pipeline_dir.get_processed_image(
                accession=accession, modality=target,
                image_type='CNNAbnormalMap_seg',
                sub_dir_name='abnormalmap')
            if not os.path.exists(lesion_seg_file):
                print(lesion_seg_file, 'lesion_seg does not exist')
                self.logger.info(
                f'Lesion seg file for {target} is missing.'
                f'Skipping quantification for {target}.')
                continue

            lesion_seg, voxel_volume = self.load(lesion_seg_file)
            tissue_seg_file = self.pipeline_dir.get_processed_image(
                accession, target, image_type='tissue_seg',
                resampling_origin='T1', resampling_target=target)
            if not os.path.exists(tissue_seg_file):
                print('no tissue seg', tissue_seg_file)
                self.logger.info(
                    f'Tissue seg file for {target} is missing.'
                    f'Skipping quantification for {target}.')
                continue
            tissue_seg, _ = self.load(tissue_seg_file)
            
            ventricles_distance_file = self.pipeline_dir.get_processed_image(
                accession=accession, modality=target,
                image_type='VentriclesDist',
                resampling_origin='T1', resampling_target=target)
            if not os.path.exists(ventricles_distance_file):
                ventricles_distance = None
            else:
                ventricles_distance, _ = self.load(ventricles_distance_file)

            modality_images = {}
            for modality in self.modalities_all:
                modality_image_to_target_file = \
                    self.pipeline_dir.get_processed_image(
                    accession, modality, image_type='skullstripped',
                    resampling_target=target)
                if os.path.exists(modality_image_to_target_file):
                    modality_images[modality], _ = self.load(
                        modality_image_to_target_file)
                else:
                    modality_images[modality] = None
            stats = self.get_lesion_stats(
                lesion_seg, tissue_seg, ventricles_distance,
                modality_images, voxel_volume)
            pd.Series(stats).to_csv(quantification_file)
