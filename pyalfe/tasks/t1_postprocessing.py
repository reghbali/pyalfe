import logging
import os

from pyalfe.data_structure import PipelineDataDir, Tissue, Modality
from pyalfe.image_processing import ImageProcessor
from pyalfe.tasks import Task


class T1Postprocessing(Task):
    logger = logging.getLogger('T1Postprocessing')

    def __init__(
            self,
            image_processor: ImageProcessor,
            pipeline_dir: PipelineDataDir,
            overwrite=True
    ):
        self.image_processing = image_processor
        self.pipeline_dir = pipeline_dir
        self.overwrite = overwrite

    def run(self, accession):
        tissue_segmentation_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='tissue_seg')

        if not os.path.exists(tissue_segmentation_image):
            self.logger.info('T1 tissue segmentation is missing.'
                             'Skipping T1 postprocessing')
            return
        background_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='background')
        csf_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='csf')
        white_matter_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='white_matter')
        cortical_gray_matter_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='cortical_gray_matter')
        deep_gray_matter_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='deep_gray_matter')
        brain_stem_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='brain_stem')
        cerebellum_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, image_type='cerebellum')

        output_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, 'VentriclesSeg')

        output_dist_image = self.pipeline_dir.get_processed_image(
            accession, Modality.T1, 'VentriclesDist')

        if self.overwrite or not os.path.exists(output_image):

            temp_image = self.pipeline_dir.get_processed_image(
                accession, Modality.T1, image_type='ventricles_intermediate_temp')

            for tissue in Tissue:
                self.image_processing.threshold(
                    tissue_segmentation_image,
                    eval(f'{tissue.name.lower()}_image'),
                    int(tissue), int(tissue), 1, 0)

            self.image_processing.union(
                deep_gray_matter_image, white_matter_image, temp_image)
            self.image_processing.dilate(temp_image, 4, temp_image)
            self.image_processing.holefill(temp_image, temp_image)
            self.image_processing.mask(temp_image, csf_image, output_image)

            self.image_processing.dilate(
                cortical_gray_matter_image, 2, temp_image)
            self.image_processing.set_subtract(
                output_image, temp_image, output_image)

            self.image_processing.dilate(output_image, -1, temp_image)
            self.image_processing.largest_mask_comp(temp_image, output_image)

            self.image_processing.union(
                deep_gray_matter_image, output_image, output_image)
            self.image_processing.dilate(output_image, 3, output_image)
            self.image_processing.mask(output_image, csf_image, output_image)

            self.image_processing.largest_mask_comp(output_image, output_image)

            self.image_processing.dilate(
                cortical_gray_matter_image, 2, temp_image)
            self.image_processing.set_subtract(
                output_image, temp_image, output_image)
    
            self.image_processing.dilate(brain_stem_image, 5, temp_image)
            self.image_processing.set_subtract(
                output_image, temp_image, output_image)

            self.image_processing.largest_mask_comp(output_image, output_image)

            os.remove(temp_image)

        if self.overwrite or not os.path.exists(output_dist_image):
            self.image_processing.distance_transform(
                output_image, output_dist_image)
