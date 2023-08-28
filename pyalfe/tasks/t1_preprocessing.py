import os
import logging

from pyalfe.data_structure import PipelineDataDir, Modality
from pyalfe.image_processing import ImageProcessor


class T1Preprocessing:
    """This task preforms neck trimming and upsampling on the T1 image
    if needed.

    Attributes
    ----------
    image_processor: ImageProcessor
        The image processor object.
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    overwrite: bool
        Whether to overwrite any existing output image. Default is True.
    """

    logger = logging.getLogger('T1Preprocessing')

    def __init__(
        self,
        image_processor: ImageProcessor,
        pipeline_dir: PipelineDataDir,
        overwrite=True,
    ):
        self.pipeline_dir = pipeline_dir
        self.image_processor = image_processor
        self.overwrite = overwrite

    def run(self, accession):
        t1ss = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='skullstripped'
        )
        if not os.path.exists(t1ss):
            self.logger.info(
                'T1 skullstripped image is missing. Skipping T1Preprocessing.'
            )
            return
        trim_output = self.pipeline_dir.get_output_image(
            accession, 'T1', image_type='trim'
        )
        if self.overwrite or not os.path.exists(trim_output):
            self.image_processor.trim_largest_comp(t1ss, trim_output, [15, 15, 15])
        output = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type='trim_upsampled'
        )
        if self.overwrite or not os.path.exists(output):
            dims = self.image_processor.get_dims(t1ss)
            upfactors = []
            for dim in dims:
                if dim < 50:
                    upfactors.append(500)
                else:
                    upfactors.append(100)

            self.image_processor.resample_new_dim(trim_output, output, *upfactors)
