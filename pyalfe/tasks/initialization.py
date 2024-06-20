import logging
import os

from pyalfe.data_structure import PipelineDataDir, Modality
from pyalfe.tasks import Task


class Initialization(Task):
    """This task creates the output directories and links to the original
    images in the input directory for each modality.

    Attributes
    ----------
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    modalities: list[Modality]
        All the modalities for which output directories should be generated.
    overwrite: bool
        Whether to overwrite any existing output image. Default is True.
    """

    logger = logging.getLogger('Initialization')

    def __init__(
        self,
        pipeline_dir: PipelineDataDir,
        modalities: list[Modality],
        overwrite: bool = True,
    ) -> None:
        super().__init__()
        self.pipeline_dir = pipeline_dir
        self.modalities = modalities
        self.overwrite = overwrite

    def run(self, accession: str) -> None:
        self.logger.info('Running initialization task.')
        for modality in self.modalities:

            classified_image = self.pipeline_dir.get_input_image(accession, modality)
            if not os.path.exists(classified_image):
                self.logger.warning(
                    f'{modality} image is missing.'
                    f'Skipping initialization for {modality}'
                )
                continue
            processed_image = self.pipeline_dir.get_output_image(accession, modality)
            self.logger.info(f'processing {modality} for accession {accession}.')
            if os.path.exists(processed_image):
                if self.overwrite:
                    os.remove(processed_image)
                else:
                    continue
            os.symlink(classified_image, processed_image)
