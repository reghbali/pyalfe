import logging
import os

from pyalfe.data_structure import PipelineDataDir, Modality
from pyalfe.tasks import Task


class Initialization(Task):
    logger = logging.getLogger('Initialization')

    def __init__(
        self,
        pipeline_dir: PipelineDataDir,
        modalities: list[Modality],
        overwrite: bool = True,
    ) -> None:
        self.pipeline_dir = pipeline_dir
        self.modalities = modalities
        self.overwrite = overwrite

    def run(self, accession: str) -> None:
        for modality in self.modalities:

            classified_image = self.pipeline_dir.get_classified_image(
                accession, modality
            )
            processed_image = self.pipeline_dir.get_processed_image(accession, modality)
            if not os.path.exists(classified_image):
                self.logger.warning(
                    f'{modality} image is missing.'
                    f'Skipping initialization for {modality}'
                )
                continue
            self.logger.info(f'processing {modality} for accession {accession}.')
            if os.path.exists(processed_image):
                if self.overwrite:
                    os.remove(processed_image)
                else:
                    continue
            os.symlink(classified_image, processed_image)
