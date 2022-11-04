import logging
import os

from pyalfe.data_structure import PipelineDataDir, Modality
from pyalfe.inference import InferenceModel


class Segmentation:

    def __init__(self, inference_model: InferenceModel):
        self.inference_model = inference_model

    def process(self, images_list, output_list):
        self.inference_model.predict_cases(images_list, output_list)


class SingleModalitySegmentation(Segmentation):

    logger = logging.getLogger('SingleModalitySegmentation')

    def __init__(
        self,
        inference_model: InferenceModel,
        pipeline_dir: PipelineDataDir,
        modality: Modality,
        image_type_input: str='skullstripped',
        image_type_output: str='CNNAbnormalMap_seg',
        segmentation_dir: str='abnormalmap',
        overwrite: bool=True
    ):
        self.pipeline_dir = pipeline_dir
        self.modality = modality
        self.image_type_input = image_type_input
        self.image_type_output = image_type_output
        self.segmentation_dir = segmentation_dir
        self.overwrite = overwrite
        super(SingleModalitySegmentation, self).__init__(inference_model)

    def run(self, accession):
        image_dir = self.pipeline_dir.get_processed_image(
            accession, self.modality, image_type=self.image_type_input)
        if not os.path.exists(image_dir):
            self.logger.info(
                f'{image_dir} is missing.'
                f'Skipping {self.image_type_output} segmentation.')
            return
        output_dir = self.pipeline_dir.get_processed_image(
            accession=accession, modality=self.modality,
            image_type=self.image_type_output,
            sub_dir_name=self.segmentation_dir)
        if not self.overwrite and os.path.exists(output_dir):
            return
        self.process([(image_dir,)], [output_dir])


class MultiModalitySegmentation(Segmentation):

    logger = logging.getLogger('MultiModalitySegmentation')

    def __init__(
        self,
        inference_model: InferenceModel,
        pipeline_dir: PipelineDataDir,
        modality_list: list[Modality],
        output_modality: Modality,
        image_type_input: str='skullstripped',
        image_type_output: str='CNNAbnormalMap_seg',
        segmentation_dir: str='abnormalmap',
        overwrite: bool=True
    ):
        self.pipeline_dir = pipeline_dir
        self.modality_list = modality_list
        self.output_modality = output_modality
        self.image_type_input = image_type_input
        self.image_type_output = image_type_output
        self.segmentation_dir = segmentation_dir
        self.overwrite = overwrite
        super(MultiModalitySegmentation, self).__init__(inference_model)

    def run(self, accession):
        image_dir_list = []
        for modality in self.modality_list:
            image_dir = self.pipeline_dir.get_processed_image(
                accession, modality, image_type=self.image_type_input,
                resampling_target=self.output_modality)
            if not os.path.exists(image_dir):
                self.logger.info(
                    f'{image_dir} is missing.'
                    f'Skipping {self.image_type_output} segmentation.')
                return
            image_dir_list.append(image_dir)
            
        output_dir = self.pipeline_dir.get_processed_image(
            accession=accession,
            modality=self.output_modality,
            image_type=self.image_type_output,
            sub_dir_name=self.segmentation_dir)
        if not self.overwrite and os.path.exists(output_dir):
            return
        self.process([image_dir_list], [output_dir])
