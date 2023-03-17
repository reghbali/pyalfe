import logging
import os
import shutil

from pyalfe.data_structure import PipelineDataDir
from pyalfe.image_processing import ImageProcessor
from pyalfe.inference import InferenceModel


class Segmentation(object):

    def __init__(
            self,
            inference_model: InferenceModel,
            image_processor: ImageProcessor):
        self.inference_model = inference_model
        self.image_processor = image_processor

    def predict(self, image_list, pred_list):
        self.inference_model.predict_cases(image_list, pred_list)

    def post_process(self, pred_list, mask, seg_list):
        if len(pred_list) != len(seg_list):
            raise ValueError(
                f'Input and output list should have the same length. '
                f'{len(pred_list)} != {len(seg_list)}')

        for pred, seg in zip(pred_list, seg_list):
            if mask:
                self.image_processor.mask(pred, mask, seg)
            else:
                shutil.copy(pred, seg)


class MultiModalitySegmentation(Segmentation):
    logger = logging.getLogger('MultiModalitySegmentation')

    def __init__(
            self,
            inference_model: InferenceModel,
            image_processor: ImageProcessor,
            pipeline_dir: PipelineDataDir,
            modality_list,
            output_modality,
            image_type_input: str='skullstripped',
            image_type_output: str='abnormal_seg',
            image_type_mask: str=None,
            segmentation_dir:str ='abnormalmap',
            overwrite: bool=True
    ):
        self.pipeline_dir = pipeline_dir
        self.modality_list = modality_list
        self.output_modality = output_modality
        self.image_type_input = image_type_input
        self.image_type_output = image_type_output
        self.image_type_mask = image_type_mask
        self.segmentation_dir = segmentation_dir
        self.overwrite = overwrite
        super(MultiModalitySegmentation, self).__init__(
            inference_model, image_processor)

    def run(self, accession):
        image_path_list = []

        for modality in self.modality_list:
            if modality != self.output_modality:
                resampling_target = self.output_modality
            else:
                resampling_target = None

            image_path = self.pipeline_dir.get_processed_image(
                accession, modality, image_type=self.image_type_input,
                resampling_target=resampling_target)
            if not os.path.exists(image_path):
                self.logger.info(
                    f'{image_path} is missing.'
                    f'Skipping {self.image_type_output} segmentation.')
                return
            image_path_list.append(image_path)

        pred_path = self.pipeline_dir.get_processed_image(
            accession=accession,
            modality=self.output_modality,
            image_type=f'{self.image_type_output}_pred',
            sub_dir_name=self.segmentation_dir)

        if self.overwrite or not os.path.exists(pred_path):
            self.predict([image_path_list], [pred_path])

        if self.image_type_mask:
            mask_path = self.pipeline_dir.get_processed_image(
                accession, self.output_modality,
                image_type=self.image_type_mask)
        else:
            mask_path = None

        output_path = self.pipeline_dir.get_processed_image(
            accession=accession, modality=self.output_modality,
            image_type=self.image_type_output,
            sub_dir_name=self.segmentation_dir)
        if self.overwrite or not os.path.exists(output_path):
            self.post_process([pred_path], mask_path, [output_path])


class SingleModalitySegmentation(MultiModalitySegmentation):
    logger = logging.getLogger('SingleModalitySegmentation')

    def __init__(
            self,
            inference_model: InferenceModel,
            image_processor: ImageProcessor,
            pipeline_dir: PipelineDataDir,
            modality,
            image_type_input:str ='skullstripped',
            image_type_output:str ='abnormal_seg',
            image_type_mask:str =None,
            segmentation_dir:str ='abnormalmap',
            overwrite:bool =True
    ):
        super(SingleModalitySegmentation, self).__init__(
            inference_model=inference_model, image_processor=image_processor,
            pipeline_dir=pipeline_dir,
            modality_list=[modality], output_modality=modality,
            image_type_input=image_type_input,
            image_type_output=image_type_output,
            image_type_mask=image_type_mask, segmentation_dir=segmentation_dir,
            overwrite=overwrite)
