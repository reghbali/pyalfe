import logging
import os
import shutil
from pathlib import Path
from typing import Union

from pyalfe.data_structure import PipelineDataDir, Modality
from pyalfe.image_processing import ImageProcessor
from pyalfe.inference import InferenceModel
from pyalfe.roi import roi_dict
from pyalfe.tasks import Task


class Segmentation(Task):
    """This is the parent class for segmentation tasks."""

    def __init__(
        self, inference_model: InferenceModel, image_processor: ImageProcessor
    ):
        self.inference_model = inference_model
        self.image_processor = image_processor

    def predict(
        self,
        images: Union[list[Union[str, Path]], tuple[Union[str, Path]]],
        pred: Union[str, Path],
    ):
        """Creates a prediction image given the input images.

        Parameters
        ----------
        images: list[str or Path] or tuple[str or Path]
            A list or tuple of input images.
        pred: list[str or Path]
            Path to the output prediction segmentation image.

        Returns
        -------

        """
        self.inference_model.predict_cases([images], [pred])

    def post_process(self, pred, seg, mask=None):
        """Post processes the prediction segmentation image by applying an
        optional mask to create the final segmentation. For example, in a
        lesion segmentation task you can remove any lesion outside the brain
        by applying a brain mask.

        Parameters
        ----------
        pred: str or Path
            Path to the prediction segmentation image.
        mask: str or Path
            Path to the mask image.
        seg: str or Path
            Path to the output post-processed segmentation image.

        Returns
        -------

        """
        if mask:
            self.image_processor.mask(pred, mask, seg)
        else:
            shutil.copy(pred, seg)

    def label_segmentation_components(self, seg, comp):
        """Breaks down a binary segmentation into individual connected
        components.

        Parameters
        ----------
        seg: str or Path
            Path to the segmentation image.
        comp: str or Path
            Path to the output components image.

        Returns
        -------

        """
        self.image_processor.label_mask_comp(seg, comp)

    def run(self, accession: str) -> None:
        raise NotImplementedError


class MultiModalitySegmentation(Segmentation):
    """This task generates a segmentation from several input modalities.

    Parameters
    ----------
    inference_model: InferenceModel
        The inference model object.
    image_processor: ImageProcessor
        The image processor object.
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    modality_list: list[Modality]
        The modalities used for segmentation. For example, `[T1, T1Post]`.
    output_modality: Modality
        The modality where the segmentation output will be saved
        in its directory.
    image_type_input: str = 'skullstripped'
        The type of image that should be used as input.
        Default is `skullstripped`.
    image_type_output: str
        The type of output image that segmentation should be saved as.
        Default is `abnormal_seg`.
    image_type_mask: str = None
        The type of output image modality that should be used as a mask.
        Default is `None`.
    segmentation_dir: str = 'abnormalmap'
        The directory where the segmentaiton should be saved into.
        Default is `abnormalmap`.
    components: bool = False
        If `True`, creates a segmentation image that separately labels all
        the connected components of the original segmentation.
        Default is `False`.
    overwrite: bool = True
        Whether to overwrite existing output segmentation images.
        Default is True.
    """

    logger = logging.getLogger('MultiModalitySegmentation')

    def __init__(
        self,
        inference_model: InferenceModel,
        image_processor: ImageProcessor,
        pipeline_dir: PipelineDataDir,
        modality_list: list[str],
        output_modality: str,
        image_type_input: str = 'skullstripped',
        image_type_output: str = 'abnormal_seg',
        image_type_mask: str = None,
        segmentation_dir: str = 'abnormalmap',
        components: bool = False,
        overwrite: bool = True,
    ):
        self.pipeline_dir = pipeline_dir
        self.modality_list = modality_list
        self.output_modality = output_modality
        self.image_type_input = image_type_input
        self.image_type_output = image_type_output
        self.image_type_mask = image_type_mask
        self.segmentation_dir = segmentation_dir
        self.components = components
        self.overwrite = overwrite
        super().__init__(inference_model, image_processor)

    def run(self, accession):
        image_path_list = []

        for modality in self.modality_list:
            if modality != self.output_modality:
                resampling_target = self.output_modality
            else:
                resampling_target = None

            image_path = self.pipeline_dir.get_output_image(
                accession,
                modality,
                image_type=self.image_type_input,
                resampling_target=resampling_target,
            )
            if not os.path.exists(image_path):
                self.logger.info(
                    f'{image_path} is missing.'
                    f' Skipping {self.image_type_output} segmentation.'
                )
                return
            image_path_list.append(image_path)

        pred_path = self.pipeline_dir.get_output_image(
            accession=accession,
            modality=self.output_modality,
            image_type=f'{self.image_type_output}_pred',
            sub_dir_name=self.segmentation_dir,
        )

        if self.overwrite or not os.path.exists(pred_path):
            self.predict(image_path_list, pred_path)

        if self.image_type_mask:
            mask_path = self.pipeline_dir.get_output_image(
                accession, self.output_modality, image_type=self.image_type_mask
            )
        else:
            mask_path = None

        seg_path = self.pipeline_dir.get_output_image(
            accession=accession,
            modality=self.output_modality,
            image_type=self.image_type_output,
            sub_dir_name=self.segmentation_dir,
        )

        if self.overwrite or not os.path.exists(seg_path):
            self.post_process(pred_path, seg_path, mask_path)

        if self.components:
            comp_path = self.pipeline_dir.get_output_image(
                accession=accession,
                modality=self.output_modality,
                image_type=f'{self.image_type_output}_comp',
                sub_dir_name=self.segmentation_dir,
            )
            self.label_segmentation_components(seg_path, comp_path)


class SingleModalitySegmentation(MultiModalitySegmentation):
    """This task generates a segmentation from a single input modality.

        Parameters
    ----------
    inference_model: InferenceModel
        The inference model object.
    image_processor: ImageProcessor
        The image processor object.
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    modality: Modality
        The modality used for segmentation. For example, `[T1, T1Post]`.
    image_type_input: str = 'skullstripped'
        The type of image that should be used as input.
        Default is `skullstripped`.
    image_type_output: str
        The type of output image that segmentation should be saved as.
        Default is `abnormal_seg`.
    image_type_mask: str = None
        The type of output image modality that should be used as a mask.
        Default is `None`.
    segmentation_dir: str = 'abnormalmap'
        The directory where the segmentaiton should be saved into.
        Default is `abnormalmap`.
    components: bool = False
        If `True`, creates a segmentation image that separately labels all
        the connected components of the original segmentation.
        Default is `False`.
    overwrite: bool = True
        Whether to overwrite existing output segmentation images.
        Default is True.
    """

    logger = logging.getLogger('SingleModalitySegmentation')

    def __init__(
        self,
        inference_model: InferenceModel,
        image_processor: ImageProcessor,
        pipeline_dir: PipelineDataDir,
        modality,
        image_type_input: str = 'skullstripped',
        image_type_output: str = 'abnormal_seg',
        image_type_mask: str = None,
        segmentation_dir: str = 'abnormalmap',
        components: bool = False,
        overwrite: bool = True,
    ):
        super().__init__(
            inference_model=inference_model,
            image_processor=image_processor,
            pipeline_dir=pipeline_dir,
            modality_list=[modality],
            output_modality=modality,
            image_type_input=image_type_input,
            image_type_output=image_type_output,
            image_type_mask=image_type_mask,
            segmentation_dir=segmentation_dir,
            components=components,
            overwrite=overwrite,
        )


class TissueWithPriorSegementation(Segmentation):
    """This task generates tissue segmentation from T1 and a prior atlas based
    segementation.

    Parameters
    ----------
    inference_model: InferenceModel
        The inference model object.
    image_processor: ImageProcessor
        The image processor object.
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    image_type_input: str = 'skullstripped'
        The type of image that should be used as input.
        Default is `skullstripped`.
    image_type_output: str
        The type of output image that segmentation should be saved as.
        Default is `tissue_seg`.
    overwrite: bool = True
        Whether to overwrite existing output segmentation images.
        Default is True.
    """

    logger = logging.getLogger('TissueWithPriorSegementation')

    def __init__(
        self,
        inference_model: InferenceModel,
        image_processor: ImageProcessor,
        pipeline_dir: PipelineDataDir,
        image_type_input: str = 'trim_upsampled',
        image_type_output: str = 'tissue_seg',
        template_name: str = 'Tissue',
        overwrite: bool = True,
    ):
        super().__init__(inference_model, image_processor)
        self.pipeline_dir = pipeline_dir
        self.image_type_input = image_type_input
        self.image_type_output = image_type_output
        if template_name not in roi_dict:
            raise ValueError(f'{template_name} is not a valid ROI.')
        self.template_name = template_name
        self.overwrite = overwrite

    def run(self, accession):
        t1_image_path = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            image_type=self.image_type_input,
        )
        if not os.path.exists(t1_image_path):
            self.logger.info(
                f'{t1_image_path} is missing.' ' Skipping Tissue segmentation.'
            )
            return
        tissue_prior_path = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            resampling_origin=self.template_name,
            resampling_target=Modality.T1,
            sub_dir_name=roi_dict[self.template_name]['sub_dir'],
        )
        if not os.path.exists(tissue_prior_path):
            self.logger.info(
                f'{tissue_prior_path} is missing.' ' Skipping Tissue segmentation.'
            )
            return
        pred_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type=f'{self.image_type_output}_pred'
        )

        if self.overwrite or not os.path.exists(pred_path):
            self.predict([t1_image_path, tissue_prior_path], pred_path)

        seg_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type=self.image_type_output
        )

        if self.overwrite or not os.path.exists(seg_path):
            self.post_process(pred_path, seg_path)


class SynthSegTissueSegmentation(Segmentation):
    """This task generates tissue segmentation using freesurfer's synthseg

    Parameters
    ----------
    inference_model: InferenceModel
        The inference model object.
    image_processor: ImageProcessor
        The image processor object.
    pipeline_dir: PipelineDataDir
        The pipeline data directory object.
    image_type_input: str = 'skullstripped'
        The type of image that should be used as input.
        Default is `skullstripped`.
    image_type_output: str
        The type of output image that segmentation should be saved as.
        Default is `tissue_seg`.
    overwrite: bool = True
        Whether to overwrite existing output segmentation images.
        Default is True.
    """

    logger = logging.getLogger('SynthSegTissueSegmentation')

    def __init__(
        self,
        inference_model: InferenceModel,
        image_processor: ImageProcessor,
        pipeline_dir: PipelineDataDir,
        image_type_input: str = 'trim_upsampled',
        image_type_output: str = 'tissue_seg',
        overwrite: bool = True,
    ):
        super().__init__(inference_model, image_processor)
        self.pipeline_dir = pipeline_dir
        self.image_type_input = image_type_input
        self.image_type_output = image_type_output
        self.overwrite = overwrite

    def post_process(self, pred, seg, mask=None):
        self.image_processor.remap_labels(
            pred,
            {
                4: 1,
                5: 1,
                14: 1,
                15: 1,
                24: 1,
                43: 1,
                44: 1,
                3: 2,
                42: 2,
                17: 2,
                53: 2,
                2: 3,
                41: 3,
                10: 4,
                11: 4,
                12: 4,
                13: 4,
                18: 4,
                26: 4,
                28: 4,
                49: 4,
                50: 4,
                51: 4,
                52: 4,
                54: 4,
                58: 4,
                60: 4,
                16: 5,
                7: 6,
                8: 6,
                46: 6,
                47: 6,
            },
            seg,
        )

    def run(self, accession):
        t1_image_path = self.pipeline_dir.get_output_image(
            accession,
            Modality.T1,
            image_type=self.image_type_input,
        )
        if not os.path.exists(t1_image_path):
            self.logger.info(
                f'{t1_image_path} is missing.' ' Skipping Tissue segmentation.'
            )
            return

        pred_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type=f'{self.image_type_output}_pred'
        )

        if self.overwrite or not os.path.exists(pred_path):
            self.predict((t1_image_path,), pred_path)

        seg_path = self.pipeline_dir.get_output_image(
            accession, Modality.T1, image_type=self.image_type_output
        )

        if self.overwrite or not os.path.exists(seg_path):
            self.post_process(pred_path, seg_path)
