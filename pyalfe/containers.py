import logging
import os

from dependency_injector import containers, providers

from pyalfe.data_structure import DefaultALFEDataDir, Modality
from pyalfe.image_processing import Convert3DProcessor, NilearnProcessor
from pyalfe.image_registration import GreedyRegistration, AntsRegistration
from pyalfe.inference import NNUnet
from pyalfe.pipeline import PyALFEPipelineRunner
from pyalfe.tasks.initialization import Initialization
from pyalfe.tasks.quantification import Quantification
from pyalfe.tasks.registration import CrossModalityRegistration, Resampling
from pyalfe.tasks.segmentation import SingleModalitySegmentation, \
    MultiModalitySegmentation
from pyalfe.tasks.skullstripping import Skullstripping
from pyalfe.tasks.t1_postprocessing import T1Postprocessing
from pyalfe.tasks.t1_preprocessing import T1Preprocessing


class Container(containers.DeclarativeContainer):

    logger = logging.getLogger('Container')
    config = providers.Configuration()

    pipeline_dir = providers.Singleton(
        DefaultALFEDataDir,
        processed=config.options.processed_dir,
        classified=config.options.classified_dir,
    )

    #if config.options.image_processor == 'c3d':
    #    if shutil.which('c3d'):
    #        image_processor = providers.Singleton(Convert3DProcessor)
    #    else:
    #        logger.warning('Image processor c3d was requested'
    #                       ' but the binary is not in the path.'
    #                       ' Falling back on nilearn processor')
    #        image_processor = providers.Singleton(NilearnProcessor)
    #elif config.options.image_processor == 'nilearn':
    #    image_processor = providers.Singleton(NilearnProcessor)
    #else:
    #    raise ValueError(f'Unknown image processor'
    #                     f' {config.options.image_processor}')
    image_processor = providers.Selector(
        config.options.image_processor,
        c3d=providers.Singleton(Convert3DProcessor),
        nilearn=providers.Singleton(NilearnProcessor),
    )

    #if config.options.image_registration == 'greedy':
    #    if shutil.which('greedy'):
    #        image_registration = providers.Singleton(GreedyRegistration)
    #    else:
    #        logger.warning('Image registration tool greedy was requested'
    #                       ' but the binary is not in the path.'
    #                       ' Falling back on PyAnts registration')
    #        image_registration = providers.Singleton(AntsRegistration)
    #elif config.options.image_registration == 'ants':
    #    image_registration = providers.Singleton(AntsRegistration)
    #else:
    #    raise ValueError('Unknown image registration'
    #                     ' {config.options.image_registration}')
    image_registration = providers.Selector(
        config.options.image_registration,
        greedy=providers.Factory(GreedyRegistration),
        ants=providers.Factory(AntsRegistration)
    )
    parent_dir = os.path.dirname(__file__)
    skullstripping_model = providers.Singleton(
        NNUnet,
        model_dir = os.path.join(
            parent_dir, 'models', 'nnunet',
            'Task502_SS', 'nnUNetTrainerV2__nnUNetPlansv2.1'),
        fold=1)
    flair_model= providers.Singleton(
        NNUnet,
        model_dir = os.path.join(
            parent_dir, 'models', 'nnunet',
            'Task500_FLAIR', 'nnUNetTrainerV2__nnUNetPlansv2.1'),
        fold=1
    )
    enhancement_model = providers.Singleton(
        NNUnet,
        model_dir = os.path.join(
            parent_dir, 'models', 'nnunet',
            'Task503_Enhancement', 'nnUNetTrainerV2__nnUNetPlansv2.1'),
        fold=1
    )
    tissue_model = providers.Singleton(
        NNUnet,
        model_dir = os.path.join(
            parent_dir, 'models', 'nnunet',
            'Task504_Tissue', 'nnUNetTrainerV2__nnUNetPlansv2.1'),
        fold=1
    )


    initialization = providers.Singleton(
        Initialization,
        pipeline_dir=pipeline_dir,
        modalities=config.options.modalities.as_(lambda s: s.split(',')),
        overwrite=config.options.override_images
    )

    skullstripping = providers.Singleton(
        Skullstripping,
        inference_model=skullstripping_model,
        image_processor=image_processor,
        pipeline_dir=pipeline_dir,
        modalities=config.options.modalities.as_(lambda s: s.split(',')),
        overwrite=config.options.override_images
    )

    t1_preprocessing = providers.Singleton(
        T1Preprocessing,
        image_processor=image_processor,
        pipeline_dir=pipeline_dir,
        overwrite=config.options.override_images
    )

    cross_modality_registration = providers.Singleton(
        CrossModalityRegistration,
        image_registration=image_registration,
        pipeline_dir=pipeline_dir,
        modalities_all=config.options.modalities.as_(lambda s: s.split(',')),
        modalities_target=config.options.targets.as_(lambda s: s.split(',')),
        overwrite=config.options.override_images
    )

    flair_segmentation = providers.Singleton(
        SingleModalitySegmentation,
        inference_model=flair_model,
        pipeline_dir=pipeline_dir,
        modality=Modality.FLAIR,
        image_type_input='skullstripped',
        image_type_output='CNNAbnormalMap_seg',
        segmentation_dir='abnormalmap',
        overwrite=config.options.override_images
    )

    enhancement_segmentation = providers.Singleton(
        MultiModalitySegmentation,
        inference_model=enhancement_model,
        pipeline_dir=pipeline_dir,
        modality_list=[Modality.T1, Modality.T1Post],
        output_modality=Modality.T1Post,
        image_type_input='skullstripped',
        image_type_output='CNNAbnormalMap_seg',
        segmentation_dir='abnormalmap',
        overwrite=config.options.override_images
    )

    tissue_segmentation = providers.Singleton(
        SingleModalitySegmentation,
        inference_model=tissue_model,
        pipeline_dir=pipeline_dir,
        modality = Modality.T1,
        image_type_input='trim_upsampled',
        image_type_output='tissue_seg',
        segmentation_dir=None,
        overwrite=config.options.override_images
    )

    t1_postprocessing = providers.Singleton(
        T1Postprocessing,
        image_processor = image_processor,
        pipeline_dir = pipeline_dir,
        overwrite=config.options.override_images
    )

    resampling = providers.Singleton(
        Resampling,
        image_processor=image_processor,
        image_registration=image_registration,
        pipeline_dir=pipeline_dir,
        modalities_target=config.options.modalities.as_(lambda s: s.split(',')),
        overwrite=config.options.override_images
    )

    quantification = providers.Singleton(
        Quantification,
        pipeline_dir=pipeline_dir,
        modalities_all=config.options.modalities.as_(lambda s: s.split(',')),
        modalities_target=config.options.targets.as_(lambda s: s.split(',')),
        dominant_tissue =config.options.dominant_tissue,
        overwrite=config.options.override_quantification
    )

    pipeline_runner = providers.Singleton(
        PyALFEPipelineRunner,
        initialization=initialization,
        skullstripping=skullstripping,
        t1_preprocessing=t1_preprocessing,
        cross_modality_registration=cross_modality_registration,
        flair_segmentation=flair_segmentation,
        enhancement_segmentation=enhancement_segmentation,
        tissue_segmentation=tissue_segmentation,
        t1_postprocessing=t1_postprocessing,
        resampling=resampling,
        quantification=quantification
    )


