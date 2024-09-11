from pyalfe.tasks.dicom_processing import DicomProcessing
from pyalfe.tasks.initialization import Initialization
from pyalfe.tasks.quantification import Quantification
from pyalfe.tasks.registration import (
    CrossModalityRegistration,
    Resampling,
    T1Registration,
)
from pyalfe.tasks.segmentation import (
    SingleModalitySegmentation,
    MultiModalitySegmentation,
)
from pyalfe.tasks.skullstripping import Skullstripping
from pyalfe.tasks.t1_postprocessing import T1Postprocessing
from pyalfe.tasks.t1_preprocessing import T1Preprocessing


class PipelineRunner:
    def __init__(self, steps):
        self.steps = steps

    def run(self, accession: str) -> None:
        for step in self.steps:
            step.run(accession)


class PyALFEPipelineRunner(PipelineRunner):
    """The pyalfe pipeline runner.

    Attributes
    ----------
    initialization: Initialization
        The initialization task object.
    skullstripping: Skullstripping
        The skulstripping task object.
    t1_preprocessing: T1Preprocessing
        The t1 preprocessing task object.
    cross_modality_registration: CrossModalityRegistration
        The cross modality registration object.
    flair_segmentation: SingleModalitySegmentation
        The flair segmentation object.
    enhancement_segmentation: MultiModalitySegmentation
        The enhancement segmentation object.
    tissue_segmentation: SingleModalitySegmentation
        The tissue segmentation object.
    t1_postprocessing: T1Postprocessing
        The t1 postprocessing object.
    t1_registration: T1Registration
        The t1 registration object.
    resampling: Resampling
        The resampling object.
    quantification: Quantification
        The quantification object.
    """

    def __init__(
        self,
        initialization: Initialization,
        skullstripping: Skullstripping,
        t1_preprocessing: T1Preprocessing,
        cross_modality_registration: CrossModalityRegistration,
        flair_segmentation: SingleModalitySegmentation,
        enhancement_segmentation: MultiModalitySegmentation,
        tissue_segmentation: SingleModalitySegmentation,
        t1_postprocessing: T1Postprocessing,
        t1_registration: T1Registration,
        resampling: Resampling,
        quantification: Quantification,
    ):
        steps = [
            initialization,
            skullstripping,
            t1_preprocessing,
            cross_modality_registration,
            flair_segmentation,
            enhancement_segmentation,
            t1_registration,
            tissue_segmentation,
            t1_postprocessing,
            resampling,
            quantification,
        ]
        super().__init__(steps)


class DicomProcessingPipelineRunner(PipelineRunner):
    """The Dicom processing pipeline runner

    Attributes
    ----------
    dicom_processing: DicomProcessing
        The dicom processing object.
    """

    def __init__(self, dicom_processing: DicomProcessing):
        steps = [dicom_processing]
        super().__init__(steps)
