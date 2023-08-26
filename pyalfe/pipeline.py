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


class PyALFEPipelineRunner:
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
        self.initialization = initialization
        self.skullstripping = skullstripping
        self.t1_preprocessing = t1_preprocessing
        self.cross_modality_registration = cross_modality_registration
        self.flair_segmentation = flair_segmentation
        self.enhancement_segmentation = enhancement_segmentation
        self.tissue_segmentation = tissue_segmentation
        self.t1_postprocessing = t1_postprocessing
        self.t1_registration = t1_registration
        self.resampling = resampling
        self.quantification = quantification

    def run(self, accession) -> None:
        self.initialization.run(accession)
        self.skullstripping.run(accession)
        self.t1_preprocessing.run(accession)
        self.cross_modality_registration.run(accession)
        self.flair_segmentation.run(accession)
        self.enhancement_segmentation.run(accession)
        self.tissue_segmentation.run(accession)
        self.t1_postprocessing.run(accession)
        self.t1_registration.run(accession)
        self.resampling.run(accession)
        self.quantification.run(accession)
