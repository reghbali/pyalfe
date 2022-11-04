from pyalfe.tasks.initialization import Initialization
from pyalfe.tasks.quantification import Quantification
from pyalfe.tasks.registration import CrossModalityRegistration, Resampling
from pyalfe.tasks.segmentation import SingleModalitySegmentation, \
    MultiModalitySegmentation
from pyalfe.tasks.skullstripping import Skullstripping
from pyalfe.tasks.t1_postprocessing import T1Postprocessing
from pyalfe.tasks.t1_preprocessing import T1Preprocessing


class PyALFEPipelineRunner(object):

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
            resampling: Resampling,
            quantification: Quantification):
        self.initialization = initialization
        self.skullstripping = skullstripping
        self.t1_preprocessing = t1_preprocessing
        self.cross_modality_registration = cross_modality_registration
        self.flair_segmentation = flair_segmentation
        self.enhancement_segmentation = enhancement_segmentation
        self.tissue_segmentation = tissue_segmentation
        self.t1_postprocessing = t1_postprocessing
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
        self.resampling.run(accession)
        self.quantification.run(accession)