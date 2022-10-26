from dependency_injector.wiring import Provide, inject

from pyalfe.containers import Container
from pyalfe.tasks.initialization import Initialization
from pyalfe.tasks.quantification import Quantification
from pyalfe.tasks.registration import CrossModalityRegistration, Resampling
from pyalfe.tasks.segmentation import SingleModalitySegmentation, \
    MultiModalitySegmentation
from pyalfe.tasks.skullstripping import Skullstripping
from pyalfe.tasks.t1_postprocessing import T1Postprocessing
from pyalfe.tasks.t1_preprocessing import T1Preprocessing


class PyALFEPipelineRunner(object):

    @inject
    def __init__(
            self,
            initialization: Initialization = Provide[Container.initialization],
            skullstripping: Skullstripping = Provide[Container.skullstripping],
            t1_preprocessing: T1Preprocessing = Provide[Container.t1_preprocessing],
            cross_modality_registration: CrossModalityRegistration = Provide[Container.cross_modality_registration],
            flair_segmentation: SingleModalitySegmentation = Provide[Container.flair_segmentation],
            enhancement_segmentation: MultiModalitySegmentation = Provide[Container.enhancement_segmentation],
            tissue_segmentation: SingleModalitySegmentation = Provide[Container.tissue_segmentation],
            t1_postprocessing: T1Postprocessing = Provide[Container.t1_postprocessing],
            resampling: Resampling = Provide[Container.resampling],
            quantification: Quantification = Provide[Container.quantification]):
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