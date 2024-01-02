import io
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional


class InferenceModel(ABC):
    """The parent class for all inference models."""

    @abstractmethod
    def predict_cases(
        self, input_image_tuple_list: list[tuple], output_list: list
    ) -> None:
        """

        Parameters
        ----------
        input_image_tuple_list
        output_list

        Returns
        -------

        """
        pass


class NNUnet(InferenceModel):
    def __init__(
        self,
        model_dir: str,
        fold: int,
        n_threads_preprocessing: int = 6,
        n_threads_save: int = 1,
    ) -> None:
        self.model_dir = model_dir
        self.fold = fold
        self.n_threads_preprocessing = n_threads_preprocessing
        self.n_threads_save = n_threads_save

    def predict_cases(self, input_images, output):

        if os.path.exists(output):
            os.remove(output)
        text_trap = io.StringIO()
        sys.stdout = text_trap
        from nnunet.inference.predict import predict_cases_fast

        sys.stdout = sys.__stdout__
        predict_cases_fast(
            self.model_dir,
            [input_images],
            [output],
            folds=self.fold,
            num_threads_nifti_save=self.n_threads_save,
            num_threads_preprocessing=self.n_threads_preprocessing,
        )


class NNUnetV2(InferenceModel):
    def __init__(
        self,
        model_dir: str,
        folds: Optional[tuple[str]] = None,
        n_threads_preprocessing: int = 6,
        n_threads_save: int = 1,
    ) -> None:
        self.model_dir = model_dir
        self.folds = folds
        self.n_threads_preprocessing = n_threads_preprocessing
        self.n_threads_save = n_threads_save

    def predict_cases(self, input_images, output):
        if os.path.exists(output):
            os.remove(output)
        text_trap = io.StringIO()
        sys.stdout = text_trap
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        sys.stdout = sys.__stdout__

        predictor = nnUNetPredictor()
        predictor.initialize_from_trained_model_folder(
            self.model_dir,
            use_folds=self.folds,
            checkpoint_name='checkpoint_final.pth',
        )
        predictor.predict_from_files(
            [input_images],
            [output],
            num_processes_preprocessing=self.n_threads_preprocessing,
            num_processes_segmentation_export=self.n_threads_save,
        )
