import re

import numpy as np

from pyalfe.data_structure import Modality, Orientation


def detect_modality(series_desc, seq, contrast_agent, te):
    """This function applies a set of rules to detect the modality

    Parameters
    ----------
    series_desc: str
        Dicom Series Discription Attribute.
    seq: str
        Dicom Scanning Sequence
    contrast_agent: str
        Dicom Contrast/Bolus Agent
    te: float or int
        Echo Time

    Returns
    -------
    pyalfe.data_structure.Modality
        The detected modality
    """
    contrast = False
    if (
        contrast_agent
        and not re.search('(?i)^NO|PRE', contrast_agent)
        and re.search(
            '(?i)\\+C|GAD|GD|POST|HANCE|VIST|ABLAVAR|OMNISCAN|OPTIMARK|ARTIREM|DOTAREM|VUEWAY|ELUCIREM',
            contrast_agent,
        )
    ):
        contrast = True

    if re.search('(?!)CBF', series_desc):
        return Modality.CBF

    if re.search('(?!)ASL', series_desc):
        return Modality.ASL

    if re.search('(?!)EADC', series_desc):
        return Modality.EADC

    if re.search('(?!)ADC', series_desc) or (
        (
            re.search(r'(?!)AVERAGE\sDC', series_desc)
            or re.search('(?!)AVDC', series_desc)
        )
        and re.search('(?!)UNIVERSAL', series_desc)
    ):
        return Modality.ADC

    if re.search('(?i)APPARENT', series_desc):
        if re.search('(?i)EXP', series_desc):
            return Modality.EADC
        else:
            return Modality.ADC

    if re.search('(?i)EP', seq) or re.search('(?i)EP', series_desc):
        if re.search('(?i)DWI|DIFF|TRACE', series_desc):
            return Modality.DWI

    if re.search('(?i)PERFUSION', series_desc):
        return Modality.PERFUSION

    if re.search('(?i)FLAIR', series_desc) or re.search('(?i)FLAIR', seq):
        return Modality.FLAIR

    if re.search('(?i)SWI|SWAN', series_desc) or re.search('(?i)SWAN', seq):
        return Modality.SWI

    if re.search('(?i)T1|BRAVO', series_desc) and not re.search(
        '(?i)T1RHO', series_desc
    ):
        return Modality.T1Post if contrast else Modality.T1

    if re.search('(?i)SPGR|FLASH|FFE|TFE', series_desc):
        if te is None:
            return
        if te <= 6:
            return Modality.T1Post if contrast else Modality.T1
        if te >= 20:
            return Modality.SWI

    if re.search(r'(?i)MPGR|T2\*', series_desc):
        return Modality.SWI

    if re.search('(?i)TSE|FSE', series_desc):
        if te is None:
            return
        if te <= 30:
            return Modality.T1Post if contrast else Modality.T1
        if te >= 80:
            return Modality.T2


def detect_orientation(orientation_vector: list[float, int]):
    """This function detects the orientation of the image.

    Parameters
    ----------
    orientation_vector: list[float, int]
        Vector of length 6 representing the patient orientation vector

    Returns
    -------
    pyalfe.data_structure.Orientation
        The detected orientation
    """
    if len(orientation_vector) != 6:
        return

    rounded_vector = np.abs(np.round(orientation_vector))
    if (rounded_vector == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).all():
        return Orientation.AXIAL
    if (rounded_vector == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).all():
        return Orientation.SAGITTAL
    if (rounded_vector == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).all():
        return Orientation.CORONAL
