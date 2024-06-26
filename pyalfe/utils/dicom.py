import binascii
import collections
import datetime
from typing import Optional

import pydicom

ImageMeta = collections.namedtuple(
    'ImageMeta',
    [
        'path',
        'series_uid',
        'manufacturer',
        'seq',
        'series_desc',
        'tr',
        'te',
        'flip_angle',
        'contrast_agent',
        'patient_orientation_vector',
        'slice_thickness',
        'echo_number',
        'date',
    ],
)


def get_dicom_metadata(dicom):
    return pydicom.filereader.dcmread(dicom, stop_before_pixels=False, force=True)


def robust_decode(bstring):
    return bstring.decode(errors='replace')


def vector_decode(bstring):
    string_vector = robust_decode(bstring).split('\\')
    return [eval(num) for num in string_vector]


def extract_value(header, tag, default=None, decoder=lambda x: x):
    if header.get_item(tag) is not None:
        value = header.get_item(tag).value
        if value is not None:
            return decoder(value)
    return default


def date_decode(bstring):
    string = robust_decode(bstring)
    return str(datetime.datetime.strptime(string, "%Y%m%d").date())


def extract_dicom_meta(dcm: str, series_uid: Optional[str] = None) -> ImageMeta:
    """extract relaevant informationfrom dicom header

    Parameters
    ----------
    dicom: str
        The path to .dcm file.
    series_uid: str | None, default None
        The series_uid where the dicom file was originated from.
        This is optional and for record keeping only. If not provided,
        the series_uid in the return object will be populated based on
        the series_uid in the dicom file.

    Returns
    -------
    ImageMeta
        Returns a ImageMeta object that representing
        some of the dicom file meta data
    """
    header = get_dicom_metadata(dcm)
    path = dcm
    if not series_uid:
        series_uid = extract_series_uid(header)
    manufacturer = extract_value(
        header, ('0008', '0070'), default='', decoder=robust_decode
    )
    if manufacturer == 'GE MEDICAL SYSTEMS':
        seq = extract_value(header, ('0019', '109C'), default='', decoder=robust_decode)
    else:
        seq = extract_value(header, ('0018', '0020'), default='', decoder=robust_decode)
    series_desc = extract_value(
        header, ('0008', '103E'), default='', decoder=robust_decode
    )
    tr = extract_value(header, ('0018', '0080'), decoder=eval)
    te = extract_value(header, ('0018', '0081'), decoder=eval)
    flip_angle = extract_value(header, ('0018', '1314'), decoder=eval)
    contrast_agent = extract_value(
        header, ('0018', '0010'), default='', decoder=robust_decode
    )
    patient_orientation_vector = extract_value(
        header, ('0020', '0037'), default='', decoder=vector_decode
    )
    slice_thickness = extract_value(header, ('0018', '0050'), decoder=eval)
    echo_number = extract_value(header, ('0018', '0086'), default=1, decoder=eval)
    date = extract_value(header, ('0008', '0020'), decoder=date_decode)
    return ImageMeta(
        path=path,
        series_uid=series_uid,
        manufacturer=manufacturer,
        seq=seq,
        series_desc=series_desc,
        tr=tr,
        te=te,
        flip_angle=flip_angle,
        contrast_agent=contrast_agent,
        patient_orientation_vector=patient_orientation_vector,
        slice_thickness=slice_thickness,
        echo_number=echo_number,
        date=date,
    )


def extract_echo_number(header, default=1):
    return extract_value(header, ('0018', '0086'), default=default, decoder=eval)


def extract_series_uid(header, default=None):
    return extract_value(
        header, ('0020', '000e'), default=default, decoder=robust_decode
    )


def get_max_echo_series_crc(series: list[str]) -> str:
    """Calculate a checksum string of the given DICOM series. Adapeted from
    https://git.fmrib.ox.ac.uk/fsl/fslpy/-/blob/main/fsl/data/dicom.py

    Parameters
    ----------
    series: list[str]
        A list containing the path to all the dicom files in a series directory.

    Returns
    -------
    str
        Returns the series CRC string
    """
    if len(series) == 0:
        raise ValueError("Cannot caculate CRC for an empty series.")

    series_uid = None
    max_echo = None
    for dcm in series:
        header = get_dicom_metadata(dcm)

        echo_number = extract_echo_number(header)

        if max_echo is None:
            max_echo = echo_number
            series_uid = extract_series_uid(header)
        elif max_echo < echo_number:
            max_echo = echo_number
            series_uid = extract_series_uid(header)

    if series_uid is None:
        return None

    crc32 = str(binascii.crc32(series_uid.encode()))

    if max_echo is not None and max_echo > 1:
        crc32 = f'{crc32}.{max_echo}'

    return crc32
