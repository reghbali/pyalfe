import configparser
import logging
import os
from pathlib import Path

import click


DEFAULT_CFG = os.path.expanduser(os.path.join('~', '.config', 'pyalfe', 'config.ini'))
DEFAULT_MODALITIES = ['T1', 'T1Post', 'FLAIR', 'T2', 'ADC', 'SWI', 'CBF']
DEFAULT_TARGETS = ['FLAIR', 'T1Post']
DEFAUlT_DOMINANT_TISSUE = 'white_matter'
DEFAULT_IMAGE_PROCESSOR = 'nilearn'
DEFAULT_IMAGE_REGISTRATION = 'greedy'
DEFUALT_DATA_DIR_STRUCTURE = 'alfe'
DEFAUlT_TISSUE_SEGMENTATION = 'prior'
DEFAULT_OVERWRITE = True


logging.basicConfig(level=logging.DEBUG)


@click.group()
def main():
    pass


@main.command()
@click.argument('assets', nargs=-1)
def download(assets):
    """
    Downloads assets such as models.

    \b
    Example
    -------
    pyalfe download models
    """
    import huggingface_hub
    from pyalfe.models import MODELS_PATH, models_url
    from pyalfe.tools import C3D_PATH, GREEDY_PATH, c3d_url, greedy_url
    from pyalfe.utils import download_archive, extract_binary_from_archive

    for asset in assets:
        if asset == 'models':
            archive_path = huggingface_hub.snapshot_download(
                repo_id=models_url, local_dir=MODELS_PATH
            )
        elif asset == 'greedy':
            archive_path = download_archive(
                url=greedy_url,
                download_dir=os.path.dirname(GREEDY_PATH),
            )
            extract_binary_from_archive(
                archive_path=archive_path,
                dst=os.path.dirname(GREEDY_PATH),
                binary_name='greedy',
            )
            if os.path.exists(archive_path):
                os.remove(archive_path)
        elif asset == 'c3d':
            archive_path = download_archive(
                url=c3d_url,
                download_dir=os.path.dirname(C3D_PATH),
            )
            extract_binary_from_archive(
                archive_path=archive_path,
                dst=os.path.dirname(C3D_PATH),
                binary_name='c3d',
            )
            if os.path.exists(archive_path):
                os.remove(archive_path)
        else:
            click.print(f'asset {asset} is not recognized.')


def _run(
    accession: str,
    config: str = None,
    input_dir: str = None,
    output_dir: str = None,
    modalities: str = None,
    targets: str = None,
    dominant_tissue: str = None,
    image_processor: str = None,
    image_registration: str = None,
    data_dir_structure: str = None,
    tissue_segmentation: str = None,
    overwrite: bool = True,
):
    """Runs the pipeline for an accession number.

    Parameters
    ----------
    accession : str
        the accession number for which you want to run the pipeline.
    config : str
        the path to the config file.
    input_dir : str
        the path to the directory containing input images
    output_dir : str
        the path to the directory containing output images
    modalities : str
        comma separated modalities
    targets : str
        comma separated target modalities
    dominant_tissue : str
        dominant tissue
    image_processor : str
        image processor that is used by the pipeline.
    image_registration : str
        image registration that is used by the pipeline.
    data_dir_structure: str
        the data directory structure, it can be 'alfe' or 'bids'.
    tissue_segmentation: str
        the tissue segmentation method
    overwrite : bool
        if True, the pipeline overwrites existing output images.

    Returns
    -------
    None
    """
    from pyalfe.containers import PipelineContainer

    container = PipelineContainer()

    if config:
        container.config.from_ini(config)
    else:
        container.config.from_dict({'options': {}})

    options = container.config.options

    if input_dir:
        options.input_dir = input_dir

    if output_dir:
        options.output_dir = output_dir

    if modalities:
        if isinstance(modalities, str):
            options.modalities = modalities
        else:
            options.modalities = ','.join(modalities)

    if targets:
        if isinstance(targets, str):
            options.targets = targets
        else:
            options.targets = ','.join(targets)

    if dominant_tissue:
        options.dominant_tissue = dominant_tissue

    if image_processor:
        options.image_processor = image_processor

    if image_registration:
        options.image_registration = image_registration

    if data_dir_structure:
        options.data_dir_structure = data_dir_structure

    if tissue_segmentation:
        options.tissue_segmentation = tissue_segmentation

    options.overwrite_images = overwrite

    container.init_resources()
    pipeline_runner = container.pyalfe_pipeline_runner

    pipeline_runner.run(accession)


def run(
    accession: str,
    input_dir: str,
    output_dir: str,
    modalities: str = DEFAULT_MODALITIES,
    targets: str = DEFAULT_TARGETS,
    dominant_tissue: str = DEFAUlT_DOMINANT_TISSUE,
    image_processor: str = DEFAULT_IMAGE_PROCESSOR,
    image_registration: str = DEFAULT_IMAGE_REGISTRATION,
    data_dir_structure: str = DEFUALT_DATA_DIR_STRUCTURE,
    tissue_segmentation: str = DEFAUlT_TISSUE_SEGMENTATION,
    overwrite: bool = DEFAULT_OVERWRITE,
) -> None:
    """Runs the pipeline for an accession number.

    Parameters
    ----------
    accession : str
        the accession number for which you want to run the pipeline.
    config : str, default: :attr:`DEFAULT_CFG`
        the path to the config file.
    input_dir : str
        the path to the directory containing input images
    output_dir : str
        the path to the directory containing output images
    modalities : str, default: :attr:`DEFAULT_MODALITIES`
        comma separated modalities
    targets : str, default: :attr:`DEFAULT_TARGETS`
        comma separated target modalities
    dominant_tissue : str, default: :attr:`DEFAUlT_DOMINANT_TISSUE`
        dominant tissue
    image_processor : str, default: :attr:`DEFAULT_IMAGE_PROCESSOR`
        image processor that is used by the pipeline.
    image_registration : str, default: :attr:`DEFAULT_IMAGE_REGISTRATION`
        image registration that is used by the pipeline.
    data_dir_structure: str, default: :attr:`DEFUALT_DATA_DIR_STRUCTURE`
        the data directory structure, it can be 'alfe' or 'bids'.
    tissue_segmentation: str, default: :attr:`DEFAULT_TISSUE_SEGMENTATION`
        the tissue segmentation method, it can be 'prior' or 'synthseg'
    overwrite : bool, default: :attr:`DEFAULT_OVERWRITE`
        if True, the pipeline overwrites existing output images.

    Returns
    -------
    None
    """
    _run(
        accession=accession,
        input_dir=input_dir,
        output_dir=output_dir,
        modalities=modalities,
        targets=targets,
        dominant_tissue=dominant_tissue,
        image_processor=image_processor,
        image_registration=image_registration,
        data_dir_structure=data_dir_structure,
        tissue_segmentation=tissue_segmentation,
        overwrite=overwrite,
    )


@main.command(name='run')
@click.argument('accession')
@click.option(
    '-c',
    '--config',
    default=DEFAULT_CFG,
    help='The path to the config file. Run `pyalfe configure` '
    f'to create the config file. Default: {DEFAULT_CFG}',
)
@click.option(
    '-id',
    '--input-dir',
    type=click.Path(resolve_path=True),
    help='The path to the directory containing input images',
)
@click.option(
    '-od',
    '--output-dir',
    type=click.Path(resolve_path=True),
    help='The path to the directory containing output images',
)
@click.option(
    '-m',
    '--modalities',
    help='Comma separated modalities. Example: T1,T1Post,FLAIR,CBF',
)
@click.option(
    '-t', '--targets', help='Comma separated target modalities. Example: T1Post,FLAIR'
)
@click.option(
    '-dt',
    '--dominant_tissue',
    type=click.Choice(['white_matter', 'gray_matter', 'auto'], case_sensitive=False),
    help='Dominant tissue for the lesions and abnormalities.',
)
@click.option(
    '-ip',
    '--image-processor',
    type=click.Choice(['c3d', 'nilearn'], case_sensitive=False),
    help='Image processor that is used by the pipeline',
)
@click.option(
    '-ir',
    '--image-registration',
    type=click.Choice(['greedy', 'ants'], case_sensitive=False),
    help='Image registration that is used by the pipeline',
)
@click.option(
    '-dds',
    '--data-dir-structure',
    type=click.Choice(['alfe', 'bids'], case_sensitive=False),
    help='The data directory structure',
)
@click.option(
    '-ts',
    '--tissue-segmentation',
    type=click.Choice(['prior', 'synthseg'], case_sensitive=False),
    help='The tissue segmentation tool used by the pipeline',
)
@click.option(
    '-ow/-now',
    '--overwrite/--no-overwrite',
    default=True,
    help='Whether to overwrite output images',
)
def run_command(
    accession: str,
    config: str,
    input_dir: str,
    output_dir: str,
    modalities: str,
    targets: str,
    dominant_tissue: str,
    image_processor: str,
    image_registration: str,
    data_dir_structure: str,
    tissue_segmentation: str,
    overwrite: bool,
) -> None:
    """Runs the pipeline for an accession number."""
    _run(
        accession=accession,
        config=config,
        input_dir=input_dir,
        output_dir=output_dir,
        modalities=modalities,
        targets=targets,
        dominant_tissue=dominant_tissue,
        image_processor=image_processor,
        image_registration=image_registration,
        data_dir_structure=data_dir_structure,
        tissue_segmentation=tissue_segmentation,
        overwrite=overwrite,
    )


@main.command()
def configure():
    """Configures the pipeline through a series of prompts."""
    input_dir = click.prompt(
        'Enter input image directory (press enter to skip)',
        default='',
        type=click.Path(resolve_path=True),
    )
    output_dir = click.prompt(
        'Enter output image directory (press enter to skip)',
        default='',
        type=click.Path(resolve_path=True),
    )
    modalities = click.prompt(
        'Enter modalities separated by comma (press enter for default)',
        default=','.join(DEFAULT_MODALITIES),
        type=str,
    )
    targets = click.prompt(
        'Enter target modalities separated by comma (press enter for default)',
        default=','.join(DEFAULT_TARGETS),
        type=str,
    )
    dominant_tissue = click.prompt(
        'Enter the dominant tissue for the lesions',
        type=click.Choice(['white_matter', 'gray_matter', 'auto']),
        default=DEFAUlT_DOMINANT_TISSUE,
    )
    image_processor = click.prompt(
        'image processor to use (press enter for default)',
        type=click.Choice(['nilearn', 'c3d']),
        default=DEFAULT_IMAGE_PROCESSOR,
    )
    image_registration = click.prompt(
        'image registration to use (press enter for default)',
        type=click.Choice(['greedy', 'ants']),
        default=DEFAULT_IMAGE_REGISTRATION,
    )
    data_dir_structure = click.prompt(
        'data directory structure (press enter for default)',
        type=click.Choice(['alfe', 'bids']),
        default=DEFUALT_DATA_DIR_STRUCTURE,
    )
    tissue_segmentation = click.prompt(
        'tissue segmentation method (press enter for default)',
        type=click.Choice(['prior', 'synthseg']),
        default=DEFAUlT_TISSUE_SEGMENTATION,
    )
    config_path = click.prompt('config path', default=DEFAULT_CFG)
    config = configparser.ConfigParser()
    config['options'] = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'modalities': modalities,
        'targets': targets,
        'dominant_tissue': dominant_tissue,
        'image_processor': image_processor,
        'image_registration': image_registration,
        'data_dir_structure': data_dir_structure,
        'tissue_segmentation': tissue_segmentation,
    }

    config_parent_path = os.path.dirname(config_path)
    if not os.path.exists(config_parent_path):
        Path(config_parent_path).mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as conf:
        config.write(conf)


def _process_dicom(
    accession: str,
    dicom_dir: str,
    config: str = None,
    nifti_dir: str = None,
    data_dir_structure: str = None,
    overwrite: bool = True,
):
    """Processes a dicom study (Experimental). Detects modalities and converts to NIfTI.
    Organizes the NIfTIs so they can be used as input for the pipeline.

    The dicom_dir should be organized as:
    dicom_dir
    └─ accession (study)
        └─ series
            ├─ instance_0.dcm
            └─ instance_1.dcm

    Parameters
    ----------
    accession: str
        the accession number for which you want to process dicoms.
    dicom_dir: str
        the directory containing raw dicoms.
    config : str, default: :attr:`DEFAULT_CFG`
        the path to the config file.
    nifti_dir: str
        the directory where the NIfTI images will be written.
        If not provided, defaults to pipeline input dir in the config.
    data_dir_structure: str, default: :attr:`DEFUALT_DATA_DIR_STRUCTURE`
        the data directory structure, it can be 'alfe' or 'bids'.
    overwrite : bool, default: :attr:`DEFAULT_OVERWRITE`
        if True, the pipeline overwrites existing output images.

    Returns
    -------
    None

    """
    from pyalfe.containers import DicomProcessingContianer

    container = DicomProcessingContianer()

    if config:
        container.config.from_ini(config)
    else:
        container.config.from_dict({'options': {}})

    options = container.config.options

    options.dicom_dir = dicom_dir

    if nifti_dir:
        options.nifti_dir = nifti_dir
    elif options.input_dir:
        options.nifti_dir = options.input_dir

    if data_dir_structure:
        options.data_dir_structure = data_dir_structure

    options.overwrite_images = overwrite

    container.init_resources()
    pipeline_runner = container.dicom_processing_pipeline_runner

    pipeline_runner.run(accession)


def process_dicom(
    accession: str,
    dicom_dir: str,
    nifti_dir: str,
    data_dir_structure: str = DEFUALT_DATA_DIR_STRUCTURE,
    overwrite: bool = DEFAULT_OVERWRITE,
):
    _process_dicom(
        accession=accession,
        dicom_dir=dicom_dir,
        nifti_dir=nifti_dir,
        data_dir_structure=data_dir_structure,
        overwrite=overwrite,
    )


@main.command(name='process_dicom')
@click.argument('accession')
@click.argument('dicom_dir')
@click.option(
    '-c',
    '--config',
    default=DEFAULT_CFG,
    help='The path to the config file. '
    f'Run `pyalfe configure` to create the config file. Default: {DEFAULT_CFG}',
)
@click.option(
    '-nd',
    '--nifti-dir',
    help='The path where the NIfTI images will be written. '
    'If not provided, defaults to pipeline input dir in the config.',
)
@click.option(
    '-dds',
    '--data-dir-structure',
    type=click.Choice(['alfe', 'bids'], case_sensitive=False),
    help='The data directory structure',
)
@click.option(
    '-ow/-now',
    '--overwrite/--no-overwrite',
    default=True,
    help='Whether to overwrite output images',
)
def process_dicom_command(
    accession: str,
    dicom_dir: str,
    config: str,
    nifti_dir: str,
    data_dir_structure: str,
    overwrite: bool,
):
    """Processes a dicom study (Experimental). Detects modalities and converts to NIfTI.
    Organizes the NIfTIs so they can be used as input for the pipeline.
    """
    _process_dicom(
        accession=accession,
        dicom_dir=dicom_dir,
        config=config,
        nifti_dir=nifti_dir,
        data_dir_structure=data_dir_structure,
        overwrite=overwrite,
    )


if __name__ == '__main__':
    main()
