import configparser
import logging
import os
from pathlib import Path

import click

from pyalfe.containers import Container
from pyalfe.models import MODELS_PATH, models_url
from pyalfe.tools import C3D_PATH, GREEDY_PATH, c3d_url, greedy_url
from pyalfe.utils import download_archive, extract_binary_from_archive
from pyalfe.utils.archive import extract_tar

DEFAULT_CFG = os.path.expanduser(os.path.join('~', '.config', 'pyalfe', 'config.ini'))
# importlib.resources.files('pyalfe').joinpath('config.ini')

logging.basicConfig(level=logging.DEBUG)


@click.group()
def main():
    pass


@main.command()
@click.argument('assets', nargs=-1)
def download(assets):
    for asset in assets:
        if asset == 'models':
            archive_path = download_archive(
                url=models_url, download_dir=MODELS_PATH, archive_name='models.tar.gz'
            )
            extract_tar(archive_path, MODELS_PATH)
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
        else:
            click.print(f'asset {asset} is not recognized.')
        if os.path.exists(archive_path):
            os.remove(archive_path)


@main.command()
@click.argument('accession')
@click.option(
    '-c',
    '--config',
    default=DEFAULT_CFG,
)
@click.option('-m', '--modalities')
@click.option('-t', '--targets')
@click.option('-id', '--input-dir')
@click.option('-pd', '--output-dir')
@click.option(
    '-dt',
    '--dominant_tissue',
    default='white_matter',
    type=click.Choice(['white_matter', 'gray_matter', 'auto'], case_sensitive=False),
)
@click.option('-ow/-now', '--overwrite/--no-overwrite', default=True)
@click.option(
    '-ip',
    '--image-processor',
    type=click.Choice(['c3d', 'nilearn'], case_sensitive=False),
)
@click.option(
    '-ir',
    '--image-registration',
    type=click.Choice(['greedy', 'ants'], case_sensitive=False),
)
def run(
    accession: str,
    config: str,
    input_dir: str,
    output_dir: str,
    modalities: str,
    targets: str,
    dominant_tissue: str,
    image_processor: str,
    image_registration: str,
    overwrite: bool,
) -> None:
    """Runs the pipeline for an accession number.

    Parameters
    ----------
    accession : str
        the accession number for which you want to run the pipeline.
    config : str, default: ~/.config/pyalfe/config.ini
        the path to the config file.
    input_dir : str
        the path to the directory containing input input images
    outpu_dir : str
        the path to the directory containing output output images
    modalities : str
        comma separated modalities
    targets : str
        comma separated target modalities
    dominant_tissue : str, default='white_matter'
        dominant tissue
    image_processor : str, default=c3d
        image processor that is used by the pipeline.
    image_registration : str, default=greedy
        image registration that is used by the pipeline.
    overwrite : bool
        if True, the pipeline overwrites existing output images.

    Returns
    -------
    None
    """
    container = Container()
    container.config.from_ini(config, required=True, envs_required=True)

    options = container.config.options()

    if input_dir:
        options['input_dir'] = input_dir
    if output_dir:
        options['output_dir'] = output_dir
    if modalities:
        options['modalities'] = modalities
    if targets:
        options['targets'] = targets
    if dominant_tissue:
        options['dominant_tissue'] = dominant_tissue
    if image_processor:
        options['image_processor'] = image_processor
    if image_registration:
        options['image_registration'] = image_registration
    options['overwrite_images'] = overwrite

    container.config.from_dict(options)
    click.echo(options)

    container.init_resources()
    pipeline_runner = container.pipeline_runner()

    pipeline_runner.run(accession)


@main.command()
def configure():
    """
    Configure the pipeline through a series of prompts.

    Returns
    -------
    None
    """
    input_dir = click.prompt(
        'Enter input image directory', type=click.Path(exists=True)
    )
    output_dir = click.prompt(
        'Enter output image directory', type=click.Path(exists=True)
    )
    modalities = click.prompt(
        'Enter modalities separated by comma (press enter for default)',
        default='T1,T1Post,FLAIR,T2,ADC,SWI,CBF',
        type=str,
    )
    targets = click.prompt(
        'Enter target modalities separated by comma (press enter for default)',
        default='T1Post,FLAIR',
        type=str,
    )
    dominant_tissue = click.prompt(
        'Enter the dominant tissue for the lesions',
        type=click.Choice(['white_matter', 'gray_matter', 'auto']),
        default='white_matter',
    )
    image_processor = click.prompt(
        'image processor to use (press enter for default)',
        type=click.Choice(['c3d', 'nilearn']),
        default='c3d',
    )
    image_registration = click.prompt(
        'image registration to use (press enter for default)',
        type=click.Choice(['greedy', 'ants']),
        default='greedy',
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
    }

    config_parent_path = os.path.dirname(config_path)
    if not os.path.exists(config_parent_path):
        Path(config_parent_path).mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as conf:
        config.write(conf)


if __name__ == '__main__':
    main()
