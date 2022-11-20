import configparser
import os
import importlib
from pathlib import Path

import click

from pyalfe.containers import Container
from pyalfe.tools import greedy_url, c3d_url
from pyalfe.models import models_url
from pyalfe.utils import download_archive, extract_binary_from_archive
from pyalfe.utils.archive import extract_tar

DEFAULT_CFG = os.path.expanduser(
    os.path.join('~', '.config', 'pyalfe', 'config.ini'))
#importlib.resources.files('pyalfe').joinpath('config.ini')


@click.group()
def main():
    pass


@main.command()
@click.argument('assets', nargs=-1)
def download(assets):
    for asset in assets:
        if asset == 'models':
            archive_path = download_archive(
                url=models_url,
                download_dir=importlib.resources.files('pyalfe.models'),
                archive_name='models.tar.gz')
            extract_tar(
                archive_path, importlib.resources.files('pyalfe.models'))
        elif asset == 'greedy':
            archive_path = download_archive(
                url=greedy_url,
                download_dir=importlib.resources.files('pyalfe.tools'),
            )
            extract_binary_from_archive(
                archive_path=archive_path,
                dst=importlib.resources.files('pyalfe.tools'),
                binary_name='greedy')
        elif asset == 'c3d':
            archive_path = download_archive(
                url=c3d_url,
                download_dir=importlib.resources.files('pyalfe.tools'),
            )
            extract_binary_from_archive(
                archive_path=archive_path,
                dst=importlib.resources.files('pyalfe.tools'),
                binary_name='c3d')
        else:
            click.print(f'asset {asset} is not recognized.')
        if os.path.exists(archive_path):
            os.remove(archive_path)


@main.command()
@click.argument('accession')
@click.option(
    '-c', '--config',
    default=DEFAULT_CFG,
)
@click.option('-cd', '--classified_dir')
@click.option('-pd', '--processed_dir')
@click.option('-m', '--modalities', default='T1,T1Post,FLAIR,T2,ADC')
@click.option('-t', '--targets', default='FLAIR,T1Post')
@click.option(
    '-dt', '--dominant_tissue',
    default='white_matter',
    type=click.Choice(
        ['white_matter', 'gray_matter', 'auto'], case_sensitive=False)
)
@click.option(
    '-oi', '--override_images',
    is_flag=True, show_default=True, default=True)
@click.option(
    '-oq', '--override_quantification',
    is_flag=True, show_default=True, default=True)
def run(
        accession: str,
        config: str,
        classified_dir: str,
        processed_dir: str,
        modalities: str,
        targets: str,
        dominant_tissue: str,
        override_images: bool,
        override_quantification: bool):

    container = Container()
    container.config.from_ini(config, required=True, envs_required=True)

    options = container.config.options()
    if classified_dir:
        options['classified_dir'] = classified_dir
    if processed_dir:
        options['processed_dir'] = processed_dir
    if modalities:
        options['modalities'] = modalities
    if targets:
        options['targets'] = targets
    if dominant_tissue:
        options['dominant_tissue'] = dominant_tissue
    if override_images:
        options['override_images'] = override_images
    if override_quantification:
        options['override_quantification'] = override_quantification

    container.config.from_dict(options)

    container.init_resources()
    pipeline_runner = container.pipeline_runner()

    pipeline_runner.run(accession)


@main.command()
def configure():

    classified_dir = click.prompt(
        'Enter classified image directory',
        type=click.Path(exists=True))
    processed_dir = click.prompt(
        'Enter processed image directory',
        type=click.Path(exists=True))
    modalities = click.prompt(
        'Enter modalities separated by comma', default='T1,T1Post,FLAIR,T2,ADC',
        type=str)
    targets = click.prompt(
        'Enter target modalities separated by comma', default='T1Post,FLAIR',
        type=str)
    overwrite_images = click.confirm(
        'Overwrite images?', default=True
    )
    overwrite_quantification = click.confirm(
        'Overwrite quantification?', default=True
    )
    image_processor = click.prompt(
        'image processor to use',
        type = click.Choice(['c3d', 'nilearn']),
        default='c3d')
    image_registration = click.prompt(
        'image registration to use',
        type = click.Choice(['greedy', 'ants']),
        default='greedy')
    config_path = click.prompt(
        'config path',
        default= DEFAULT_CFG)
    config = configparser.ConfigParser()
    config['options'] = {
        'classified_dir': classified_dir,
        'processed_dir': processed_dir,
        'modalities': modalities,
        'targets': targets,
        'overwrite_images': overwrite_images,
        'overwrite_quantification': overwrite_quantification,
        'image_processor': image_processor,
        'image_registration': image_registration
        }

    config_parent_path = os.path.dirname(config_path)
    if not os.path.exists(config_parent_path):
        Path(config_parent_path).mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as conf:
        config.write(conf)


if __name__ == '__main__':
    main()
