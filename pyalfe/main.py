import os
import pathlib
import tarfile
import requests

import click

from pyalfe.containers import Container

DEFAULT_CFG = pathlib.Path(__file__).with_name('config.ini')
MODELS_URL = ('https://ucsf.box.com/shared/static/'
              'a93ruk3m26mso38jm8gk3qs5iod0h90h.gz')


def download_models(
        models_url: str,
        models_dir: str,
        tar_file_name: str = 'models.tar.gz'
):
    tar_file_path = os.path.join('/tmp', tar_file_name)
    models = requests.get(models_url)

    with open(tar_file_path, 'wb') as file:
        file.write(models.content)

    with tarfile.open(tar_file_path) as tar_file:
        tar_file.extractall(models_dir)


@click.command()
@click.argument('accession')
@click.option(
    '-c', '--config',
    default=DEFAULT_CFG,
)
@click.option('--classified_dir')
@click.option('--processed_dir')
def main(accession: str, config: str, classified_dir: str, processed_dir: str):

    models_dir = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(models_dir, 'models')):
        click.echo('--------------------------------------------------------')
        click.echo('Inference models directory not found.'
                   ' Downloading the models. This may take a few minutes ...')
        download_models(models_url=MODELS_URL, models_dir=models_dir)
    container = Container()
    container.config.from_ini(config, required=True, envs_required=True)

    options = container.config.options()
    if classified_dir:
        options['classified_dir'] = classified_dir
    if processed_dir:
        options['processed_dir'] = processed_dir
    container.config.from_dict(options)

    container.init_resources()
    pipeline_runner = container.pipeline_runner()

    pipeline_runner.run(accession)


if __name__ == '__main__':
    main()
