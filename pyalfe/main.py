import pathlib

import click

from pyalfe.containers import Container
from pyalfe.pipeline import PyALFEPipelineRunner

DEFAULT_CFG = pathlib.Path(__file__).with_name('config.ini')


@click.command()
@click.argument('accession')
@click.option(
    '-c', '--config',
    default=DEFAULT_CFG,
)
def main(accession, config):
    container = Container()
    container.config.from_ini(config, required=True, envs_required=True)
    container.wire(modules=['pyalfe'])

    pipeline_runner = PyALFEPipelineRunner()
    pipeline_runner.run(accession)


if __name__ == '__main__':
    main()
