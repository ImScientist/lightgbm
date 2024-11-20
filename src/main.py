import os
import logging

import click

from preprocessing import preprocess_data
from optimization import hyperparameter_optimization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-6s [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
opj = os.path.join

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')

DATA_DIR_RAW = opj(OUTPUT_DIR, 'data', 'raw')
DATA_DIR_PREPROCESSED = opj(OUTPUT_DIR, 'data', 'preprocessed')


@click.group()
def cli():
    """ cli """


@cli.command('preprocess-data')
@click.option('--data-dir-raw', type=str, default=DATA_DIR_RAW)
@click.option('--data-dir-preprocessed', type=str, default=DATA_DIR_PREPROCESSED)
def preprocess_data_fn(data_dir_raw, data_dir_preprocessed):
    """ Store training data from BQ in GCS """

    preprocess_data(
        src_dir=data_dir_raw,
        dst_dir=data_dir_preprocessed)


@cli.command('hyperparameter-optimization')
@click.option('--data-dir-preprocessed', type=str)
@click.option('--storage', type=str)
@click.option('--study-name', type=str)
def hyperparameter_optimization_fn(data_dir_preprocessed, storage, study_name):
    """ Hyperparameter optimization """

    hyperparameter_optimization(
        data_dir=data_dir_preprocessed,
        storage=storage,
        study_name=study_name)


if __name__ == "__main__":
    cli()
