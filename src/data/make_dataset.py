# -*- coding: utf-8 -*-
from pre_processing import process_dataset
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_clean = process_dataset(input_filepath, output_filepath)
    logger.info(f'Cantidad de datos guardada: {len(data_clean)}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    input_url = f'{project_dir}\data\\raw'
    output_url = f'{project_dir}\data\interim'
    main(input_url, output_url)
