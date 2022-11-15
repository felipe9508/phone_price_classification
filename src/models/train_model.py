import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import logging
from pathlib import Path


def main(input_filepath, output_filepath):
    """ Runs model training scripts to turn processed data from (../processed) into
        a machine learning model (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('making a ML model from processed data')

    data = pd.read_parquet(f"{input_filepath}/phone_data_clean.parquet")

    X = data.drop('price_range', axis=1)
    y = data['price_range']

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y.values.ravel())

    dump(decision_tree, f'{output_filepath}/decision_tree_model.joblib')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    input_url = f'{project_dir}\data\interim'
    output_url = f'{project_dir}\models'
    main(input_url, output_url)
