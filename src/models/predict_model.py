import pandas as pd
import logging
from joblib import load
from pathlib import Path
from sklearn import model_selection

# libraries to import function from other folder
import sys
import os
sys.path.append(os.path.abspath('src/'))


def main(input_filepath, output_filepath, report_filepath):
    """ Runs model training scripts to turn processed data from (../processed) into
        a machine learning model (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('evaluating ML model')

    model = load(f'{output_filepath}/decision_tree_model.joblib')
    data = pd.read_parquet(f"{input_filepath}/phone_data_clean.parquet")

    X = data.drop('price_range', axis=1)
    y = data['price_range']

    k_fold = model_selection.KFold(n_splits=10)
    scoring = 'accuracy'
    score = (model_selection.cross_val_score(
        model, X, y.values.ravel(),  scoring=scoring, cv=k_fold))

    print(f"({score.mean()}, {score.std()})")

    with open(f'{report_filepath}/model_prediction.txt', 'w') as f:
        f.write(
            f"Validación Cruzada\n Media: {score.mean()} Desviación Estandar: {score.std()}")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    input_url = f'{project_dir}\data\interim'
    output_url = f'{project_dir}\models'
    reports_url = f'{project_dir}\\reports'

    main(input_url,
         output_url,
         reports_url)
