import pandas as pd
import numpy as np


def process_dataset(input_filepath: str, output_filepath: str) -> pd.DataFrame:
    data = pd.read_csv(f'{input_filepath}/CellPhoneDS.csv')
    data_processed = process_data(data)
    data_processed.to_parquet(
        f'{output_filepath}/phone_data_clean.parquet', index=False)

    return data_processed


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    return (data
            .pipe(drop_duplicates)
            .pipe(process_outliers)
            .pipe(drop_null)
            .pipe(remove_columns))


def process_outliers(data_frame: pd.DataFrame) -> pd.DataFrame:

    data_frame['talk_time'].replace('??????', np.nan, inplace=True)
    data_frame['talk_time'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['talk_time'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['talk_time'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['talk_time'] = data_frame['talk_time'].astype('float')

    data_frame['battery_power'].replace(
        'nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['battery_power'].replace('??????', np.nan, inplace=True)
    data_frame['battery_power'] = data_frame['battery_power'].astype('float')

    data_frame['pc'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['pc'].replace('??????', np.nan, inplace=True)
    data_frame['pc'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['pc'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['pc'] = data_frame['pc'].astype('float')

    data_frame['three_g'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['three_g'].replace('??????', np.nan, inplace=True)
    data_frame['three_g'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['three_g'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['three_g'] = data_frame['three_g'].astype('category')

    data_frame['mobile_wt'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['mobile_wt'].replace('??????', np.nan, inplace=True)
    data_frame['mobile_wt'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['mobile_wt'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['mobile_wt'] = data_frame['mobile_wt'].astype('float')

    data_frame['px_width'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['px_width'].replace('??????', np.nan, inplace=True)
    data_frame['px_width'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['px_width'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['px_width'] = data_frame['px_width'].astype('float')

    data_frame['sc_h'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['sc_h'].replace('??????', np.nan, inplace=True)
    data_frame['sc_h'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['sc_h'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['sc_h'] = data_frame['sc_h'].astype('float')

    data_frame['sc_w'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['sc_w'].replace('??????', np.nan, inplace=True)
    data_frame['sc_w'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['sc_w'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['sc_w'] = data_frame['sc_w'].astype('float')

    data_frame['m_dep'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['m_dep'].replace('??????', np.nan, inplace=True)
    data_frame['m_dep'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['m_dep'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['m_dep'] = data_frame['m_dep'].astype('float')

    data_frame['touch_screen'].replace(
        'nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['touch_screen'].replace('??????', np.nan, inplace=True)
    data_frame['touch_screen'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['touch_screen'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['touch_screen'] = data_frame['touch_screen'].astype('category')

    data_frame['fc'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['fc'].replace('??????', np.nan, inplace=True)
    data_frame['fc'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['fc'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['fc'] = data_frame['fc'].astype('float')

    data_frame['four_g'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['four_g'].replace('??????', np.nan, inplace=True)
    data_frame['four_g'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['four_g'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['four_g'] = data_frame['four_g'].astype('category')

    data_frame['blue'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['blue'].replace('??????', np.nan, inplace=True)
    data_frame['blue'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['blue'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['blue'] = data_frame['blue'].astype('category')

    data_frame['n_cores'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['n_cores'].replace('??????', np.nan, inplace=True)
    data_frame['n_cores'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['n_cores'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['n_cores'] = data_frame['n_cores'].astype('category')

    data_frame['wifi'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['wifi'].replace('??????', np.nan, inplace=True)
    data_frame['wifi'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['wifi'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['wifi'] = data_frame['wifi'].astype('category')

    data_frame['dual_sim'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['dual_sim'].replace('??????', np.nan, inplace=True)
    data_frame['dual_sim'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['dual_sim'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['dual_sim'] = data_frame['dual_sim'].astype('category')

    data_frame['ram'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['ram'].replace('??????', np.nan, inplace=True)
    data_frame['ram'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['ram'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['ram'] = data_frame['ram'].astype('float')

    data_frame['int_memory'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['int_memory'].replace('??????', np.nan, inplace=True)
    data_frame['int_memory'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['int_memory'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['int_memory'] = data_frame['int_memory'].astype('float')

    data_frame['px_height'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['px_height'].replace('??????', np.nan, inplace=True)
    data_frame['px_height'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['px_height'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['px_height'] = data_frame['px_height'].astype('float')

    data_frame['clock_speed'].replace('nhbgvfrtd 56gyub', np.nan, inplace=True)
    data_frame['clock_speed'].replace('??????', np.nan, inplace=True)
    data_frame['clock_speed'].replace('5285988458456.0', np.nan, inplace=True)
    data_frame['clock_speed'].replace('-948961565145.0', np.nan, inplace=True)
    data_frame['clock_speed'] = data_frame['clock_speed'].astype('float')

    return data_frame


def remove_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.drop(['wifi', 'blue', 'n_cores', 'four_g', 'dual_sim', 'touch_screen', 'three_g', 'pc', 'sc_h', 'sc_w'], axis=1)


def drop_duplicates(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.drop_duplicates()


def drop_null(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.dropna()
