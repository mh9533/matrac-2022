"""Global read-only constants."""

import pathlib

data_path = pathlib.Path(__file__).resolve().parents[2] / 'data/v1'  # Set absolute path to your data folder

arso_csv_path = data_path / 'arso_vrbanski_plato.csv'

experiment_results_path = data_path / 'experiments'

# Actually device ids (not serials, not uuids)
# Ordered: SmartLight 1, 2, ..., 10
device_ids = [8, 1, 9, 11, 10, 3, 7, 5, 12, 13]

# TODO set ranges per device to make more data available
# YYYY-MM-DD start and end date
download_date_range = ('2022-03-01', '2022-05-14')
load_date_range = ('2022-03-15', '2022-05-12')
arso_date_range = load_date_range

# Used for the interim report (March 2021 -- March 2022)
master_serials = ["134225002", "134225010", "134225019"]

api_secrets_path = pathlib.Path(__file__).resolve().with_name('secrets.txt')

def get_api_csv_path(serial):
    return data_path / f'{serial}.csv'
