"""Global read-only constants."""

import pathlib

data_path = pathlib.Path(__file__).resolve().parents[2] / 'data/v0/'  # Set absolute path to your data folder

arso_csv_path = data_path / 'arso_vrbanski_plato.csv'

experiment_results_path = data_path / 'experiments'

master_serials = ["134225002", "134225010", "134225019"]

def get_api_csv_path(serial):
    return data_path / f'{serial}.csv'
