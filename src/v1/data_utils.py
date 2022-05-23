from functools import lru_cache, wraps

import pandas as pd
import numpy as np 

from . import config as cfg
from ..api import read_entries
from ..v0.data_utils import df_copy_cache


@df_copy_cache
def load_api_data(serials=cfg.device_ids, preprocess=True, include_device_src=False):
    if isinstance(serials, str):
        serials = [serials]
    
    DFs = []
    for serial in serials:
        df = read_entries(cfg.get_api_csv_path(serial))
        DFs.append(df)
    df = pd.concat(DFs, ignore_index=True)
    if preprocess:
        df = preprocess_api_data(df)

    # For evaluation purposes remember where each observation came from.
    if include_device_src:
        # TODO This is a hacky way to do it. Maybe a better way?
        # 'device_src' is like: ([0, 0, 0, ..., 2, 2, 2], [29, 25, 31] -- device_id)
        # device_id should correspond to serials (but they are not the same)
        df.attrs['device_src'] = df['device_id'].factorize()
    return df

def preprocess_api_data(df):
    # TODO remove outliers, duplicates, missing data, unnecessary columns, etc...
    
    # Use only date range selected in config
    start_date, end_date = pd.to_datetime(cfg.load_date_range)
    df = df.loc[(df['inserted_at'].dt.date >= start_date.date()) & 
                (df['inserted_at'].dt.date <= end_date.date())]

    # TODO a better way?
    # Remove `luminlux` because not all devices seem to measure it
    # (but some measure it always)
    # `id` is the unique measurement id -> useless for modelling
    df = df.drop(columns=['luminlux', 'id'])

    # Remove columns which are mostly null
    df = df.dropna(axis=1, how='any', thresh=0.9 * len(df))

    df = remove_absurd_api_data(df)
    df = df.dropna(how='any', subset=['inserted_at', 'tempc', 'relhumperc', 'pressmbar'])
    df = df.set_index('inserted_at', drop=False)
    df = df.sort_index()
    return df

def remove_absurd_api_data(df):
    df = df.drop(df[(df['tempc'] > 60) | (df['tempc'] < -60)].index)
    return df

@df_copy_cache
def load_arso_data(preprocess=True):
    df = pd.read_csv(cfg.arso_csv_path, parse_dates=['date'])
    if preprocess:
        return preprocess_arso_data(df)
    else:
        return df

def preprocess_arso_data(df):
    # Use only date range selected in config
    start_date, end_date = pd.to_datetime(cfg.arso_date_range)
    df = df.loc[(df['date'].dt.date >= start_date.date()) & 
                (df['date'].dt.date <= end_date.date())]

    df = df.set_index('date', drop=False)
    df = df.sort_index()
    return df

def _merge_arso_csv_data():
    gt = []
    for p in cfg.data_path.glob('arso_vrbanski_plato_*.csv'):
        gt.append(pd.read_csv(p, header=0, usecols=[1,2,3,4], names=['station', 'date', 'tempc', 'relhumperc'], parse_dates=['date']))
        print(gt[-1].shape)
    gt = pd.concat(gt, ignore_index=True)
    gt = gt.drop_duplicates()
    gt.to_csv(cfg.arso_csv_path, index=False)
