import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, HuberRegressor, RANSACRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from ..experiment import BaseExperiment, stack_last_k, stack_recent_per_device, mean_features, sin_cos_time
from ..evaluate import get_closest_pairs
from .. import models
from . import config as cfg
from .data_utils import load_api_data, load_arso_data, df_copy_cache


@df_copy_cache
def load_tempc_data(*, serials, include_device_src=True):
    X_df = load_api_data(serials=serials, preprocess=True, include_device_src=include_device_src)
    y_df = load_arso_data(preprocess=True)

    X_df = X_df.loc[:, ['tempc']]  # Must be 2D, not a 1D Series
    y_df = y_df['tempc']

    closest_idx = get_closest_pairs(X_df, y_df)
    y_df = y_df.iloc[closest_idx]

    return X_df, y_df


class Experiment(BaseExperiment):
    def __init__(self, 
        *,
        device_serials=cfg.master_serials, 
        verbose=True, 
        random_seed=0, 
        n_splits=10,
        model_params=dict(), 
        **params
    ):
        super().__init__(device_serials=device_serials, 
            verbose=verbose, random_seed=random_seed, n_splits=n_splits, 
            model_params=model_params, **params)

    def get_results_path(self):
        return cfg.experiment_results_path / f'{self.__class__.__name__}.dump'

class IdentityTempcExperiment(Experiment):
    """Copy input temperature to output."""

    def load_data(self, include_device_src=True):
        return load_tempc_data(serials=self.params['device_serials'], 
            include_device_src=include_device_src)
    
    def create_model(self):
        return models.Identity()

class MeanTempcExperiment(IdentityTempcExperiment):
    """Predict MEAN temperature of merged master devices."""

    def create_model(self):
        return DummyRegressor(strategy='mean')

class MedianTempcExperiment(IdentityTempcExperiment):
    """Predict MEDIAN temperature of merged master devices."""

    def create_model(self):
        return DummyRegressor(strategy='median')

class MeanOffsetExperiment(IdentityTempcExperiment):
    """Predict by offsetting mean error."""

    def create_model(self):
        return models.MeanOffsetModel()

class LastKTempcMeanE(Experiment):
    """Predict by taking the mean of the last k tempc measurements."""

    def __init__(self, k=5, **params):
        super().__init__(**params)
        self.params['last_k'] = k 

    def load_data(self, include_device_src=True):
        X, y = load_tempc_data(serials=self.params['device_serials'], 
            include_device_src=include_device_src)
        X = stack_last_k(X, k=self.params['last_k'], select=['tempc'])
        return X, y

    def create_model(self):
        return models.ColumnMean()

class RecentDeviceTempcMeanE(Experiment):
    """Predict by taking the mean of the most recent tempc measurements across all devices."""

    def load_data(self, include_device_src=True):
        X, y = load_tempc_data(serials=self.params['device_serials'], 
            include_device_src=True)
        X = stack_recent_per_device(X)
        if not include_device_src:
            del X.attrs['device_src']
        return X, y

    def create_model(self):
        return models.ColumnMean()

class RollingTempcMeanE(Experiment):
    """Predict by taking the mean of a time window of tempc measurements."""

    def __init__(self, time_window='30min', **params):
        super().__init__(**params)
        self.params['time_window'] = time_window

    def load_data(self, include_device_src=True):
        X, y = load_tempc_data(serials=self.params['device_serials'], 
            include_device_src=include_device_src)
        X = mean_features(X, dt=self.params['time_window'])
        return X, y

    def create_model(self):
        return models.Identity()

class RollingMeanOffsetE(RollingTempcMeanE):
    """Predict by offsetting mean error of the mean of a time window of tempc measurements."""

    def create_model(self):
        return models.MeanOffsetModel()

class RFExperimentV0(IdentityTempcExperiment):
    """Random forest using tempc only."""

    def create_model(self):
        return RandomForestRegressor(random_state=self.params['random_seed'], **self.params['model_params'])

class RFExperimentV1(Experiment):
    """RF using basic weather data (tempc, pressmbar, relhumperc)."""

    def load_data(self, include_device_src=True):
        X = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X.loc[:, ['tempc', 'pressmbar', 'relhumperc']]
        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]

        return X, y      

    def create_model(self):
        return RandomForestRegressor(random_state=self.params['random_seed'], **self.params['model_params'])

class RFExperimentV2(Experiment):
    """RF using basic weather data (tempc, pressmbar, relhumperc) and device information."""

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc', 'pressmbar', 'relhumperc']]
        X['device_src'] = X_df['device_id'].factorize()[0]

        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]

        return X, y      

    def create_model(self):
        return RandomForestRegressor(random_state=self.params['random_seed'], **self.params['model_params'])

class RFExperimentV3(Experiment):
    """RF using tempc and measurement time information.
    
    Time is encoded with two features:
     - Hour: 24-hour float (e.g. 13.4)
     - Month & day: 12-month float (e.g. 0.5 == middle of January)
    """

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc']]

        dtime = X_df['inserted_at']
        X['hour'] = dtime.dt.hour + dtime.dt.minute / 60
        X['month'] = dtime.dt.month - 1 + dtime.dt.day / dtime.dt.days_in_month

        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]

        return X, y      

    def create_model(self):
        return RandomForestRegressor(random_state=self.params['random_seed'], **self.params['model_params'])

class RFExperimentV4(Experiment):
    """RF using tempc and measurement time information.
    
    Time is encoded with FOUR features:
     - Hour: 24-hour float (e.g. 13.4) -> (x, y) cartesian circle coordinates,
     - Month & day: 12-month float (e.g. 1.5 == middle of January) -> (x, y) cartesian circle coordinates
    """

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc']]
        time_df = sin_cos_time(X_df['inserted_at'])
        X = pd.concat([X, time_df], axis=1)
        X.attrs = X_df.attrs

        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]

        return X, y      

    def create_model(self):
        return RandomForestRegressor(random_state=self.params['random_seed'], **self.params['model_params'])

class RFExperimentV5(Experiment):
    """RF using basic weather data, circular measurement time, and device information."""

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc', 'pressmbar', 'relhumperc']]
        X['device_src'] = X_df['device_id'].factorize()[0]

        time_df = sin_cos_time(X_df['inserted_at'])
        X = pd.concat([X, time_df], axis=1)
        X.attrs = X_df.attrs

        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]

        return X, y

    def create_model(self):
        return RandomForestRegressor(random_state=self.params['random_seed'], **self.params['model_params'])

class RFExperimentV6(Experiment):
    """RF using basic weather data, linear measurement time, and device information."""

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc', 'pressmbar', 'relhumperc']]
        X['device_src'] = X_df['device_id'].factorize()[0]

        dtime = X_df['inserted_at']
        X['hour'] = dtime.dt.hour + dtime.dt.minute / 60
        X['month'] = dtime.dt.month - 1 + dtime.dt.day / dtime.dt.days_in_month

        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]

        return X, y      

    def create_model(self):
        return RandomForestRegressor(random_state=self.params['random_seed'], **self.params['model_params'])

class RF_AllVariablesE(Experiment):
    """RF using all variables with circular measurement time."""
    
    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        y = y['tempc']

        X = X_df.drop(columns=['inserted_at'])
        # X = X.dropna(axis=1)  # Remove all columns which contain a missing value (mostly 'useless' columns anyway...)
        X = X.fillna(method='ffill')  # Forward fill missing values
        X = X.dropna(axis=1)
        time_df = sin_cos_time(X_df['inserted_at'])
        X = pd.concat([X, time_df], axis=1)
        X.attrs = X_df.attrs
        if self.verbose: print(X.columns)

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]

        return X, y

    def create_model(self):
        return RandomForestRegressor(random_state=self.params['random_seed'], **self.params['model_params'])

class RidgeExperimentV1(Experiment):
    """ridge regression using basic weather data(tempc, pressmbar, relhumperc)"""

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc', 'pressmbar', 'relhumperc']]
        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]
        return X, y

    def create_model(self):
        return Ridge(random_state=self.params['random_seed'], **self.params['model_params'])

class RidgeExperimentV2(RidgeExperimentV1):
    """ridge regression using basic weather data(tempc, pressmbar, relhumperc) + data normalization"""

    def create_model(self):
        r = Ridge(random_state=self.params['random_seed'], **self.params['model_params'])
        return make_pipeline(StandardScaler(), r)

class RidgeExperimentV3(Experiment):
    """ridge regression using basic weather data(tempc, pressmbar, relhumperc) + hour and month"""

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc', 'pressmbar', 'relhumperc']]
        dtime = X_df['inserted_at']
        hour = dtime.dt.hour
        month = dtime.dt.month
        # one hot encode the hour and month
        hour_onehot = pd.get_dummies(hour)
        month_onehot = pd.get_dummies(month)
        X = pd.concat([X, hour_onehot, month_onehot], axis=1)
        X.attrs = X_df.attrs

        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]
        return X, y

    def create_model(self):
        return Ridge(random_state=self.params['random_seed'], **self.params['model_params'])

class RidgeExperimentV4(Experiment):
    """Ridge regression using rolling window averaged basic weather data and one-hot hour & month."""

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc', 'pressmbar', 'relhumperc']]
        X = mean_features(X, dt='30min')

        dtime = X_df['inserted_at']
        hour = dtime.dt.hour
        month = dtime.dt.month
        # one hot encode the hour and month
        hour_onehot = pd.get_dummies(hour)
        month_onehot = pd.get_dummies(month)
        X = pd.concat([X, hour_onehot, month_onehot], axis=1)
        X.attrs = X_df.attrs

        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]
        return X, y

    def create_model(self):
        # Don't standardize dummy variables!
        # slice(0,3) denotes the indices for ['tempc', 'pressmbar', 'relhumperc']
        transformer = make_column_transformer((StandardScaler(), slice(0,3)), remainder="passthrough")
        r = Ridge(random_state=self.params['random_seed'], **self.params['model_params'])
        return make_pipeline(transformer, r)

class RidgeExperimentV5(Experiment):
    """Ridge regression using rolling window averaged basic weather data and circular time."""
    # NOTE: polynomial regression overfits (deg=2)

    def load_data(self, include_device_src=True):
        X_df = load_api_data(serials=self.params['device_serials'], preprocess=True, include_device_src=include_device_src)
        y = load_arso_data(preprocess=True)

        X = X_df.loc[:, ['tempc', 'pressmbar', 'relhumperc']]
        X = mean_features(X, dt='30min')
        time_df = sin_cos_time(X_df['inserted_at'])
        X = pd.concat([X, time_df], axis=1)
        X.attrs = X_df.attrs

        y = y['tempc']

        closest_idx = get_closest_pairs(X, y)
        y = y.iloc[closest_idx]
        return X, y

    def create_model(self):
        r = Ridge(random_state=self.params['random_seed'], **self.params['model_params'])
        return make_pipeline(StandardScaler(), r)

class HuberE(RidgeExperimentV5):
    """Huber regression using rolling window averaged basic weather data and circular time."""

    def create_model(self):
        r = HuberRegressor(epsilon=2, **self.params['model_params'])
        return make_pipeline(StandardScaler(), r)


def make_parameterized_experiment(e_class, **params):
    def inner(**new_params):
        params.update(new_params)
        return e_class(**params)
    return inner


BASELINE_EXPERIMENTS = [
    MeanTempcExperiment, 
    MedianTempcExperiment, 
    IdentityTempcExperiment, 
    MeanOffsetExperiment, 
    LastKTempcMeanE,
    RecentDeviceTempcMeanE,
    RollingTempcMeanE,
    RollingMeanOffsetE,
]

RF_EXPERIMENTS = [
    RFExperimentV0,
    RFExperimentV1,
    RFExperimentV2,
    RFExperimentV3,
    RFExperimentV6,
    RFExperimentV4,
    RFExperimentV5,
    make_parameterized_experiment(RFExperimentV5, name='regularized', model_params=dict(max_depth=20, min_samples_split=5, max_features='sqrt')),
    RF_AllVariablesE,
]

LINEAR_MODEL_EXPERIMENTS = [
    RidgeExperimentV1,
    RidgeExperimentV2,
    RidgeExperimentV3,
    RidgeExperimentV4,
    RidgeExperimentV5,
    HuberE,
]

ALL_EXPERIMENTS = [
    *BASELINE_EXPERIMENTS,
    *RF_EXPERIMENTS[-4:],
    *LINEAR_MODEL_EXPERIMENTS,
]
