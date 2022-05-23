import copy
import datetime
import pathlib

import joblib
import numpy as np
import pandas as pd

from .evaluate import evaluate_model, make_combined_summary, make_evaluation_summary


def stack_last_k(X_df, k, select=['tempc']):
    # X columns: T_0, T_1, ..., T_(k-1)
    #   where T_i is the i-th most recent previous temperature to T0
    # If there is no previous measurement for T_i, copy from T_(i-1).
    # NB: T_i can come from any device.

    X = { f'{c}_{i}': [] for c in select for i in range(k) }
    for c in select:
        X[f'{c}_{0}'] = X_df[c]
    # Just shift the whole column down i steps.
    # NB These are called "Lag features"
    for c in select:
        for i in range(1, k):
            T_prev = X[f'{c}_{i - 1}']
            T_i = T_prev.shift(1)
            T_i.iloc[0] = T_prev.iloc[0]
            X[f'{c}_{i}'] = T_i 
    X = pd.DataFrame(X)
    X.attrs = X_df.attrs
    return X

def stack_recent_per_device(X, select=['tempc'], max_time_gap=pd.Timedelta('4h')):
    """Stack most recent selected features from each device."""

    device_mask, device_ids = X.attrs['device_src']
    cols = {}
    for i in range(len(device_ids)):
        index = X.index[device_mask == i]
        index_i = X.reset_index().index[device_mask == i]
        def find(j):
            ts = X.index[j]
            pos = min(index.searchsorted(ts, side='right'), len(index) - 1)
            while pos >= 0 and index[pos] > ts:
                pos = pos - 1
            if pos < 0 or ts - index[pos] > max_time_gap:
                # TODO Fill missing value
                return j
            return index_i[pos]
        index = X.reset_index().index.map(find)
        cols[device_ids[i]] = X.iloc[index][select]
    df = pd.concat([x.reset_index(drop=True) for x in cols.values()], axis=1, ignore_index=True)
    df = df.set_index(X.index)
    df.attrs = X.attrs
    return df

def mean_features(X, dt='30min'):
    """Replace each feature with the mean of the last `dt` time units."""
    df = X.rolling(dt, min_periods=1).mean()
    df.attrs = X.attrs
    return df

def sin_cos_time(dtime):
    hour_time = dtime.dt.hour + dtime.dt.minute / 60
    hour_time = 2 * np.pi * hour_time / 24
    hour_x = np.cos(hour_time)
    hour_y = np.sin(hour_time)
    month_time = dtime.dt.month - 1 + dtime.dt.day / dtime.dt.days_in_month
    month_time = 2 * np.pi * month_time / 12
    month_x = np.cos(month_time)
    month_y = np.sin(month_time)
    return pd.DataFrame(dict(hour_x=hour_x, hour_y=hour_y, month_x=month_x, month_y=month_y))

def load_all_experiment_results(results_path: pathlib.Path):
    if not results_path.exists():
        return []
    return joblib.load(results_path)

def save_all_experiment_results(all_results, path: pathlib.Path):
    # Just dump the dictionary ... 
    # TODO use specialized storage algorithms if bothered...
    joblib.dump(all_results, path, compress=3)

class BaseExperiment:
    def __init__(self, 
        *,
        device_serials, 
        verbose=True, 
        random_seed=0, 
        n_splits=10,
        model_params=dict(), 
        **params
    ):
        params['device_serials'] = device_serials  # Load data only from specified devices
        params['random_seed'] = random_seed  # For reproducability
        params['model_params'] = model_params  # Model specific parameters (mostly for quick testing)
        params['n_splits'] = n_splits  # See evaluate.evaluate_model
        self.params = params
        self.verbose = verbose

    def name(self):
        if 'name' in self.params:
            return f'{self.__class__.__name__}_{self.params["name"]}'
        return f'{self.__class__.__name__}'        

    def get_params(self, deep=True) -> dict:
        params = copy.deepcopy(self.params)
        model = self.create_model()
        if hasattr(model, 'get_params'):
            params['model_params'].update(model.get_params(deep=deep))
        return params

    def load_data(self, include_device_src=True): raise NotImplementedError
    def create_model(self): raise NotImplementedError

    def save_results(self, results):
        """Save experiment evaluation results.

        Notes
        -----
        Each Experiment saves its results to self.get_results_path.
        An Experiment can be run with different parameters, so each
        file contains a list of saved runs.
        For each run we save some metadata, the used parameters, and
        the evaluation results (for each split).
        """
        path = self.get_results_path()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        all_results = load_all_experiment_results(path)
        
        metadata = { 'date': datetime.datetime.now().isoformat() }
        params = self.get_params(deep=True)
        r = next((r for r in all_results if r['params'] == params), False) # NB. python does deep comparisons
        if r:
            # If trying to save a run that already exists, replace it with the new results.
            r['metadata'] = metadata
            r['results'] = results
        else:
            r = {
                'metadata': metadata,
                'params': params,
                'results': results
            }
            all_results.append(r)

        save_all_experiment_results(all_results, path)
        if self.verbose: print(f'Saved results to: {path}')
    
    def load_results(self):
        """Load experiment results for this experiment.
        
        Returns
        -------
        None if:
          - the experiment doesn't have any saved results
          - the saved experiment hasn't been run with exactly 
            the same parameters as returned by get_params()

        See also
        --------
        load_all_experiment_results
        """
        path = self.get_results_path()
        if self.verbose: print(f'Loading results from: {path}')
        all_results = load_all_experiment_results(path)
        params = self.get_params(deep=True)
        r = next((r for r in all_results if r['params'] == params), False) # NB. python does deep comparisons
        if r:
            return r
        return None

    def get_results_path(self): raise NotImplementedError  # return cfg.experiment_results_path / f'{self.__class__.__name__}.dump'

    def evaluate(self,
        force=False,
        return_fold_summaries=True,
        **summary_params
    ):
        """Run and evaluate this experiment.

        Parameters
        ----------
        force : bool
            Whether to force run the experiment even if a saved run exists. 
        rest : see make_evaluation_summary

        WARNING
        -------
        If the experiment/framework/data... has changed, delete existing results and re-run!
        """
        np.random.seed(self.params['random_seed'])

        if self.verbose: print('Loading data...', end='')
        X_df, y_df = self.load_data(include_device_src=summary_params.get('include_device_src', True))
        if self.verbose: print('DONE')

        model = self.create_model()

        def run_and_save():
            eval_results = []
            for results in evaluate_model(model, X_df, y_df, n_splits=self.params['n_splits']):
                eval_results.append(results)
            self.save_results(eval_results)
            return eval_results

        if force:
            eval_results = run_and_save()
        else:
            eval_results = self.load_results()
            if eval_results:
                eval_results = eval_results['results']
            else:
                eval_results = run_and_save()
        
        all_summary = make_combined_summary(y_df, eval_results, **summary_params)
        if return_fold_summaries:
            folds_summary = make_evaluation_summary(X_df, y_df, eval_results, **summary_params)
            return folds_summary, all_summary
        return all_summary
    
    def train_model(self):
        X_df, y_df = self.load_data()
        
        X = X_df.to_numpy()
        y = y_df.to_numpy()

        model = self.create_model()
        model.fit(X, y)
        return model 
