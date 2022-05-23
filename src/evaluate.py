import pandas as pd 
import numpy as np 
import scipy.stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


def RMSE(*ys):
    return mean_squared_error(*ys, squared=False)

default_evaluation_metrics = [
    ('MSE', mean_squared_error),
    ('MAE', mean_absolute_error),
    ('MedAE', median_absolute_error),
    ('RMSE', RMSE),
]

def get_closest_pairs(X_df, y_df, method='fast'):
    """Return indices of closest in time ground-truth in y to each observation in X."""
    if method == 'fast':
        def find(x):
            # TODO searchsorted supports vectorization
            pos = min(y_df.index.searchsorted(x, side='right'), len(y_df) - 1)
            pos = min((y_df.index[pos] - x, pos),
                    (x - y_df.index[max(0, pos - 1)], pos - 1))
            return pos[1]
    else:   
        def find(x):
            # Below method is slow. We can do better, since y_df is sorted!
            # NOTE: if same distance, get_loc rounds up, but we round down.
            return y_df.index.get_loc(x, method='nearest')

    # Assume index has datetime type
    return X_df.index.map(find)
    
def get_closest_le_pairs(X_df, y_df, method='fast'):
    """Return indices of closest in time ground-truth in y to each observation in X,
    such that y.time <= x.time.
    """
    if method == 'fast':
        def find(x):
            pos = min(y_df.index.searchsorted(x), len(y_df) - 1)
            if x == y_df.index[pos]:
                return pos 
            return pos - 1
    else:
        def find(x):
            return y_df.index.get_loc(x, method='pad')

    # Assume index has datetime type
    return X_df.index.map(find)

def time_train_test_split(X_df, y_df, n_splits=10, time_gap=pd.to_timedelta('5min')):
    """Yield train, test data splits for time series cross validation.
    
    Parameters
    ----------
    time_gap : pd.Timedelta
        Minimum time gap between train and test examples.
    """
    # TODO maybe split by timedelta instead of n_splits?
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X_df):
        # Take the min gap into account (to avoid train/test overlap).
        test_start_time = X_df.index[train_idx].max() + time_gap
        test_start_idx = X_df.index.searchsorted(test_start_time)
        test_idx = test_idx[test_idx > test_start_idx]

        # Make sure no train/test overlap.
        last_train_y_time = y_df.index[train_idx].max()
        test_idx = test_idx[y_df.index[test_idx] > last_train_y_time]

        yield train_idx, test_idx

def evaluate_model(model, X_df, y_df, n_splits=10):
    """Evaluate model based on time series data."""
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    for train, test in time_train_test_split(X_df, y_df, n_splits=n_splits):
        clf = model.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        yield {
            'y_pred': y_pred,
            'train': train,
            'test': test,
        }

# TODO optimize this function!!!!
def bootstrap_standard_error(y_true, y_pred, scorer, n_resamples=500, random_state=42):
    def statistic(idx):
        return scorer(y_true[idx], y_pred[idx])

    data = np.arange(0, len(y_true))
    res = scipy.stats.bootstrap([data], statistic=statistic, vectorized=False, 
        n_resamples=n_resamples, method='percentile', confidence_level=0.95, 
        random_state=random_state)
    return res.standard_error, res.confidence_interval

def count_significant_errors(y_true, y_pred, threshold=3):
    if len(y_true) == 0:
        return np.nan
    n = np.sum(np.abs(y_true - y_pred) >= threshold)
    return n / len(y_true)

def make_evaluation_summary(
    X_df, y_df,
    evaluation_results, 
    metrics=default_evaluation_metrics,
    include_support=True,
    include_standard_error=True,
    include_confidence_intervals=True,
    include_device_src=True,
    include_device_uncertainty=False,
    count_error_thresholds=[],  
    **ignored  # For pass-through params...
):
    """Create a per fold summary of the model's performance.
    
    Parameters
    ----------
    include_device_src : bool
        Whether to include per device evaluation results in the output summary.
    include_support : bool
        Whether to include the number of train/test samples for each split.
    count_errors_threshold : List[int]
        Count how many times this absolute error was exceeded.
    TODO ...
    """
    scorer_labels, scorers = zip(*metrics)

    columns = ['Train_start', 'Train_end']
    if include_support:
        columns.append('Train_support')
    columns += ['Test_start', 'Test_end']
    if include_support:
        columns.append('Test_support')
    for label in scorer_labels:
        columns.append(label)
        if include_standard_error:
            columns.append(f'{label}_SE')
        if include_confidence_intervals:
            columns.append(f'{label}_CI_lo')
            columns.append(f'{label}_CI_hi')
    for thr in count_error_thresholds:
        columns.append(f'n_thresh_error_{thr}')

    if include_device_src and 'device_src' in X_df.attrs:
        # 'device_src' is like: ([0, 0, 0, ..., 2, 2, 2], [29, 25, 31] -- device_id)
        device_src_lkp, device_src_ids = X_df.attrs['device_src']
        for id_ in device_src_ids:
            if include_support:
                columns.append(f'd{id_}_support')
            for label in scorer_labels:
                columns.append(f'{label}_d{id_}')
                if include_device_uncertainty:
                    if include_standard_error:
                        columns.append(f'{label}_SE_d{id_}')
                    if include_confidence_intervals:
                        columns.append(f'{label}_CI_lo_d{id_}')
                        columns.append(f'{label}_CI_hi_d{id_}')
            for thr in count_error_thresholds:
                columns.append(f'n_thresh_error_{thr}_d{id_}')

    df = { c: [] for c in columns }

    for results in evaluation_results:
        y_true, y_pred = y_df.iloc[results['test']], results['y_pred']

        df['Train_start'].append(X_df.index[results['train']].min())
        df['Train_end'].append(X_df.index[results['train']].max())
        df['Test_start'].append(X_df.index[results['test']].min())
        df['Test_end'].append(X_df.index[results['test']].max())
        if include_support:
            df['Train_support'].append(len(results['train']))
            df['Test_support'].append(len(results['test']))
        for thr in count_error_thresholds:
            df[f'n_thresh_error_{thr}'].append(count_significant_errors(y_true, y_pred, threshold=thr))

        for scorer, label in zip(scorers, scorer_labels):
            df[label].append(scorer(y_true, y_pred))
            
            if include_standard_error or include_confidence_intervals:
                se, ci = bootstrap_standard_error(y_true, y_pred, scorer)
                if include_standard_error:
                    df[f'{label}_SE'].append(se)
                if include_confidence_intervals:
                    df[f'{label}_CI_lo'].append(ci[0])
                    df[f'{label}_CI_hi'].append(ci[1])

        # Per device results
        if include_device_src and 'device_src' in X_df.attrs:
            test_ids = device_src_lkp[results['test']]
            for i in range(len(device_src_ids)):
                device_id = device_src_ids[i]
                m = (test_ids == i)
                support = m.sum()
                if include_support:
                    df[f'd{device_id}_support'].append(support)
                for thr in count_error_thresholds:
                    df[f'n_thresh_error_{thr}_d{device_id}'].append(count_significant_errors(y_true[m], y_pred[m], threshold=thr))
                for scorer, label in zip(scorers, scorer_labels):
                    if support > 0:
                        device_score = scorer(y_true[m], y_pred[m])
                    else:
                        device_score = None  # This device was not tested in this fold
                    df[f'{label}_d{device_id}'].append(device_score)
                    
                    if include_device_uncertainty and (include_standard_error or include_confidence_intervals):
                        if support > 0:
                            se, ci = bootstrap_standard_error(y_true[m], y_pred[m], scorer)
                        else:
                            se, ci = None, (None, None)
                        if include_standard_error:
                            df[f'{label}_SE_d{device_id}'].append(se)
                        if include_confidence_intervals:
                            df[f'{label}_CI_lo_d{device_id}'].append(ci[0])
                            df[f'{label}_CI_hi_d{device_id}'].append(ci[1])
    df = pd.DataFrame(df)
    return df 

def make_combined_summary(
    y_df,
    evaluation_results, 
    *,
    metrics=default_evaluation_metrics,
    include_standard_error=True,
    include_confidence_intervals=True,
    count_error_thresholds=[],  
    **ignored  # For pass-through params...
):
    """Combined (all folds) summary. One row for each evaluation metric."""
    scorer_labels, scorers = zip(*metrics)
    for thr in count_error_thresholds:
        def g(thr):
            def f(*ys):
                return count_significant_errors(*ys, threshold=thr)
            return f
        scorer_labels += f'n_thresh_error_{thr}',
        scorers += g(thr),

    df = { 'Loss': [] }
    if include_standard_error: 
        df['SE'] = []
    if include_confidence_intervals: 
        df['95_CI_lo'] = []
        df['95_CI_hi'] = []

    combined_y_pred = []
    combined_y_true = []
    for results in evaluation_results:
        y_true, y_pred = y_df.iloc[results['test']], results['y_pred']
        combined_y_pred.append(y_pred)
        combined_y_true.append(y_true)
    y_pred = np.concatenate(combined_y_pred)
    y_true = np.concatenate(combined_y_true)
    for scorer in scorers:
        df['Loss'].append(scorer(y_true, y_pred))
        if include_standard_error or include_confidence_intervals:
            se, ci = bootstrap_standard_error(y_true, y_pred, scorer)
            if include_standard_error: 
                df['SE'].append(se)
            if include_confidence_intervals:
                df['95_CI_lo'].append(ci[0])
                df['95_CI_hi'].append(ci[1])
    return pd.DataFrame(df, index=scorer_labels)

def evaluate_model_summary(
    model, X_df, y_df, n_splits=10, 
    **summary_params
):
    # Convenience function
    eval_results = []
    for results in evaluate_model(model, X_df, y_df, n_splits=n_splits):
        eval_results.append(results)
    folds_summary = make_evaluation_summary(X_df, y_df, eval_results, **summary_params)
    all_summary = make_combined_summary(y_df, eval_results, **summary_params)
    return folds_summary, all_summary
