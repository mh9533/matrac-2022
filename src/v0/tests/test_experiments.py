import pathlib 
import tempfile

import pandas as pd 
import numpy as np 

from src.experiment import load_all_experiment_results
from src.v0 import experiments
from src.v0 import config as cfg


def dummy_tempc_small(serials=cfg.master_serials, preprocess=True, include_device_src=False):
    X = {'tempc': [1,2,3,4]}
    dates = pd.date_range('2022-04-01 00:10:00', periods=len(X['tempc']))
    return pd.DataFrame(X, index=dates)

def dummy_tempc_big(serials=cfg.master_serials, preprocess=True, include_device_src=False):
    X = {'tempc': np.random.randn(1000) }
    dates = pd.date_range('2022-04-01 00:10:00', periods=len(X['tempc']))
    return pd.DataFrame(X, index=dates)

def test_stack_last_k():
    X = dummy_tempc_small()
    np.testing.assert_array_equal(experiments.stack_last_k(X, k=1).to_numpy(), X.to_numpy())
    np.testing.assert_array_equal(experiments.stack_last_k(X, k=4).to_numpy(),
        np.array([[1,2,3,4],[1,1,2,3],[1,1,1,2],[1,1,1,1]]).T)
    X = pd.concat([X, X], axis=1)
    X.columns = ['tempc', 'tempc2']
    np.testing.assert_array_equal(experiments.stack_last_k(X, k=1, select=X.columns).to_numpy(), X.to_numpy())
    np.testing.assert_array_equal(experiments.stack_last_k(X, k=4, select=X.columns).to_numpy(),
        np.array([[1,2,3,4],[1,1,2,3],[1,1,1,2],[1,1,1,1],[1,2,3,4],[1,1,2,3],[1,1,1,2],[1,1,1,1]]).T)

def test_load_save_experiments():
    experiments.load_api_data = dummy_tempc_big
    experiments.load_arso_data = dummy_tempc_big
    with tempfile.TemporaryDirectory() as tmpdir:
        experiments.cfg.experiment_results_path = pathlib.Path(tmpdir)

        e = experiments.IdentityTempcExperiment(n_splits=10)
        assert len(load_all_experiment_results(e.get_results_path())) == 0
        e.evaluate()
        res1 = load_all_experiment_results(e.get_results_path())
        assert len(res1) == 1
        e.evaluate()
        res2 = load_all_experiment_results(e.get_results_path())
        assert len(res2) == 1
        np.testing.assert_equal(res1, res2)
        e.evaluate(force=True)
        res3 = load_all_experiment_results(e.get_results_path())
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_equal(res1, res3)

        e = experiments.IdentityTempcExperiment(n_splits=5)
        assert len(load_all_experiment_results(e.get_results_path())) == 1
        e.evaluate()
        assert len(load_all_experiment_results(e.get_results_path())) == 2
