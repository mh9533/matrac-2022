import numpy as np
import pandas as pd 

from src import evaluate
from src.v0 import data_utils

def dummy_tempc_big():
    X = {'tempc': np.random.randn(1000) }
    dates = pd.date_range('2022-04-01 00:10:00', periods=len(X['tempc']))
    return pd.DataFrame(X, index=dates)

def test_get_closest_pairs_1():
    X = data_utils.load_api_data(preprocess=True)
    y = data_utils.load_arso_data(preprocess=True)
    idx1 = evaluate.get_closest_pairs(X, y, method='fast')
    idx2 = evaluate.get_closest_pairs(X, y, method='pandas')
    # NOTE: if same distance, get_loc rounds up, but we round down.
    # pd.testing.assert_index_equal(idx2, idx1)
    assert len(idx1) == len(idx2)
    assert (idx1 == idx2).sum() / len(idx1) > 0.99

def test_get_closest_pairs_2():
    X = dummy_tempc_big()
    y = dummy_tempc_big()
    idx1 = evaluate.get_closest_pairs(X, y, method='fast')
    idx2 = evaluate.get_closest_pairs(X, y, method='pandas')
    np.testing.assert_array_equal(idx1, idx2)

def test_get_closest_le_pairs_1():
    X = data_utils.load_api_data(preprocess=True)
    y = data_utils.load_arso_data(preprocess=True)
    idx1 = evaluate.get_closest_le_pairs(X, y, method='fast')
    idx2 = evaluate.get_closest_le_pairs(X, y, method='pandas')
    pd.testing.assert_index_equal(idx2, idx1)

def test_get_closest_le_pairs_2():
    X = dummy_tempc_big()
    y = dummy_tempc_big()
    idx1 = evaluate.get_closest_le_pairs(X, y, method='fast')
    idx2 = evaluate.get_closest_le_pairs(X, y, method='pandas')
    np.testing.assert_array_equal(idx1, idx2)

def test_train_test_split():
    X = data_utils.load_api_data(preprocess=True)
    y = data_utils.load_arso_data(preprocess=True)
    y = y.iloc[evaluate.get_closest_pairs(X, y)]
    for train, test in evaluate.time_train_test_split(X, y):
        """
        > Pri napovedovanju skozi čas skrbno preverimo, da model NIKOLI za učenje 
        > nima na voljo podatkov, ki so po času za tistimi, za katere mora 
        > napovedovati. Npr. učne množice ne smemo opremiti z ground-truth, ki 
        > je najbližje po času, ampak z zadnjim ground truth, ki je še na voljo 
        > pred časom. Za testne primere pa je bolj smiselno vzeti najbližjo.
        """
        # assert (y.iloc[train].index <= X.iloc[train].index).all()  # train y time <= train X time? or nearest?
        assert X.iloc[train].index.max() < X.iloc[test].index.min()  # train X time < test X time
        assert y.iloc[train].index.max() < X.iloc[test].index.min()  # train y time < test X time
        assert X.iloc[train].index.max() < y.iloc[test].index.min()  # train X time < test y time
        assert y.iloc[train].index.max() < y.iloc[test].index.min()  # train y time < test y time
        assert len(X.iloc[train]) > 0
        assert len(y.iloc[train]) > 0
        assert len(X.iloc[test]) > 0
        assert len(y.iloc[test]) > 0

        assert X.iloc[train]['tempc'].notna().values.all()
        assert y.iloc[train]['tempc'].notna().values.all()
        assert X.iloc[test]['tempc'].notna().values.all()
        assert y.iloc[test]['tempc'].notna().values.all()

    for train, test in evaluate.time_train_test_split(X, y, time_gap=pd.to_timedelta(0)):
        # assert (y.iloc[train].index <= X.iloc[train].index).all()  # train y time <= train X time? or nearest?
        assert X.iloc[train].index.max() < X.iloc[test].index.min()  # train X time < test X time
        assert y.iloc[train].index.max() < X.iloc[test].index.min()  # train y time < test X time
        assert X.iloc[train].index.max() < y.iloc[test].index.min()  # train X time < test y time
        assert y.iloc[train].index.max() < y.iloc[test].index.min()  # train y time < test y time
        assert len(X.iloc[train]) > 0
        assert len(y.iloc[train]) > 0
        assert len(X.iloc[test]) > 0
        assert len(y.iloc[test]) > 0

        assert X.iloc[train]['tempc'].notna().values.all()
        assert y.iloc[train]['tempc'].notna().values.all()
        assert X.iloc[test]['tempc'].notna().values.all()
        assert y.iloc[test]['tempc'].notna().values.all()