from src.v0 import config as cfg
from src.v0 import data_utils

def test_load_api_data():
    df = data_utils.load_api_data(preprocess=False)
    assert len(df) > 0
    df = data_utils.load_api_data(preprocess=True)
    assert len(df) > 0
    assert df[['inserted_at', 'tempc', 'relhumperc', 'pressmbar']].notna().values.all()
    
    DFs = []
    for serial in cfg.master_serials:
        DFs.append(data_utils.load_api_data(serials=[serial], preprocess=True))
    df = data_utils.load_api_data(serials=cfg.master_serials, preprocess=True)
    assert sum((len(x) for x in DFs)) == len(df)

    df = data_utils.load_api_data(serials=cfg.master_serials, preprocess=True, include_device_src=True)
    for i, serial in enumerate(cfg.master_serials):
        assert (df.attrs['device_src'][0] == i).sum() in [len(df_) for df_ in DFs]

    assert df.index[0] == df.index.min()
    assert df.last_valid_index() == df.index.max()

def test_load_arso_data():
    df = data_utils.load_arso_data(preprocess=False)
    assert len(df) > 0
    df = data_utils.load_arso_data(preprocess=True)
    assert len(df) > 0
    assert df.notna().values.all()
