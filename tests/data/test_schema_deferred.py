import pytest

def test_schema_and_paths(tmp_path):
    pl=pytest.importorskip('polars')
    from xau_lab.data.schema import RAW_COLUMNS,DataPaths,canonicalize_bar_frame
    assert RAW_COLUMNS==['time','open','high','low','close','tick_volume','spread','real_volume']
    assert DataPaths(tmp_path).raw_csv==tmp_path/'data/raw/XAUUSD_M1_RAW.csv'
    f=pl.DataFrame({'close':[2.],'time':['2026-01-01 00:00:00'],'open':[1.],'high':[3.],'low':[.5],'spread':[35],'tick_volume':[1],'real_volume':[0]})
    o=canonicalize_bar_frame(f); assert o.columns==RAW_COLUMNS
