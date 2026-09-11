import pytest

def test_validation_does_not_fill_gap():
    pl=pytest.importorskip('polars')
    from xau_lab.data.validate import validate_bars
    f=pl.DataFrame({'time':['2026-01-05 10:00:00','2026-01-05 10:02:00'],'open':[10.,10.],'high':[11.,11.],'low':[9.,9.],'close':[10.,10.],'tick_volume':[1,1],'spread':[1,1],'real_volume':[0,0]})
    r=validate_bars(f); assert 'unexpected_gap' in r.issues['issue_type'].to_list(); assert r.clean.height==2

def test_sma5_feature_is_lagged():
    pl=pytest.importorskip('polars')
    from xau_lab.data.features import build_shared_features
    vals=[10,11,12,13,14,100]
    f=pl.DataFrame({'time':[f'2026-01-05 10:0{i}:00' for i in range(6)],'open':vals,'high':[x+1 for x in vals],'low':[x-1 for x in vals],'close':vals,'tick_volume':[1]*6,'spread':[1]*6,'real_volume':[0]*6})
    o=build_shared_features(f); assert o['sma_5_lag1'][5]==12.0
