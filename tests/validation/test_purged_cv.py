import numpy as np

from src.validation.purged_cv import PurgedTimeSeriesSplit


def test_event_interval_is_purged_before_test_block():
    ts=np.arange(20,dtype=np.int64)
    event_end=ts+3
    splitter=PurgedTimeSeriesSplit(n_splits=3,embargo=2)
    for train,test in splitter.split(ts,event_end):
        if len(train)==0: continue
        cutoff=ts[test].min()-2
        assert np.all(event_end[train]<cutoff)
