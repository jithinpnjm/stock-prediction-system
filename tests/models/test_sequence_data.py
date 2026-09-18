import numpy as np

from src.models.sequence_data import build_sequences


def test_sequence_windows_are_causal():
    x=np.arange(20*2,dtype=float).reshape(20,2)
    y=np.zeros(20,dtype=int)
    ts=np.arange(20)
    batch=build_sequences(x,y,ts,sequence_length=5)
    assert batch.X.shape==(16,5,2)
    assert np.array_equal(batch.timestamps,np.arange(4,20))
    assert np.array_equal(batch.X[0,-1],x[4])
