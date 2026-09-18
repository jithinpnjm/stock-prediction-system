import polars as pl
import numpy as np
from numba import njit

@njit
def _compute_barriers(close: np.ndarray, high: np.ndarray, low: np.ndarray, target_pts: float, stop_pts: float, time_barrier: int):
    """
    Numba JIT compiled exact path traversal logic for triple barrier labels.
    Resolves targets intrabar with defensive collision handling (stop assumed hit first).
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        c_price = close[i]
        upper_barrier = c_price + target_pts
        lower_barrier = c_price - stop_pts
        
        hit_target = False
        hit_stop = False
        
        for j in range(i + 1, min(i + 1 + time_barrier, n)):
            if high[j] >= upper_barrier and low[j] <= lower_barrier:
                # Intrabar collision: Without strictly timestamped tick data, assume defensive stop loss hit.
                hit_stop = True
                break
            elif high[j] >= upper_barrier:
                hit_target = True
                break
            elif low[j] <= lower_barrier:
                hit_stop = True
                break
                
        if hit_target:
            labels[i] = 1
        elif hit_stop:
            labels[i] = -1
        else:
            labels[i] = 0
            
    return labels

def apply_triple_barrier_labels(
    df: pl.DataFrame,
    pt_sl_ratio: float = 200.0 / 70.0,
    stop_loss_pts: float = 70.0,
    time_barrier_bars: int = 75 
) -> pl.DataFrame:
    """
    Applies the triple barrier labeling method rigorously scanning forward price history.
    """
    target_pts = stop_loss_pts * pt_sl_ratio
    
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    
    labels = _compute_barriers(
        close_arr, high_arr, low_arr, target_pts, stop_loss_pts, time_barrier_bars
    )
    
    return df.with_columns(pl.Series("label", labels, dtype=pl.Int32))
