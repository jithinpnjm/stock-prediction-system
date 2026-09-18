import numpy as np
from typing import Generator, Tuple

class PurgedTimeSeriesSplit:
    """
    Time Series cross-validator that provides train/test indices to split time series data samples.
    It implements an embargo gap between train and test to prevent data leakage 
    due to overlapping observation windows (e.g., from rolling features or barrier labels).
    """
    def __init__(self, n_splits: int = 5, embargo_size: int = 0):
        """
        :param n_splits: Number of splits.
        :param embargo_size: Number of samples to drop between train and test sets.
        """
        self.n_splits = n_splits
        self.embargo_size = embargo_size

    def split(self, X: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Determine the size of the test sets
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Test block
            test_start = (i + 1) * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            
            # Train block (everything before the test block, minus embargo)
            train_end = max(0, test_start - self.embargo_size)
            train_indices = indices[0:train_end]
            
            # Only yield if we have training data
            if len(train_indices) > 0:
                yield train_indices, test_indices
