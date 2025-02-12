from numba import cuda
from numpy import float64

class PairwiseBufferFactory():
    def __init__(self, batch, length, dtype=None):
        self.dtype = float64 if dtype is None else dtype
        self.batch = batch
        self.length = length
        self.shape = self._shape(batch, length)

    def _shape(self, batch, length):
        return (batch, length - 1, 3)
    
    def __call__(self, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return cuda.device_array(self.shape, dtype)
    
class GramBufferFactory():
    def __init__(self, batch_x, batch_y, length, dtype=None):
        self.dtype = float64 if dtype is None else dtype
        self.batch_x = batch_x
        self.batch_y = batch_y
        self.length = length
        self.shape = self._shape(batch_x, batch_y, length)
    
    def _shape(self, batch_x, batch_y, length):
        return (batch_x, batch_y, length - 1, 3)
    
    def __call__(self, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return cuda.device_array(self.shape, dtype)
    
class SymmetricGramBufferFactory():
    def __init__(self, batch_x, batch_y, length, dtype=None):
        self.dtype = float64 if dtype is None else dtype
        self.batch_x = batch_x
        self.batch_y = batch_y
        self.length = length
        self.shape = self._shape(batch_x, batch_y, length)
        
    def triangular_cache(self, n, m):
        if m == 0 or n <= 0:
            return 0
    
        k = max(0, min(n, m))
        
        total = m * k
        remaining = max(0, (m - k) * (m - k + 1) // 2)
        
        return total - remaining
        
    def _shape(self, batch_x, batch_y, length):
        tri_shape = self.triangular_cache(batch_x, batch_y)
        return (tri_shape, length - 1, 3)
    
    def __call__(self, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return cuda.device_array(self.shape, dtype)