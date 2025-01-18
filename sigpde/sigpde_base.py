from numba import cuda
from cuda_pairwise_kernels import (
    sigpde_pairwise
)
from bufferFactory import (
    PairwiseBufferFactory,
    GramBufferFactory,
    SymmetricGramBufferFactory
)

def cuda_array_attributes(x):
    x = x.__cuda_array_interface__
    return x["shape"], x["typestr"]


class SigPDEBaseCUDA():
    @staticmethod
    def pairwise_kernel(increments, dyadic_order=0, out=None, max_batch=10000):
        shape = increments.__cuda_array_interface__["shape"]
                   
        assert x_shape[0] == y_shape[0] and x_shape[1] == y_shape[1]
        batch_size = x_shape[0]
        dim = x_shape[1]
        
        x_length = x_shape[2]
        y_length = y_shape[2]
        
        assert x_typestr == y_typestr
        dtype = x_typestr
        
        if out is None:
            out = cuda.device_array((batch_size), dtype=dtype)
            
        try:
            out_shape, out_typestr = cuda_array_attributes(out)    
        except AttributeError as e:
            print(str(e))
        
        assert out_shape[0] == batch_size
        assert out_typestr == dtype

class SigPDELinear():
    @staticmethod
    def pairwise_kernel(X, Y=None, dyadic_order=0, out=None, max_batch=50000):
        Y = X if Y is None else Y
        
        try:
            x_shape, x_typestr = cuda_array_attributes(X)
            y_shape, y_typestr = cuda_array_attributes(Y)            
        except AttributeError as e:
            print(str(e))
            
        assert x_shape[0] == y_shape[0] and x_shape[1] == y_shape[1]
        batch_size = x_shape[0]
        dim = x_shape[1]
        
        x_length = x_shape[2]
        y_length = y_shape[2]
        
        assert x_typestr == y_typestr
        dtype = x_typestr
        
        if out is None:
            out = cuda.device_array((batch_size), dtype=dtype)
            
        try:
            out_shape, out_typestr = cuda_array_attributes(out)    
        except AttributeError as e:
            print(str(e))
        
        assert out_shape[0] == batch_size
        assert out_typestr == dtype
        
        
        
        