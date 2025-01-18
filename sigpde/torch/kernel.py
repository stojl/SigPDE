import torch
from numba import cuda

from sigpde.BatchIterator import BatchIterator

from sigpde.PDESolvers import (
    PairwisePDESolver
)

def pairwise_inner_product(x, y, static_kernel, dyadic_order):
    ip = static_kernel.pairwise(x, y)
    ip = ip[:,1:,1:] + ip[:,:-1,:-1] - ip[:,1:,:-1] - ip[:,:-1,1:]
    return ip / float(2**(2 * dyadic_order))

def gram_inner_product(x, y, static_kernel, dyadic_order):
    ip = static_kernel.gram(x, y)
    ip = ip[:, :, 1:, 1:] + ip[:, :, :-1, :-1] - ip[:, :, 1:, :-1] - ip[:, :, :-1, 1:]
    return ip / float(2**(2 * dyadic_order))

class SigPDE():
    def __init__(self, static_kernel, dyadic_order=0):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        
    def pairwise(self, x, y=None, x_scale=None, y_scale=None, max_batch=1000, max_threads=1024):
        y = x if y is None else y
        y_scale = x_scale if y is None else y_scale
        
        batch_size = x.shape[0]
        is_scaled = x_scale is not None or y_scale is not None
        if is_scaled:
            x_scale = torch.ones(batch_size, device=x.device, dtype=x.dtype) if x_scale is None else x_scale
            y_scale = torch.ones(batch_size, device=y.device, dtype=y.dtype) if y_scale is None else y_scale
            x_scale = x_scale.repeat(batch_size) if x_scale.shape[0] == 1 else x_scale
            y_scale = y_scale.repeat(batch_size) if y_scale.shape[0] == 1 else y_scale
               
        solver = PairwisePDESolver(
            batch_size, 
            x.shape[1], 
            y.shape[1], 
            self.dyadic_order,
            max_batch,
            max_threads
        )

        result = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        
        for _, start, stop in BatchIterator(batch_size, solver.batch_size):
            inc = pairwise_inner_product(x[start:stop,:,:], y[start:stop,:,:], self.static_kernel, self.dyadic_order) 
            if is_scaled:
                solver.solve_scaled(inc, x_scale[start:stop], y_scale[start:stop], result[start:stop])
            else:
                solver.solve(inc, result[start:stop])
            
        return result
        
    
        