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

def log_normalizer(x):
    return 2 - 1 / (1 + x.log())

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
        
class RobustSigPDE():
    def __init__(self, static_kernel, dyadic_order=0):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        
    def _norm_factors(self, solver, inc, norms, scales, normalizer, bisections, nr_iterations):
        solver.solve(inc, norms)
        norms = normalizer(norms)
        
        solver.solve_norms(
            inc,
            norms,
            scales,
            bisections,
            nr_iterations
        )
        
    def pairwise(self, x, y=None, normalizer=None, bisections=15, nr_iterations=10, max_batch=1000, max_threads=1024):
        symmetric = y is None
        y = x if symmetric else y
        normalizer = log_normalizer if normalizer is None else normalizer
        
        batch_size = x.shape[0]
               
        solver = PairwisePDESolver(
            batch_size, 
            x.shape[1], 
            y.shape[1], 
            self.dyadic_order,
            max_batch,
            max_threads
        )
        
        x_scales = torch.zeros(solver.batch_size, device=x.device, dtype=x.dtype)
        y_scales = None if symmetric else torch.zeros(solver.batch_size, device=y.device, dtype=y.dtype)
        x_norms = torch.zeros(solver.batch_size, device=x.device, dtype=x.dtype)
        y_norms = None if symmetric else torch.zeros(solver.batch_size, device=y.device, dtype=y.dtype)
        result = torch.zeros(batch_size, device=x.device, dtype=x.dtype)        
        
        for _, start, stop in BatchIterator(batch_size, solver.batch_size):
            x_inc = pairwise_inner_product(x[start:stop,:,:], x[start:stop,:,:], self.static_kernel, self.dyadic_order)
            
            self._norm_factors(
                solver,
                x_inc,
                x_norms,
                x_scales,
                normalizer,
                bisections,
                nr_iterations
            )
            
            if symmetric:
                y_scales = x_scales
                inc = x_inc
            else:
                y_inc = pairwise_inner_product(y[start:stop,:,:], y[start:stop,:,:], self.static_kernel, self.dyadic_order)
            
                self._norm_factors(
                    solver,
                    y_inc,
                    y_norms,
                    y_scales,
                    normalizer,
                    bisections,
                    nr_iterations
                )
                    
                inc = pairwise_inner_product(x[start:stop,:,:], y[start:stop,:,:], self.static_kernel, self.dyadic_order) 
            
            solver.solve_scaled(inc, x_scales, y_scales, result[start:stop])
            
        return result