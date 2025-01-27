import torch

from numba import cuda
from sigpde.BatchIterator import BatchIterator
from sigpde.utils import tensor_type, sqrt_ceil

from sigpde.PDESolvers import (
    PairwisePDESolver,
    GramPDESolver,
    SymmetricGramPDESolver
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
        y_scale = x_scale if y is None else y_scale
        y = x if y is None else y
        
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
            tensor_type(x),
            self.dyadic_order,
            max_batch,
            max_threads
        )

        result = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        
        for _, start, stop in BatchIterator(batch_size, solver.max_batch):
            cuda.synchronize()
            inc = pairwise_inner_product(x[start:stop,:,:], y[start:stop,:,:], self.static_kernel, self.dyadic_order) 
            if is_scaled:
                solver.solve_scaled(inc, x_scale[start:stop], y_scale[start:stop], result[start:stop])
            else:
                solver.solve(inc, result[start:stop])
                
        cuda.synchronize()
        
        return result
    
    def gram(self, x, y=None, x_scale=None, y_scale=None, max_batch=1000, max_threads=1024):
        max_batch = sqrt_ceil(max_batch)
        
        if y is None:
            return self._gram_sym(x, x_scale, max_batch, max_threads)
        else:
            return self._gram(x, y, x_scale, y_scale, max_batch, max_threads)
        
    def _gram(self, x, y, x_scale, y_scale, max_batch, max_threads):
        is_scaled = x_scale is not None or y_scale is not None
        batch_size_x = x.shape[0]
        batch_size_y = y.shape[0]
        
        if is_scaled:
            x_scale = torch.ones(batch_size_x, device=x.device, dtype=x.dtype) if x_scale is None else x_scale
            y_scale = torch.ones(batch_size_y, device=y.device, dtype=y.dtype) if y_scale is None else y_scale
            x_scale = x_scale.repeat(batch_size_x) if x_scale.shape[0] == 1 else x_scale
            y_scale = y_scale.repeat(batch_size_y) if y_scale.shape[0] == 1 else y_scale
            
        solver = GramPDESolver(
            batch_size_x,
            batch_size_y,
            x.shape[1],
            y.shape[1],
            tensor_type(x),
            self.dyadic_order,
            max_batch,
            max_threads
        )
        
        final_result = torch.zeros((batch_size_x, batch_size_y), device=x.device, dtype=x.dtype)
        result = torch.zeros((solver.batch_size_x, solver.batch_size_y), device=x.device, dtype=x.dtype)
        
        for _, start_y, stop_y in BatchIterator(batch_size_y, solver.max_batch_y):
            for _, start_x, stop_x in BatchIterator(batch_size_x, solver.max_batch_x):
                inc = gram_inner_product(
                    x[start_x:stop_x,:,:],
                    y[start_y:stop_y,:,:],
                    self.static_kernel,
                    self.dyadic_order
                ) 
                
                if is_scaled:
                    solver.solve_scaled(
                        inc, 
                        x_scale[start_x:stop_x], 
                        y_scale[start_y:stop_y],
                        result
                    )
                else:
                    solver.solve(
                        inc,
                        result
                    )
                    
                final_result[start_x:stop_x, start_y:stop_y] = result[0:inc.shape[0], 0:inc.shape[1]]
                
        cuda.synchronize()
        
        return final_result
    
    def _gram_sym(self, x, x_scale, max_batch, max_threads):
        is_scaled = x_scale is not None
        
        batch_size = x.shape[0]
        
        solver = SymmetricGramPDESolver(
            batch_size,
            x.shape[1],
            tensor_type(x),
            self.dyadic_order,
            max_batch,
            max_threads
        )
        
        result = torch.zeros((batch_size, batch_size), device=x.device, dtype=x.dtype)
        for _, start_y, stop_y in BatchIterator(batch_size, solver.max_batch):
            for _, start_x, stop_x in BatchIterator(batch_size, solver.max_batch):
                inc = gram_inner_product(
                    x[start_x:stop_x,:,:],
                    x[start_y:stop_y,:,:],
                    self.static_kernel,
                    self.dyadic_order
                ) 
                
                if is_scaled:
                    solver.solve_scaled(
                        inc, 
                        x_scale[start_x:stop_x], 
                        x_scale[start_y:stop_y], 
                        start_x, 
                        start_y,
                        result
                    )
                else:
                    solver.solve(
                        inc,
                        start_x, 
                        start_y,
                        result
                    )
                    
        cuda.synchronize()
            
        return result
        
        
class RobustSigPDE():
    def __init__(self, static_kernel, dyadic_order=0, normalizer=None):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self.normalizer = log_normalizer if normalizer is None else normalizer
        
    def _norm_factors(self, solver, inc, norms, scales, normalizer, tol=1e-8, maxit=100):
        solver.solve(inc, norms)

        normalized_norms = normalizer(norms)
        
        solver.solve_norms(
            inc,
            normalized_norms,
            norms,
            scales,
            tol,
            maxit
        )
        
        scales[norms < 1] = 0
        
    def normalization(self, x, normalizer=None, tol=1e-8, maxit=100, max_batch=1000, max_threads=1024):
        normalizer = self.normalizer if normalizer is None else normalizer
        
        batch_size = x.shape[0]
               
        solver = PairwisePDESolver(
            batch_size, 
            x.shape[1], 
            x.shape[1], 
            tensor_type(x),
            self.dyadic_order,
            max_batch,
            max_threads
        )
        
        x_scales = torch.zeros(solver.batch_size, device=x.device, dtype=x.dtype)
        x_norms = torch.zeros(solver.batch_size, device=x.device, dtype=x.dtype)
        
        for _, start, stop in BatchIterator(batch_size, solver.max_batch):
            x_inc = pairwise_inner_product(x[start:stop,:,:], x[start:stop,:,:], self.static_kernel, self.dyadic_order)
            
            self._norm_factors(
                solver,
                x_inc,
                x_norms[start:stop],
                x_scales[start:stop],
                normalizer,
                tol,
                maxit
            )
            
        cuda.synchronize()
            
        return x_scales
        
    def pairwise(self, x, y=None, normalizer=None, tol=1e-8, maxit=100, max_batch=1000, max_threads=1024):
        symmetric = y is None
        y = x if symmetric else y
        normalizer = self.normalizer if normalizer is None else normalizer
        
        batch_size = x.shape[0]
               
        solver = PairwisePDESolver(
            batch_size, 
            x.shape[1], 
            y.shape[1], 
            tensor_type(x),
            self.dyadic_order,
            max_batch,
            max_threads
        )
        
        x_scales = torch.zeros(solver.max_batch, device=x.device, dtype=x.dtype)
        y_scales = None if symmetric else torch.zeros(solver.max_batch, device=y.device, dtype=y.dtype)
        x_norms = torch.zeros(solver.max_batch, device=x.device, dtype=x.dtype)
        y_norms = None if symmetric else torch.zeros(solver.max_batch, device=y.device, dtype=y.dtype)
        result = torch.zeros(batch_size, device=x.device, dtype=x.dtype)        
        
        for _, start, stop in BatchIterator(batch_size, solver.max_batch):
            cuda.synchronize()
            x_inc = pairwise_inner_product(x[start:stop,:,:], x[start:stop,:,:], self.static_kernel, self.dyadic_order)
            
            self._norm_factors(
                solver,
                x_inc,
                x_norms,
                x_scales,
                normalizer,
                tol,
                maxit
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
                    tol,
                    maxit
                )
                    
                inc = pairwise_inner_product(x[start:stop,:,:], y[start:stop,:,:], self.static_kernel, self.dyadic_order) 
            
            solver.solve_scaled(inc, x_scales, y_scales, result[start:stop])
        
        cuda.synchronize()
        
        return result
    
    def gram(self, x, y=None, x_scale=None, y_scale=None, normalizer=None, tol=1e-8, maxit=100, max_batch=1000, max_threads=1024):
        x_scales = self.normalization(x, normalizer, tol, maxit, max_batch, max_threads) if x_scale is None else x_scale
        if y is None:
            y_scales = None
        else:
            y_scales = self.normalization(y, normalizer, tol, maxit, max_batch, max_threads) if y_scale is None else y_scale
        
        solver = SigPDE(self.static_kernel, self.dyadic_order)
        return solver.gram(x, y, x_scales, y_scales, max_batch, max_threads)