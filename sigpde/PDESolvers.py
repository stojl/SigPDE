from numba import cuda
from numpy import float64

from sigpde.bufferFactory import (
    PairwiseBufferFactory,
    GramBufferFactory,
    SymmetricGramBufferFactory
)

from sigpde.cuda_pairwise_kernels import (
    sigpde_pairwise,
    sigpde_pairwise_scaled,
    sigpde_pairwise_norm_chandraputla
)

from sigpde.cuda_gram_kernels import (
    sigpde_gram,
    sigpde_gram_scaled,
    sigpde_gram_symmetric,
    sigpde_gram_symmetric_scaled
)

from sigpde.utils import (
    anti_diagonals,
    round_to_multiple_of_32,
    dyadic_refinement_length,
    ceil_div
)

def thread_multiplicity(n, threads):
    return ceil_div(n - 1, threads)

def threads_per_block(n, max_threads=1024):
    threads_per_block = min(max_threads, n - 1, 1024)
    return round_to_multiple_of_32(threads_per_block)

class PairwisePDESolver():
    def __init__(self, batch_size, x_length, y_length, dtype=None, dyadic_order=0, max_batch=1000, max_threads=1024):
        y_length = x_length if y_length is None else y_length
        
        self.dtype = float64 if dtype is None else dtype
        self.batch_size = batch_size
        self.length_x = dyadic_refinement_length(x_length, dyadic_order)
        self.length_y = dyadic_refinement_length(y_length, dyadic_order)
        self.anti_diagonals = anti_diagonals(self.length_x, self.length_y)
        self.dyadic_order = dyadic_order
        self.threads_per_block = threads_per_block(self.length_x, max_threads)
        self.max_batch = min(max_batch, self.batch_size)
        self.batch_mult = ceil_div(self.batch_size, self.max_batch)
        self.thread_mult = thread_multiplicity(self.length_x, self.threads_per_block)
        
        self.buffer_factory = PairwiseBufferFactory(self.max_batch, self.length_x, self.dtype)
        self.buffer = self.buffer_factory()
        self.norm_buffer = None
        
    def solve(self, increments, result):
        blocks = min(increments.__cuda_array_interface__["shape"][0], self.max_batch)
        
        sigpde_pairwise[blocks, self.threads_per_block](
            cuda.as_cuda_array(increments),
            self.length_x,
            self.length_y,
            self.dyadic_order,
            self.thread_mult,
            self.anti_diagonals,
            self.buffer,
            cuda.as_cuda_array(result)
        )
        
    def solve_scaled(self, increments, scale_x, scale_y, result):
        blocks = min(increments.__cuda_array_interface__["shape"][0], self.max_batch)
                
        sigpde_pairwise_scaled[blocks, self.threads_per_block](
            cuda.as_cuda_array(increments),
            self.length_x,
            self.length_y,
            cuda.as_cuda_array(scale_x),
            cuda.as_cuda_array(scale_y),
            self.dyadic_order,
            self.thread_mult,
            self.anti_diagonals,
            self.buffer,
            cuda.as_cuda_array(result)
        )
        
    def solve_norms(self, increments, norms, f_norms, result, tol=1e-8, maxit=100):
        blocks = min(increments.__cuda_array_interface__["shape"][0], self.max_batch)
    
        sigpde_pairwise_norm_chandraputla[blocks, self.threads_per_block](
            cuda.as_cuda_array(increments), 
            cuda.as_cuda_array(norms), 
            cuda.as_cuda_array(f_norms),
            self.length_x, 
            self.dyadic_order, 
            self.thread_mult, 
            self.anti_diagonals, 
            self.buffer, 
            cuda.as_cuda_array(result),
            maxit,
            tol
        )
        
    def solve_norms_legacy(self, increments, norms, f_norms, result, init_tol=0.01, tol=1e-8, maxit=100):
        blocks = min(increments.__cuda_array_interface__["shape"][0], self.max_batch)
    
        if self.norm_buffer is None:
            self.norm_buffer = cuda.device_array((self.max_batch, 4), dtype=self.dtype)
            
        sigpde_pairwise_norm_init[blocks, self.threads_per_block](
            cuda.as_cuda_array(increments), 
            cuda.as_cuda_array(norms), 
            cuda.as_cuda_array(f_norms),
            self.length_x, 
            self.dyadic_order, 
            self.thread_mult, 
            self.anti_diagonals, 
            self.buffer, 
            self.norm_buffer,
            maxit,
            init_tol
        )
        
        sigpde_pairwise_norm[blocks, self.threads_per_block](
            cuda.as_cuda_array(increments), 
            cuda.as_cuda_array(norms),
            self.norm_buffer,
            self.length_x,
            self.dyadic_order, 
            self.thread_mult, 
            self.anti_diagonals, 
            self.buffer, 
            cuda.as_cuda_array(result),
            maxit,
            tol
        )
        
class GramPDESolver():
    def __init__(self, x_batch_size, y_batch_size, x_length, y_length, dtype=None, dyadic_order=0, max_batch=1000, max_threads=1024):
        y_length = x_length if y_length is None else y_length
        
        self.dtype = float64 if dtype is None else dtype
        self.batch_size_x = x_batch_size
        self.batch_size_y = y_batch_size
        
        self.length_x = dyadic_refinement_length(x_length, dyadic_order)
        self.length_y = dyadic_refinement_length(y_length, dyadic_order)
        self.anti_diagonals = anti_diagonals(self.length_x, self.length_y)
        self.dyadic_order = dyadic_order
        self.threads_per_block = threads_per_block(self.length_x, max_threads)
        self.max_batch_x = min(max_batch, self.batch_size_x)
        self.max_batch_y = min(max_batch, self.batch_size_y)
        self.batch_mult_x = ceil_div(self.batch_size_x, self.max_batch_x)
        self.batch_mult_y = ceil_div(self.batch_size_y, self.max_batch_y)
        self.thread_mult = thread_multiplicity(self.length_x, self.threads_per_block)
        
        self.buffer_factory = GramBufferFactory(self.max_batch_x, self.max_batch_y, self.length_x, self.dtype)
        self.buffer = self.buffer_factory()
        self.norm_buffer = None
    
    def solve(self, increments, result):
        blocks_x = min(increments.__cuda_array_interface__["shape"][0], self.max_batch_x)
        blocks_y = min(increments.__cuda_array_interface__["shape"][1], self.max_batch_y)

        sigpde_gram[(blocks_x, blocks_y), self.threads_per_block](
            cuda.as_cuda_array(increments),
            self.length_x,
            self.length_y,
            self.dyadic_order,
            self.thread_mult,
            self.anti_diagonals,
            self.buffer,
            cuda.as_cuda_array(result)
        )
        
    def solve_scaled(self, increments, scale_x, scale_y, result):
        blocks_x = min(increments.__cuda_array_interface__["shape"][0], self.max_batch_x)
        blocks_y = min(increments.__cuda_array_interface__["shape"][1], self.max_batch_y)

        sigpde_gram_scaled[(blocks_x, blocks_y), self.threads_per_block](
            cuda.as_cuda_array(increments),
            self.length_x,
            self.length_y,
            cuda.as_cuda_array(scale_x),
            cuda.as_cuda_array(scale_y),
            self.dyadic_order,
            self.thread_mult,
            self.anti_diagonals,
            self.buffer,
            cuda.as_cuda_array(result)
        )
        
class SymmetricGramPDESolver():
    def __init__(self, batch_size, length, dtype=None, dyadic_order=0, max_batch=1000, max_threads=1024):        
        self.dtype = float64 if dtype is None else dtype
        self.batch_size = batch_size   
        
        self.length = dyadic_refinement_length(length, dyadic_order)
        self.anti_diagonals = anti_diagonals(self.length, self.length)
        self.dyadic_order = dyadic_order
        self.threads_per_block = threads_per_block(self.length, max_threads)
        self.max_batch = min(max_batch, self.batch_size)
        self.batch_mult = ceil_div(self.batch_size, self.max_batch)
        
        self.thread_mult = thread_multiplicity(self.length, self.threads_per_block)

        self.buffer_factory = SymmetricGramBufferFactory(self.max_batch, self.max_batch, self.length, self.dtype)
        self.buffer = self.buffer_factory()
    
    def solve(self, increments, x_offset, y_offset, result):
        blocks_x = min(increments.__cuda_array_interface__["shape"][0], self.max_batch)
        blocks_y = min(increments.__cuda_array_interface__["shape"][1], self.max_batch)

        sigpde_gram_symmetric[(blocks_x, blocks_y), self.threads_per_block](
            cuda.as_cuda_array(increments),
            self.length,
            x_offset,
            y_offset,
            self.max_batch,
            self.batch_size,
            self.dyadic_order,
            self.thread_mult,
            self.anti_diagonals,
            self.buffer,
            cuda.as_cuda_array(result)
        )
        
    def solve_scaled(self, increments, scale_x, scale_y, x_offset, y_offset, result):
        blocks_x = min(increments.__cuda_array_interface__["shape"][0], self.max_batch)
        blocks_y = min(increments.__cuda_array_interface__["shape"][1], self.max_batch)

        sigpde_gram_symmetric_scaled[(blocks_x, blocks_y), self.threads_per_block](
            cuda.as_cuda_array(increments),
            self.length,
            cuda.as_cuda_array(scale_x),
            cuda.as_cuda_array(scale_y),
            x_offset,
            y_offset,
            self.max_batch,
            self.batch_size,
            self.dyadic_order,
            self.thread_mult,
            self.anti_diagonals,
            self.buffer,
            cuda.as_cuda_array(result)
        )