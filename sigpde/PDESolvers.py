from numba import cuda

from bufferFactory import (
    PairwiseBufferFactory,
    GramBufferFactory,
    SymmetricGramBufferFactory
)

from cuda_pairwise_kernels import (
    sigpde_pairwise,
    sigpde_pairwise_scaled
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
    def __init__(self, batch_size, x_length, y_length, dyadic_order=0, max_batch=1000, max_threads=1024):
        y_length = x_length if y_length is None else y_length
        
        self.batch_size = batch_size
        self.length_x = dyadic_refinement_length(x_length, dyadic_order)
        self.length_y = dyadic_refinement_length(y_length, dyadic_order)
        self.anti_diagonals = anti_diagonals(self.length_x, self.length_y)
        self.dyadic_order = dyadic_order
        self.threads_per_block = threads_per_block(self.length_x, max_threads)
        self.max_batch = min(max_batch, self.batch_size)
        self.batch_mult = ceil_div(self.batch_size, self.max_batch)
        self.thread_mult = thread_multiplicity(self.length_x, threads_per_block)
        
        self.buffer_factory = PairwiseBufferFactory(self.max_batch, self.length_x)
        self.buffer = self.buffer_factory()
        
    def solve(self, increments, result):
        blocks = min(increments.__cuda_array_interface__["shape"][0], self.max_batch)
        
        sigpde_pairwise[blocks, self.threads_per_block](
            incs = cuda.as_cuda_array(increments),
            length_x = self.length_x,
            length_y = self.length_y,
            order = self.dyadic_order,
            sol = self.buffer,
            out = cuda.as_cuda_array(result)
        )
        
    def solve_scaled(self, increments, scale_x, scale_y, result):
        blocks = min(increments.__cuda_array_interface__["shape"][0], self.max_batch)
        
        sigpde_pairwise_scaled[blocks, self.threads_per_block](
            incs = cuda.as_cuda_array(increments),
            length_x = self.length_x,
            length_y = self.length_y,
            scale_x = cuda.as_cuda_array(scale_x),
            scale_y = cuda.as_cuda_array(scale_y),
            order = self.dyadic_order,
            sol = self.buffer,
            out = cuda.as_cuda_array(result)
        )