from numba import cuda

from sigpde.cuda_device_functions import (
    thread_multiplicty,
    anti_diagonals,
    device_linear_kernel,
    pde_step,
    dyadic_refinement_length
)
       
@cuda.jit
def sigpde_pairwise(incs, length_x, length_y, order, L, N, sol, out):
    """
    incs: Inner product of increments <x_{i}-x_{i-1}, y_{j}-y_{j-1}>
    length_x: Length of the sequence x
    length_y: Length of the sequence y
    order: Dyadic order of the PDE-solver
    sol: Solution buffer
    out: Result buffer
    """
    
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_y:
                inc = incs[block_id, (i - 1) >> order, (j - 1) >> order]
                
                k_01 = 1.0 if i == 1 else sol[block_id, i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[block_id, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[block_id, i - 2, K3]
                               
                sol[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                
                if p == N - 1:
                    out[block_id] = sol[block_id, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
@cuda.jit
def sigpde_pairwise_scaled(incs, length_x, length_y, scale_x, scale_y, order, L, N, sol, out):
    """
    incs: Inner product of increments <x_{i} - x_{i-1}, y_{j} - y_{j-1}>
    length_x: Length of the sequence x after dyadic refinement
    length_y: Length of the sequence y after dyadic refinement
    scale_x: Scaling of x
    scale_y: Scaling of y
    order: Dyadic order of the PDE-solver
    sol: Solution buffer
    out: Result buffer
    """
       
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
      
    K1 = 0
    K2 = 2
    K3 = 1
    
    scale = scale_y[block_id] * scale_x[block_id]
    
    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_y:
                inc = incs[block_id, (i - 1) >> order, (j - 1) >> order] * scale
                
                k_01 = 1.0 if i == 1 else sol[block_id, i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[block_id, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[block_id, i - 2, K3]
                               
                sol[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                
                if p == N - 1:
                    out[block_id] = sol[block_id, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
@cuda.jit
def sigpde_pairwise_linear(x, y, length_x, length_y, dim, order, sol, out):
    """
    Specialized version for the linear kernel.
    x: Array of shape (batch, dim, length_y)
    y: Array of shape (batch, dim, length_x)
    length_x: 2**order * (length_x - 1) + 1
    length_y: 2**order * (length_y - 1) + 1
    dim: Dimension of sequences
    order: Dyadic order of the PDE-solver
    sol: Solution buffer
    out: Result buffer
    """
    
    L = thread_multiplicty(length_x, cuda.blockDim.x)
    N = anti_diagonals(length_x, length_y)
    
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_y:               
                inc = device_linear_kernel(x[block_id,:,:], y[block_id,:,:], (i - 1) >> order, (j - 1) >> order, dim)

                k_01 = 1.0 if i == 1 else sol[block_id, i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[block_id, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[block_id, i - 2, K3]
                               
                sol[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                
                if p == N - 1:
                    out[block_id] = sol[block_id, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
@cuda.jit
def sigpde_pairwise_linear_scaled(x, y, length_x, length_y, dim, scale_x, scale_y, order, sol, out):
    """
    Specialized version for the linear kernel.
    x: Array of shape (batch, dim, length_y)
    y: Array of shape (batch, dim, length_x)
    length_x: 2**order * (length_x - 1) + 1
    length_y: 2**order * (length_y - 1) + 1
    dim: Dimension of sequences
    order: Dyadic order of the PDE-solver
    sol: Solution buffer
    out: Result buffer
    """
    
    L = thread_multiplicty(length_x, cuda.blockDim.x)
    N = anti_diagonals(length_x, length_y)
    
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    s_x = scale_x[block_id]
    s_y = scale_y[block_id]
    
    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_y:               
                inc = device_linear_kernel(x[block_id,:,:], y[block_id,:,:], (i - 1) >> order, (j - 1) >> order, dim)

                k_01 = 1.0 if i == 1 else sol[block_id, i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[block_id, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[block_id, i - 2, K3]
                               
                sol[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc * s_y * s_x)
                
                if p == N - 1:
                    out[block_id] = sol[block_id, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()