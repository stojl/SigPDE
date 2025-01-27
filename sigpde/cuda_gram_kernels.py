from numba import cuda

from sigpde.cuda_device_functions import (
    pde_step,
    cache_offset
)

@cuda.jit
def sigpde_gram(incs, length_x, length_y, order, L, N, sol, result):
    block_x = cuda.blockIdx.x
    block_y = cuda.blockIdx.y
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1

    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_y:
                inc = incs[block_x, block_y, (i - 1) >> order, (j - 1) >> order]
                
                k_01 = 1.0 if i == 1 else sol[block_x, block_y, i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[block_x, block_y, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[block_x, block_y, i - 2, K3]
                               
                sol[block_x, block_y, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                
                if p == N - 1:
                    result[block_x, block_y] = sol[block_x, block_y, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
@cuda.jit
def sigpde_gram_scaled(incs, length_x, length_y, scale_x, scale_y, order, L, N, sol, result):
    block_x = cuda.blockIdx.x
    block_y = cuda.blockIdx.y
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1

    scale = scale_x[block_x] * scale_y[block_y]

    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_y:
                inc = incs[block_x, block_y, (i - 1) >> order, (j - 1) >> order] * scale
                
                k_01 = 1.0 if i == 1 else sol[block_x, block_y, i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[block_x, block_y, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[block_x, block_y, i - 2, K3]
                               
                sol[block_x, block_y, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                
                if p == N - 1:
                    result[block_x, block_y] = sol[block_x, block_y, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
        
@cuda.jit
def sigpde_gram_symmetric(incs, length_x, off_x, off_y, M_grid, N_grid, order, L, N, sol, result):
    block_x = cuda.blockIdx.x
    block_y = cuda.blockIdx.y
    thread_id = cuda.threadIdx.x
    
    if block_y + off_y > block_x + off_x:
        return
    
    c_off = cache_offset(N_grid, M_grid, block_y)
    cx = M_grid - min(N_grid - (block_y + off_y), M_grid)
    sol = sol[c_off + block_x - cx,:,:]
    
    K1 = 0
    K2 = 2
    K3 = 1

    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_x:
                inc = incs[block_x, block_y, (i - 1) >> order, (j - 1) >> order]
                
                k_01 = 1.0 if i == 1 else sol[i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[i - 2, K3]
                               
                sol[i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                
                if p == N - 1:
                    result[block_x + off_x, block_y + off_y] = sol[i - 1, K1]
                    if block_x + off_x != block_y + off_y:
                        result[block_y + off_y, block_x + off_x] = sol[i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
        
        cuda.syncthreads()
        
@cuda.jit
def sigpde_gram_symmetric_scaled(incs, length_x, scale_x, scale_y, off_x, off_y, M_grid, N_grid, order, L, N, sol, result):
    block_x = cuda.blockIdx.x
    block_y = cuda.blockIdx.y
    thread_id = cuda.threadIdx.x
    
    if block_y + off_y > block_x + off_x:
        return
    
    c_off = cache_offset(N_grid, M_grid, block_y)
    cx = M_grid - min(N_grid - (block_y + off_y), M_grid)
    sol = sol[c_off + block_x - cx,:,:]
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    scale = scale_x[block_x] * scale_y[block_y]

    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_x:
                inc = incs[block_x, block_y, (i - 1) >> order, (j - 1) >> order] * scale
                
                k_01 = 1.0 if i == 1 else sol[i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[i - 2, K3]
                               
                sol[i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                
                if p == N - 1:
                    result[block_x + off_x, block_y + off_y] = sol[i - 1, K1]
                    if block_x + off_x != block_y + off_y:
                        result[block_y + off_y, block_x + off_x] = sol[i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
        
        cuda.syncthreads()