from numba import cuda

from sigpde.cuda_device_functions import (
    pde_step
)
       
@cuda.jit
def sigpde_pairwise(incs, length_x, length_y, order, L, N, sol, out):   
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
        
        
@cuda.jit(max_registers=40)
def sigpde_pairwise_norm_chandraputla(incs, norms, f_norms, length_x, order, L, N, sol_1, out, maxit, tol):
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    a = 0.0
    c = 0.0
    b = 1.0
    fa = 1.0 - norms[block_id]
    fb = f_norms[block_id] - norms[block_id]
    fc = fa
    t = 0.5
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    for _ in range(maxit):       
        xt = a + t * (b - a)
                
        for p in range(2, N):
            for l in range(L):
                i = thread_id * L + l + 1
                j = p - i
                
                if i < min(length_x, p) and j < length_x:
                    inc = incs[block_id, (i - 1) >> order, (j - 1) >> order] * xt**2
                    
                    k_01 = 1.0 if i == 1 else sol_1[block_id, i - 2, K2]
                    k_10 = 1.0 if j == 1 else sol_1[block_id, i - 1, K2]
                    k_00 = 1.0 if j == 1 or i == 1 else sol_1[block_id, i - 2, K3]
                                
                    sol_1[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                    
                    if p == N - 1:
                        sol_1[block_id, 0, 0] = sol_1[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
            
        ft = sol_1[block_id, 0, 0] - norms[block_id]
            
        if ft * fa >= 0:
            c = a
            fc = fa
        else:
            c = b
            b = a
            fc = fb
            fb = fa

        a = xt
        fa = ft
            
        if abs(fa) < abs(fb):
            if abs(fa) < tol:
                if thread_id == 0:
                    out[block_id] = a
                return
        else:
            if abs(fb) < tol:
                if thread_id == 0:
                    out[block_id] = b
                return
            
        xi = (a - b) / (c - b)
        phi = (fa - fb) / (fc - fb)
        
        if phi**2 < xi and (1 - phi)**2 < 1 - xi:
            t = fa / (fb - fa) * fc / (fb - fc) + (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb)
        else:
            t = 0.5
            
        if not (0 < t and t < 1):
            t = 0.5
    
@cuda.jit
def sigpde_pairwise_norm_init(incs, norms, f_norms, length_x, order, L, N, sol_1, out, maxit, tol):
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    if thread_id == 0:
        out[block_id, 0] = 0.0 #a
        out[block_id, 1] = 1.0 #b
        out[block_id, 2] = 1.0 - norms[block_id] #fa
        out[block_id, 3] = f_norms[block_id] - norms[block_id] #fb
        
    cuda.syncthreads()
    
    K1 = 0
    K2 = 2
    K3 = 1

    for _ in range(maxit):       
        c_1 = 0.5 * (out[block_id, 0] + out[block_id, 1])
        c_2 = (out[block_id, 0] * out[block_id, 3] - out[block_id, 1] * out[block_id, 2]) / (out[block_id, 3] - out[block_id, 2])
        c = (1 - 0.2) * c_1 + 0.2 * c_2
        
        for p in range(2, N):
            for l in range(L):
                i = thread_id * L + l + 1
                j = p - i
                
                if i < min(length_x, p) and j < length_x:
                    inc = incs[block_id, (i - 1) >> order, (j - 1) >> order] * c**2
                    
                    k_01 = 1.0 if i == 1 else sol_1[block_id, i - 2, K2]
                    k_10 = 1.0 if j == 1 else sol_1[block_id, i - 1, K2]
                    k_00 = 1.0 if j == 1 or i == 1 else sol_1[block_id, i - 2, K3]
                                
                    sol_1[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                    
                    if p == N - 1:
                        sol_1[block_id, 0, 0] = sol_1[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
            
        fc = sol_1[block_id, 0, 0] - norms[block_id]
            
        if thread_id == 0:       
            if fc * out[block_id, 2] < 0:
                out[block_id, 1], out[block_id, 3] = c, fc
            else:
                out[block_id, 0], out[block_id, 2] = c, fc
                
        cuda.syncthreads()
            
        if abs(fc) < tol:
            return
        
@cuda.jit
def sigpde_pairwise_norm(incs, norms, init, length_x, order, L, N, sol_1, out, maxit, tol):
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    a = init[block_id, 0]
    b = init[block_id, 1]
    fa = init[block_id, 2]
    fb = init[block_id, 3]
    
    K1 = 0
    K2 = 2
    K3 = 1

    for _ in range(maxit):
        c = b - fb * (b - a) / (fb - fa)
        
        if abs(fb) < tol:
            if thread_id == 0:
                out[block_id] = c
            break
        
        for p in range(2, N):
            for l in range(L):
                i = thread_id * L + l + 1
                j = p - i
                
                if i < min(length_x, p) and j < length_x:
                    inc = incs[block_id, (i - 1) >> order, (j - 1) >> order] * c**2
                    
                    k_01 = 1.0 if i == 1 else sol_1[block_id, i - 2, K2]
                    k_10 = 1.0 if j == 1 else sol_1[block_id, i - 1, K2]
                    k_00 = 1.0 if j == 1 or i == 1 else sol_1[block_id, i - 2, K3]
                                
                    sol_1[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                    
                    if p == N - 1:
                        sol_1[block_id, 0, 0] = sol_1[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
        
        a, fa = b, fb
        b, fb = c, sol_1[block_id, 0, 0] - norms[block_id]
        