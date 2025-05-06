from numba import cuda
import math

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
def sigpde_pairwise_norm_chandrupatla(incs, norms, f_norms, length_x, order, L, N, sol_1, out, maxit, tol):
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    if math.isinf(f_norms[block_id]):
        return
    
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
            
        ft = float('inf') if math.isnan(sol_1[block_id, 0, 0]) else sol_1[block_id, 0, 0] - norms[block_id]
            
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
            if thread_id == 0:
                out[block_id] = a
            if abs(fa) < tol:
                return
        else:
            if thread_id == 0:
                out[block_id] = b
            if abs(fb) < tol:
                return
            
        xi = (a - b) / (c - b)
        phi = (fa - fb) / (fc - fb)
        
        if phi**2 < xi and (1 - phi)**2 < 1 - xi:
            t = fa / (fb - fa) * fc / (fb - fc) + (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb)
        else:
            t = 0.5
            
        if not (0 < t and t < 1):
            t = 0.5
            
@cuda.jit(max_registers=40)
def sigpde_pairwise_norm_chandrupatla_log(incs, norms, f_norms, length_x, order, L, N, sol_1, out, maxit, tol):
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    if math.isinf(f_norms[block_id]):
        return
    
    a = 0.0
    c = 0.0
    b = 1.0
    fa = -math.log(norms[block_id])
    fb = math.log(f_norms[block_id]) - math.log(norms[block_id])
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
            
        ft = float('inf') if math.isnan(sol_1[block_id, 0, 0]) else math.log(sol_1[block_id, 0, 0]) - math.log(norms[block_id])
            
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
            if thread_id == 0:
                out[block_id] = a
            if abs(fa) < tol:
                return
        else:
            if thread_id == 0:
                out[block_id] = b
            if abs(fb) < tol:
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
def sigpde_pairwise_norm_bisection(incs, norms, length_x, order, L, N, sol_1, out, maxit, tol):
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    a = 0.0
    b = 1.0
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    for _ in range(maxit):       
        xt = (a + b) / 2
                
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
                    
        if math.isnan(sol_1[block_id, 0, 0]):
            b = xt
        else:
            ft = sol_1[block_id, 0, 0] - norms[block_id]
            
            if abs(ft) < tol and thread_id == 0:
                out[block_id] = xt
                return
                
            if ft < 0:
                a = xt
            else:
                b = xt
                
                
@cuda.jit
def sigpde_pairwise_norm_newton_raphson(incs, norms, f_norms, length_x, order, L, N, sol, sol2, out, maxit, tol):
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    if math.isinf(f_norms[block_id]):
        return
    
    xt = out[block_id]
    c = math.log(norms[block_id])
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    for i in range(maxit):
        for p in range(2, N):
            for l in range(L):
                i = thread_id * L + l + 1
                j = p - i
                
                if i < min(length_x, p) and j < length_x:
                    inc = incs[block_id, (i - 1) >> order, (j - 1) >> order] * xt
                    
                    k_01 = 1.0 if i == 1 else sol[block_id, i - 2, K2]
                    k_10 = 1.0 if j == 1 else sol[block_id, i - 1, K2]
                    k_00 = 1.0 if j == 1 or i == 1 else sol[block_id, i - 2, K3]
                    
                    k_01_d = 0.0 if i == 1 else sol2[block_id, i - 2, K2]
                    k_10_d = 0.0 if j == 1 else sol2[block_id, i - 1, K2]
                    k_00_d = 0.0 if j == 1 or i == 1 else sol2[block_id, i - 2, K3]
                                
                    sol[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, xt * inc)
                    sol2[block_id, i - 1, K1] = pde_step(k_00_d, k_01_d, k_10_d, xt * inc) + 0.5 * inc * (k_01 + k_10 + k_00 + sol[block_id, i - 1, K1])
                    
                    if p == N - 1:
                        sol[block_id, 0, 0] = sol[block_id, i - 1, K1]
                        sol2[block_id, 0, 0] = sol2[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
            
        f = sol[block_id, 0, 0]
        df = sol2[block_id, 0, 0]
        
        if math.isnan(f) or math.isnan(df) or f <= 0:
            xt = xt * 0.5
        else:
            if abs(math.log(f) - c) < tol:
                return
            
            if abs(df) < tol:
                return
            
            xt = xt - (math.log(f) - c) * f / df                
            
        if thread_id == 0:
            out[block_id] = xt
                
        cuda.syncthreads()
        
@cuda.jit
def sigpde_pairwise_norm_secant(incs, norms, f_norms, length_x, order, L, N, sol, out, maxit, tol):
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    if math.isinf(f_norms[block_id]):
        return
    
    c = math.log(norms[block_id])
    
    x0 = out[block_id]
    x1 = out[block_id] + 0.01
    f0 = f_norms[block_id]
        
    K1 = 0
    K2 = 2
    K3 = 1
    
    for i in range(maxit):
        for p in range(2, N):
            for l in range(L):
                i = thread_id * L + l + 1
                j = p - i
                
                if i < min(length_x, p) and j < length_x:
                    inc = incs[block_id, (i - 1) >> order, (j - 1) >> order] * x1**2
                    
                    k_01 = 1.0 if i == 1 else sol[block_id, i - 2, K2]
                    k_10 = 1.0 if j == 1 else sol[block_id, i - 1, K2]
                    k_00 = 1.0 if j == 1 or i == 1 else sol[block_id, i - 2, K3]
                    
                    sol[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc)
                    
                    if p == N - 1:
                        sol[block_id, 0, 0] = sol[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
            
        f = sol[block_id, 0, 0]
        
        if math.isnan(f) or f <= 0:
            x0, x1 = x1 * 0.5, x1 * 0.5 + tol
        else:
            if abs(math.log(f) - c) < tol:
                return
            
            if abs(math.log(f0) - math.log(f)) < tol:
                return
            
            x0, x1 = x1, x1 - (math.log(f) - c) * (x1 - x0) / (math.log(f) - math.log(f0))
            f0 = f
            
        if thread_id == 0:
            out[block_id] = x0
                
        cuda.syncthreads()
                
@cuda.jit
def sigpde_pairwise_scale_derivative(incs, length_x, scale_x, order, L, N, sol, sol2, out, out2):   
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    scale = scale_x[block_id]
    
    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_x:
                inc = incs[block_id, (i - 1) >> order, (j - 1) >> order] * scale
                
                k_01 = 1.0 if i == 1 else sol[block_id, i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[block_id, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[block_id, i - 2, K3]
                
                k_01_d = 0.0 if i == 1 else sol2[block_id, i - 2, K2]
                k_10_d = 0.0 if j == 1 else sol2[block_id, i - 1, K2]
                k_00_d = 0.0 if j == 1 or i == 1 else sol2[block_id, i - 2, K3]
                               
                sol[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, scale * inc)
                sol2[block_id, i - 1, K1] = pde_step(k_00_d, k_01_d, k_10_d, scale * inc) + 0.5 * inc * (k_01 + k_10 + k_00 + sol[block_id, i - 1, K1])
                
                if p == N - 1:
                    out[block_id] = sol[block_id, i - 1, K1]
                    out2[block_id] = sol2[block_id, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
        
'''     
@cuda.jit
def sigpde_pairwise_scale_curvature(incs, length_x, scale_x, order, L, N, sol, sol2, sol3, out, out2, out3):   
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    K1 = 0
    K2 = 2
    K3 = 1
    
    scale = scale_x[block_id]
    
    for p in range(2, N):
        for l in range(L):
            i = thread_id * L + l + 1
            j = p - i
            
            if i < min(length_x, p) and j < length_x:
                inc = incs[block_id, (i - 1) >> order, (j - 1) >> order]
                
                k_01 = 1.0 if i == 1 else sol[block_id, i - 2, K2]
                k_10 = 1.0 if j == 1 else sol[block_id, i - 1, K2]
                k_00 = 1.0 if j == 1 or i == 1 else sol[block_id, i - 2, K3]
                
                k_01_d = 0.0 if i == 1 else sol2[block_id, i - 2, K2]
                k_10_d = 0.0 if j == 1 else sol2[block_id, i - 1, K2]
                k_00_d = 0.0 if j == 1 or i == 1 else sol2[block_id, i - 2, K3]
                
                k_01_c = 0.0 if i == 1 else sol3[block_id, i - 2, K2]
                k_10_c = 0.0 if j == 1 else sol3[block_id, i - 1, K2]
                k_00_c = 0.0 if j == 1 or i == 1 else sol3[block_id, i - 2, K3]
                               
                sol[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, scale * scale * inc)
                sol2[block_id, i - 1, K1] = (
                    pde_step(k_00_d, k_01_d, k_10_d, scale * scale * inc) + 
                    0.5 * scale * inc * (k_01 + k_10 + k_00 + sol[block_id, i - 1, K1])
                )
                sol3[block_id, i - 1, K1] = (
                    pde_step(k_00_c, k_01_c, k_10_c, scale * scale * inc) + 
                    0.5 * inc * (k_01 + k_10 + k_00 + sol[block_id, i - 1, K1]) +
                    scale * inc * (k_01_d + k_10_d + k_00_d + sol2[block_id, i - 1, K1])
                )
                
                if p == N - 1:
                    out[block_id] = sol[block_id, i - 1, K1]
                    out2[block_id] = sol2[block_id, i - 1, K1]
                    out3[block_id] = sol3[block_id, i - 1, K1]

        K1 = K1 ^ K2 ^ K3
        K2 = K1 ^ K2 ^ K3
        K3 = K1 ^ K2 ^ K3
        K1 = K1 ^ K2 ^ K3
            
        cuda.syncthreads()
'''