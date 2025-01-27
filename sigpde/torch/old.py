@cuda.jit
def sigpde_pairwise_norm(incs, norms, f_norms, length_x, order, L, N, sol_1, out, maxit, init_tol, tol):
    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    
    fa = 1.0 - norms[block_id]
    fb = f_norms[block_id] - norms[block_id]
    
    a = 0.0
    b = 1.0
    c = 0.0
    
    K1 = 0
    K2 = 2
    K3 = 1

    for _ in range(maxit):
        c_1 = 0.5 * (a + b)
        c_2 = (a * fb - b * fa) / (fb - fa)
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
        
        if (sol_1[block_id, 0, 0] - norms[block_id]) * fa < 0:
            b, fb = c, sol_1[block_id, 0, 0] - norms[block_id]
        else:
            a, fa = c, sol_1[block_id, 0, 0] - norms[block_id]
            
        if abs(sol_1[block_id, 0, 0] - norms[block_id]) < init_tol:
            break
                           
    # Secant method
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
                    inc = incs[block_id, (i - 1) >> order, (j - 1) >> order]
                    
                    k_01 = 1.0 if i == 1 else sol_1[block_id, i - 2, K2]
                    k_10 = 1.0 if j == 1 else sol_1[block_id, i - 1, K2]
                    k_00 = 1.0 if j == 1 or i == 1 else sol_1[block_id, i - 2, K3]               
                                                   
                    sol_1[block_id, i - 1, K1] = pde_step(k_00, k_01, k_10, inc * c**2)
                    
                    if p == N - 1:
                        sol_1[block_id, 0, 0] = sol_1[block_id, i - 1, K1]

            K1 = K1 ^ K2 ^ K3
            K2 = K1 ^ K2 ^ K3
            K3 = K1 ^ K2 ^ K3
            K1 = K1 ^ K2 ^ K3
                
            cuda.syncthreads()
            
        a, fa = b, fb
        b, fb = c, sol_1[block_id, 0, 0] - norms[block_id]