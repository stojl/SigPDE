from numba import cuda

@cuda.jit(device=True, inline=True)
def dyadic_refinement_length(n, order):
    return ((n - 1) << order) + 1

@cuda.jit(device=True, inline=True)
def anti_diagonals(n, m):
    """
    n: Length of first sequence
    m: Length of second sequence
    """
    return n + m - 1

@cuda.jit(device=True, inline=True)
def thread_multiplicty(n, threads):
    """
    n: Length of sequence
    threads: Number of threads
    """
    return -(-(n - 1) // threads)


@cuda.jit(device=True, inline=True)
def device_linear_kernel(x, y, idx_x, idx_y, dim):
    S = 0
    for i in range(dim):
        S += (x[idx_x + 1, i] - x[idx_x, i]) * (y[idx_y + 1, i] - y[idx_y, i])
        
    return S

@cuda.jit(device=True, inline=True)
def pde_step(k00, k01, k10, inc):
    return (k01 + k10) * (1. + 0.5 * inc + (1./12) * inc**2) - k00 * (1. - (1./12) * inc**2)

@cuda.jit(device=True, inline=True)
def cache_offset(n, m, j):
    s = 0
    for i in range(j):
        s += min(n - i, m)
    return s
