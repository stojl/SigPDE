from numba import cuda

@cuda.jit(device=True, inline=True)
def pde_step(k00, k01, k10, inc):
    return (k01 + k10) * (1. + 0.5 * inc + (1./12) * inc**2) - k00 * (1. - (1./12) * inc**2)

@cuda.jit(device=True, inline=True)
def cache_offset(n, m, j):
    s = 0
    for i in range(j):
        s += min(n - i, m)
    return s
