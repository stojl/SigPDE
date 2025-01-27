from math import ceil, sqrt

def anti_diagonals(n, m):
    return n + m - 1

def dyadic_refinement_length(length, order):
    return ((length - 1) << order) + 1

def ceil_div(a, b):
    return -(-a // b)

def round_to_multiple_of_32(x):
    return ((x + 31) // 32) * 32

def tensor_type(x):
    return x.__cuda_array_interface__["typestr"]
    
def sqrt_ceil(x):
    return ceil(sqrt(x))