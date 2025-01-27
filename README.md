# SigPDE

**SigPDE** is a Python package for efficient computation of the **untruncated signature kernel** by solving the Goursat PDE as described in the paper:

> @article{Salvi_2021,  
> title={The Signature Kernel Is the Solution of a Goursat PDE},  
> volume={3},  
> ISSN={2577-0187},  
> url={http://dx.doi.org/10.1137/20M1366794},  
> DOI={10.1137/20m1366794},  
> number={3},  
> journal={SIAM Journal on Mathematics of Data Science},  
> publisher={Society for Industrial & Applied Mathematics (SIAM)},  
> author={Salvi, Cristopher and Cass, Thomas and Foster, James and Lyons, Terry and Yang, Weixin},  
> year={2021},  
> month=jan,  
> pages={873–899}  
> }

The package also includes implementations of the robust, untruncated signature kernel as outlined in:

> @misc{chevyrev2022signaturemomentscharacterizelaws,  
> title={Signature moments to characterize laws of stochastic processes},  
> author={Ilya Chevyrev and Harald Oberhauser},  
> year={2022},  
> eprint={1810.10971},  
> archivePrefix={arXiv},  
> primaryClass={math.ST},  
> url={https://arxiv.org/abs/1810.10971}  
> }

Currently, **only GPU PyTorch implementations** are available.

---

## How to Use the Package

```python
import torch
import sigpde.torch as sig

# Parameters
batch_size = 30
length = 50
dim = 30
dyadic_order = 2

# Sample data on GPU
x = torch.randn((batch_size, length, dim), device='cuda')
y = torch.randn((batch_size, length, dim), device='cuda')

# Define kernels
static_kernel = sig.kernels.LinearKernel()
kernel = sig.SigPDE(static_kernel, dyadic_order)
robust_kernel = sig.RobustSigPDE(static_kernel, dyadic_order)

# Pairwise computation
kernel.pairwise(x, y)
kernel.pairwise(x)

robust_kernel.pairwise(x, y)
robust_kernel.pairwise(x)

# Gram matrix computation
kernel.gram(x, y)
kernel.gram(x)

robust_kernel.gram(x)
robust_kernel.gram(x, y)

# Compute normalization factors with the robust kernel
robust_kernel.normalization(x)

# Custom normalization function
# Must satisfy: f(x) <= x for all x >= 1
def my_normalizer(x):
    return 2 - 1 / x.pow(0.1)

# Pairwise computation with custom normalizer
robust_kernel.pairwise(x, normalizer=my_normalizer)
```

### References
1. [Salvi et al., 2021 - "The Signature Kernel Is the Solution of a Goursat PDE"](http://dx.doi.org/10.1137/20M1366794)
2. [Chevyrev & Oberhauser, 2022 - "Signature moments to characterize laws of stochastic processes"](https://arxiv.org/abs/1810.10971)
