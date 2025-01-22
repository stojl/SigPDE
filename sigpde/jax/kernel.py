import jax
import jax.numpy as jnp

from sigpde.BatchIterator import BatchIterator
from sigpde.utils import sqrt_ceil
from sigpde.PDESolvers import (
    PairwisePDESolver,
    GramPDESolver,
    SymmetricGramPDESolver
)

class SigPDE:
    def __init__(self, static_kernel, dyadic_order=0):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order

    def pairwise(self, x, y=None, x_scale=None, y_scale=None, max_batch=1000, max_threads=1024):
        y = x if y is None else y
        y_scale = x_scale if y_scale is None and x_scale is not None else y_scale

        batch_size = x.shape[0]
        is_scaled = x_scale is not None or y_scale is not None

        if is_scaled:
            x_scale = jnp.ones(batch_size) if x_scale is None else x_scale
            y_scale = jnp.ones(batch_size) if y_scale is None else y_scale
            x_scale = jnp.broadcast_to(x_scale, (batch_size,)) if x_scale.ndim == 0 else x_scale
            y_scale = jnp.broadcast_to(y_scale, (batch_size,)) if y_scale.ndim == 0 else y_scale

        solver = PairwisePDESolver(
            batch_size,
            x.shape[1],
            y.shape[1],
            x.dtype,
            self.dyadic_order,
            max_batch,
            max_threads
        )

        result = jnp.zeros(batch_size, dtype=x.dtype)

        for _, start, stop in BatchIterator(batch_size, solver.batch_size):
            inc = pairwise_inner_product(x[start:stop, :, :], y[start:stop, :, :], self.static_kernel, self.dyadic_order)
            if is_scaled:
                solver.solve_scaled(inc, x_scale[start:stop], y_scale[start:stop], result[start:stop])
            else:
                solver.solve(inc, result[start:stop])

        return result

    def gram(self, x, y=None, x_scale=None, y_scale=None, max_batch=1000, max_threads=1024):
        max_batch = sqrt_ceil(max_batch)
        if y is None:
            return self._gram_sym(x, x_scale, max_batch, max_threads)
        else:
            return self._gram(x, y, x_scale, y_scale, max_batch, max_threads)

    def _gram(self, x, y, x_scale, y_scale, max_batch, max_threads):
        is_scaled = x_scale is not None or y_scale is not None
        batch_size_x = x.shape[0]
        batch_size_y = y.shape[0]

        if is_scaled:
            x_scale = jnp.ones(batch_size_x) if x_scale is None else x_scale
            y_scale = jnp.ones(batch_size_y) if y_scale is None else y_scale
            x_scale = jnp.broadcast_to(x_scale, (batch_size_x,)) if x_scale.ndim == 0 else x_scale
            y_scale = jnp.broadcast_to(y_scale, (batch_size_y,)) if y_scale.ndim == 0 else y_scale

        solver = GramPDESolver(
            batch_size_x,
            batch_size_y,
            x.shape[1],
            y.shape[1],
            x.dtype,
            self.dyadic_order,
            max_batch,
            max_threads
        )

        final_result = jnp.zeros((batch_size_x, batch_size_y), dtype=x.dtype)
        result = jnp.zeros((solver.batch_size_x, solver.batch_size_y), dtype=x.dtype)

        for _, start_y, stop_y in BatchIterator(batch_size_y, solver.batch_size_y):
            for _, start_x, stop_x in BatchIterator(batch_size_x, solver.batch_size_x):
                inc = gram_inner_product(
                    x[start_x:stop_x, :, :],
                    y[start_y:stop_y, :, :],
                    self.static_kernel,
                    self.dyadic_order
                )
                if is_scaled:
                    solver.solve_scaled(inc, x_scale[start_x:stop_x], y_scale[start_y:stop_y], result)
                else:
                    solver.solve(inc, result)

                final_result = final_result.at[start_x:stop_x, start_y:stop_y].set(result[:inc.shape[0], :inc.shape[1]])

        return final_result

    def _gram_sym(self, x, x_scale, max_batch, max_threads):
        is_scaled = x_scale is not None
        batch_size = x.shape[0]

        solver = SymmetricGramPDESolver(
            batch_size,
            x.shape[1],
            x.dtype,
            self.dyadic_order,
            max_batch,
            max_threads
        )

        result = jnp.zeros((batch_size, batch_size), dtype=x.dtype)

        for _, start_y, stop_y in BatchIterator(batch_size, solver.batch_size):
            for _, start_x, stop_x in BatchIterator(batch_size, solver.batch_size):
                inc = gram_inner_product(
                    x[start_x:stop_x, :, :],
                    x[start_y:stop_y, :, :],
                    self.static_kernel,
                    self.dyadic_order
                )
                if is_scaled:
                    solver.solve_scaled(inc, x_scale[start_x:stop_x], x_scale[start_y:stop_y], start_x, start_y, result)
                else:
                    solver.solve(inc, start_x, start_y, result)

        return result