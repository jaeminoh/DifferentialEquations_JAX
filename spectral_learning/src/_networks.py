from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray


class Filter1d(eqx.Module):
    """
    Differentiable filter.

    **Fields**

    -
    """

    mu: jax.Array
    tau: jax.Array

    def __init__(self, *, mu: float = 0.0, tau: float = 1.0):
        self.mu = jnp.array(mu)
        self.tau = jnp.array(tau)

    def __call__(self, x):
        return jnp.exp(-((self.tau * (x - self.mu)) ** 2))


class MultiLayerPerceptron(eqx.Module):
    """
    Multi-Layer Perceptron.
    w0 is a frequency hyper-parameter.
    """

    layers: list
    w0: jax.Array = eqx.field(static=True)

    def __init__(
        self,
        d_in: Union[str, int],
        width: int,
        depth: int,
        d_out: Union[str, int],
        *,
        w0: float = 10.0,
        key: PRNGKeyArray,
    ):
        layers = [d_in] + [width for _ in range(depth - 1)] + [d_out]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]
        self.w0 = jnp.array(w0)

    def __call__(self, inputs):
        inputs = jnp.sin(self.w0 * self.layers[0](inputs))
        for layer in self.layers[1:-1]:
            inputs = jnp.tanh(layer(inputs))
        return self.layers[-1](inputs)
