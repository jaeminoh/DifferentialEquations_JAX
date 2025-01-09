import time

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt
import numpy as np
from numpy.polynomial import chebyshev as cheb
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

class MLP(eqx.Module):
    """
    Multi-Layer Perceptron.
    w0 is a frequency hyper-parameter.
    """

    layers: list
    mu: jax.Array
    tau: jax.Array
    w0: float = eqx.field(static=True)

    def __init__(self, d_in, width, depth, d_out, *, w0=10.0, key):
        layers = [d_in] + [width for _ in range(depth - 1)] + [d_out]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]
        self.w0 = w0
        self.mu = jnp.array(0.0)
        self.tau = jnp.array(1.0)

    def __call__(self, inputs):
        gaussian = jnp.exp(- (self.tau * (inputs - self.mu))**2)
        inputs = jnp.sin(self.w0 * self.layers[0](inputs))
        for layer in self.layers[1:-1]:
            inputs = jnp.tanh(layer(inputs))
        return self.layers[-1](inputs) * gaussian
    

def f(x):
    return np.sin(6 * x) + np.sin(60 * np.exp(x))

cc = cheb.chebinterpolate(f, 150)
xx_test = np.linspace(-1, 1, 1000)
ff = f(xx_test)
reconstruction = cheb.chebval(xx_test, cc)
print(f"Interpolation: {np.linalg.norm(ff - reconstruction) / np.linalg.norm(ff):.3e}")

plt.semilogy(np.abs(cc))
plt.savefig("coefficients.png", dpi=300)

model = MLP("scalar", width=64, depth=3, d_out="scalar", key=jr.key(0))

xx = np.linspace(-1, 1, 151)

def loss(model):
    pred = jax.vmap(model)(xx)
    return ((pred - cc) ** 2).mean()
    #rel_loss = (pred - cc) / (jnp.abs(cc) + 1e-3)
    #return (rel_loss**2).mean()

print("Fitting a neural network..")
opt = jaxopt.LBFGS(loss, maxiter=10**5, tol=1e-13)
tic = time.time()
model, state = opt.run(model)
toc = time.time()
print(f"Done! Elapsed time: {toc - tic:.2f}s.")

pred = jax.vmap(model)(xx)
pred = cheb.chebval(xx_test, pred)
print(f"""Spectral Neural Network: {np.linalg.norm(pred - ff) / np.linalg.norm(ff):.3e}.
      mu: {model.mu},
      tau: {model.tau}.""")