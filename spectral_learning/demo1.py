import time

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import optax
import orthax.chebyshev as cheb

from src._networks import Filter1d, MultiLayerPerceptron

jax.config.update("jax_enable_x64", True)
print(f"Precision: {jnp.array(1.0).dtype}")


class FilteredMLP1d(eqx.Module):
    mlp: MultiLayerPerceptron
    filter: Filter1d

    def __init__(self, *, key):
        self.mlp = MultiLayerPerceptron("scalar", 64, 2, "scalar", key=key)
        self.filter = Filter1d(mu=0.0, tau=1.0)

    def __call__(self, x):
        return self.filter(x) * self.mlp(x)


net = FilteredMLP1d(key=jr.key(7777))


def f(x):
    return np.sign(x)


N = 127

# empricial function
xx = cheb.chebpts2(N + 1)
ff = f(xx)

# coefficient domain
zz = np.linspace(0, 1, N + 1)


def pred(net, xx=xx):
    ff_pred_hat = jax.vmap(net)(zz)
    ff_pred = cheb.chebval(xx, ff_pred_hat)
    return ff_pred


# loss
def loss(net):
    ff_pred = pred(net)
    return ((ff_pred - ff) ** 2).mean()


# optimization
maxiter = 100000

print("adam..")
opt = jaxopt.OptaxSolver(
    loss, optax.adam(optax.cosine_decay_schedule(1e-3, maxiter)), maxiter=maxiter
)
tic = time.time()
net, state = opt.run(net)
toc = time.time()
print(f"Done! Elapsed time: {toc - tic:.2f}, Final Loss: {state.value:.3e}")


print("lbfgs..")
opt = jaxopt.LBFGS(loss, maxiter=maxiter, tol=1e-13)
tic = time.time()
net, state = opt.run(net)
toc = time.time()
print(f"Done! Elapsed time: {toc - tic:.2f}, Final Loss: {state.value:.3e}")

# error
xx_test = cheb.chebpts2(2 * N + 1)
ff_test = f(xx_test)
ff_pred = pred(net, xx_test)

print(f"Relative L2: {np.linalg.norm(ff_pred - ff_test) / np.linalg.norm(ff_test)}")
print(f"mu: {net.filter.mu}, tau: {net.filter.tau}")

plt.plot(xx_test, ff_test, label="Exact")
plt.plot(xx_test, ff_pred, ":", label="Prediction")
plt.xlabel(r"$x$")
plt.legend()
plt.tight_layout()
plt.savefig("demo1.pdf", dpi=300)
