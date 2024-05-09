import time
import fire
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
import matplotlib.pyplot as plt

from nn import NeuralOde


def main(cut: int):
    with np.load("data/spiral.npz", allow_pickle=True) as f:
        t, x, y = f["t"], f["x"], f["y"]
        key1, key2 = jr.split(jr.key(1234))
        x_train = (x + jr.normal(key1, x.shape) * 0.05)[:cut]
        y_train = (y + jr.normal(key2, y.shape) * 0.05)[:cut]

    model = NeuralOde(2, 128, 2, 2, activation=jnp.tanh, key=jr.key(1234))

    def loss(model, ts=t[:cut]):
        y0 = jnp.stack([x_train[0], y_train[0]])
        pred = model(ts, y0)
        return jnp.mean((jnp.stack([x_train, y_train]) - pred.T) ** 2)

    opt = optax.adam(1e-3)
    state = opt.init(model)

    @jax.jit
    def step(model, state):
        v, g = jax.value_and_grad(loss)(model)
        updates, state = opt.update(g, state, model)
        model = optax.apply_updates(model, updates)
        return model, state, v

    tic = time.time()
    for it in range(1, 1 + 1000):
        model, state, v = step(model, state)
        if it % 100 == 0:
            toc = time.time()
            print(f"it: {it}, loss: {v}, time: {toc - tic:.2f}")

    y0 = jnp.stack([x[0], y[0]])
    pred = model(t, y0)
    xs, ys = pred.T
    plt.scatter(x, y, label="true")
    plt.scatter(xs, ys, label="pred", alpha=0.7)
    plt.scatter(x_train, y_train, label="data", alpha=0.3)
    plt.legend()
    plt.savefig(f"figures/neural_ode_{cut}.png", format="png", dpi=100)


if __name__ == "__main__":
    fire.Fire(main)
