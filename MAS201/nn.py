import equinox as eqx
import jax.random as jr
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve


class MLP(eqx.Module):
    layers: list
    activation: callable = eqx.field(static=True)

    def __init__(self, d_in, width, depth, d_out, *, activation, key):
        layers = [d_in] + [width for _ in range(depth - 1)] + [d_out]
        keys = jr.split(key, depth)
        self.layers = [
            eqx.nn.Linear(_in, _out, key=_k)
            for _in, _out, _k in zip(layers[:-1], layers[1:], keys)
        ]
        self.activation = activation

    def __call__(self, t, y, args):
        del t, args
        inputs = y
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        return self.layers[-1](inputs)


class NeuralOde(eqx.Module):
    func: MLP

    def __init__(self, d_in, width, depth, d_out, *, activation, key):
        self.func = MLP(d_in, width, depth, d_out, activation=activation, key=key)

    def __call__(self, ts, y0):
        solution = diffeqsolve(
            ODETerm(self.func),
            Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            saveat=SaveAt(ts=ts),
        )
        return solution.ys
