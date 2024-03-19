import jax.random as jr
import jax.numpy as np


def MLP(layers: list[int], act: callable = np.sin, w0: float = 4.0):
    """
    Multi-layer Perceptron with sine activation function, known as 'SIREN'.
    It has the same architecture as MLP, but different initialization is employed.
    w0 controls the frequency of the output.
    """

    def _siren_init(key, d_in, d_out, is_first=False):
        if is_first is True:
            variance = 1 / d_in
            W = jr.uniform(key, (d_in, d_out), minval=-variance, maxval=variance)
            std = np.sqrt(1 / d_in)
            b = jr.uniform(key, (d_out,), minval=-std, maxval=std)
            return W, b
        else:
            variance = np.sqrt(6 / d_in) / w0
            W = jr.uniform(key, (d_in, d_out), minval=-variance, maxval=variance)
            std = np.sqrt(1 / d_in)
            b = jr.uniform(key, (d_out,), minval=-std, maxval=std)
            return W, b

    def init(rng_key):
        _, *keys = jr.split(rng_key, len(layers))
        params = [_siren_init(keys[0], layers[0], layers[1], is_first=True)] + list(
            map(_siren_init, keys[1:], layers[1:-1], layers[2:])
        )
        return params

    def _activation(x):
        return act(w0 * x)

    def apply(params, inputs):
        # Forward Pass
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = _activation(outputs)
        # Final inner product
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply
