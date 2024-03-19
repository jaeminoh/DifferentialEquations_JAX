import jax
import jax.numpy as np
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray



class Sampler1d:
    def __init__(self, length, minval, maxval):
        self.length = length
        self.minval = minval
        self.maxval = maxval


class Uniform1d(Sampler1d):
    def __init__(self, *args):
        super().__init__(*args)
    
    def __call__(self,
                 key: PRNGKeyArray,
                 length,
                 minval,
                 maxval):
        sample = jr.uniform(key, shape=(length,), minval=minval, maxval=maxval)
        return sample


def stratifying(
    sampler: Sampler1d, divide: int, n: int
):
    sub_intervals = np.linspace(sampler.minval, sampler.maxval, divide + 1)

    def stratified_sampler(key):
        sample = jax.vmap(sampler, (None, None, 0, 0))(
            key, n, sub_intervals[:-1], sub_intervals[1:]
        )
        return sample.ravel()

    return stratified_sampler

if __name__ == "__main__":
    unif_sampler = Uniform1d(100, 0., 10.)
    seed = jr.PRNGKey(0)
    stratified = stratifying(unif_sampler, 10, 10)

    import matplotlib.pyplot as plt
    s1 = unif_sampler(seed, unif_sampler.length, unif_sampler.minval, unif_sampler.maxval)
    s2 = stratified(seed)

    _, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8,4))
    ax0.hist(s1, bins=10, density=True)
    ax0.set_title("unif")
    ax1.hist(s2, bins=10, density=True)
    ax1.set_title("stratified unif")
    plt.tight_layout()
    plt.savefig("test")