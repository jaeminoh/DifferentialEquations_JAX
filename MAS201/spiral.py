import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
import matplotlib.pyplot as plt


# true dynamics
def f(t, y, args):
    del t, args
    y = jnp.array([[-0.1, -1.0], [1.0, -0.1]]) @ y
    return y


term = ODETerm(f)
solver = Dopri5()
t0 = 0.0
t1 = 20.0
saveat = SaveAt(ts=np.linspace(t0, t1, 200))
solution = diffeqsolve(
    term, solver, t0=t0, t1=t1, dt0=1e-2, y0=jnp.array([1.0, 1.0]), saveat=saveat
)

t = solution.ts
x, y = solution.ys.T
plt.scatter(x, y)
plt.savefig("figures/spiral_full", dpi=100)


def _draw(cut):
    plt.cla()
    plt.scatter(x[:cut], y[:cut])
    plt.savefig(f"figures/spiral_{cut}.png", format="png", dpi=100)


_draw(50)
_draw(100)

np.savez("data/spiral.npz", t=t, x=x, y=y)
