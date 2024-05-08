import jax
import jax.numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve

jax.config.update("jax_enable_x64", True)

y0 = np.array([-14.0, -15.0, 20.0])


def f(t, y, args):
    output = np.array(
        [
            10 * (y[1] - y[0]),
            28 * y[0] - y[1] - y[0] * y[2],
            -8 / 3 * y[2] + y[0] * y[1],
        ]
    )
    return output


term = ODETerm(f)
solver = Dopri5()
saveat = SaveAt(ts=np.linspace(0, 5, 200))
solution = diffeqsolve(term, solver, t0=0.0, t1=5, dt0=1e-2, y0=y0, saveat=saveat)


t = solution.ts
x, y, z = solution.ys.T
_, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(x, y, z)
plt.tight_layout()
plt.show()


plt.cla()
fig, ax = plt.subplots()
ax.plot(t, x, label=r"$x$")
ax.plot(t, y, label=r"$y$")
ax.plot(t, z, label=r"$z$")
ax.set_xlabel(r"$t$")
ax.legend()
plt.tight_layout()
plt.show()


plt.cla()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlim(-20, 20)
ax.set_ylim(-25, 20)
ax.set_zlim(0, 41)
line = ax.plot(x[0], y[0], z[0], marker="o")[0]
tx = ax.set_title(f"time:{t[0]:.2f}")


def update(j):
    line.set_data(solution.ys[:j, :2].T)
    line.set_3d_properties(solution.ys[:j, 2])
    tx.set_text(f"time:{t[j]:.2f}")
    return [line]


ani = animation.FuncAnimation(fig, update, frames=range(200), interval=25, repeat=False)
ani.save("../figures/lorenz.mp4")
