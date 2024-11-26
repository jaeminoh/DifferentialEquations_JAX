import matplotlib.pyplot as plt
import numpy as np
from cheb import cheb
from scipy.integrate import RK45

# spatial grid
N = 128
scale = 2.5
D, x = cheb(N)
D2 = D @ D
D = D[1:-1, 1:-1]
D2 = D2[1:-1, 1:-1]
x = x[1:-1]


def standard_normal(x, scale: float = 2.5):
    return np.exp(-2 * (x * scale) ** 2)


u0 = standard_normal(x)
print(u0[0], u0[-1])
plt.plot(x, u0, label="initial")


def burgers(u):
    u_x, u_xx = D @ u, D2 @ u
    return -u * (u_x / scale) + (u_xx / scale**2) / (10 * np.pi)


sol = RK45(lambda t, y: burgers(y), 0, u0, 1.5, rtol=1e-8, atol=1e-8)

while sol.status == "running":
    sol.step()
    print(f"t: {sol.t:.3f}")


plt.plot(x, sol.y, label=f"t={sol.t:.2f}")
plt.legend()
plt.savefig("burgers.pdf")
