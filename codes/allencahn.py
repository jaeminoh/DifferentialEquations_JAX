import numpy as np
import scipy.linalg as sl
from tqdm import trange
import matplotlib.pyplot as plt

from cheb import cheb


# spatial grid
N = 50
D, x = cheb(N)
x = x[1:-1]
w = 0.53 * x + 0.47 * np.sin(-1.5 * np.pi * x) - x
u = np.hstack([1, w+x, -1])


# precomputing
h = 1/4
M = 32
r = 15 * np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
L = D[1:-1, 1:-1]
L = 0.01 * L @ L
A = h * L
E = sl.expm(A)
E2 = sl.expm(A / 2)
I = np.eye(N-1, dtype="complex128")
Z = np.zeros((N-1, N-1), dtype="complex128")
f1, f2, f3, Q = np.copy(Z), np.copy(Z), np.copy(Z), np.copy(Z)
for z in r:
    zIA = sl.inv(z*I - A)
    Q += h * zIA * (np.exp(z/2) - 1)
    f1+= h * zIA * (-4 - z + np.exp(z) * (4 - 3 * z + z**2)) / z**2
    f2+= h * zIA * (2 + z + np.exp(z) * (z - 2)) / z**2
    f3+= h * zIA * (-4 - 3 * z - z**2 + np.exp(z) * (4 - z)) / z**2

f1 = f1.real / M
f2 = f2.real / M
f3 = f3.real / M
Q  = Q.real  / M

uu = [u]
t = 0.
tt = [t]
tmax = 70
nmax = int(tmax / h)
nplt = 2


def step(t, w):
    t = t + h
    Nu = (w + x) - (w + x)**3
    a = E2 @ w + Q @ Nu
    Na = (a + x) - (a + x)**3
    b = E2 @ w + Q @ Na
    Nb = (b + x) - (b + x)**3
    c = E2 @ a + Q @ (2 * Nb - Nu)
    Nc = (c + x) - (c + x)**3
    w = E @ w + f1 @ Nu + 2 * f2 @ (Na + Nb) + f3 @ Nc
    return t, w

for n in trange(1, nmax + 1):
    t, w = step(t, w)
    if n % nplt == 0:
        u = np.hstack([1, w+x, -1])
        uu.append(u), tt.append(t)

tt = np.stack(tt)
uu = np.stack(uu)
_, ax = plt.subplots(figsize=(6,6), subplot_kw={"projection":"3d"})
xx = np.hstack([1, x, -1])
t, x = np.meshgrid(tt, xx, indexing="ij")
surf = ax.plot_surface(x,t, uu, cmap='jet')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$t$")
ax.view_init(elev=45, azim=-135)
plt.colorbar(surf)
plt.savefig("../figures/allencahn.png")