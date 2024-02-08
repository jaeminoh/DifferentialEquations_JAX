from functools import partial

import jax
import jax.numpy as np
from jax.numpy.fft import (
    rfft, irfft, rfftfreq, fft, ifft, fftfreq
)
from jax.scipy.integrate import trapezoid
from scipy.integrate import RK45
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class vlasov_poisson:
    """
    Fourier discretization for the Vlasov Poisson equation on the periodic domain.
    Computational domain: [0,T] x [-X, X] x [-V, V].
    """
    dim: int = 2
    def __init__(self,
                 T: float = 1.5,
                 X: float = 0.5,
                 V: float = 2*np.pi,
                 Nx: int = 128,
                 Nv: int = 256):
        self.T = T
        self.X = X
        self.xx = np.linspace(-X, X, Nx+1)[1:]
        self.V = V
        self.vv = np.linspace(-V, V, Nv+1)
        self.Nx = Nx
        self.Nv = Nv
        self.x_ik2pi = 2j * np.pi * fftfreq(Nx, 2*X/Nx)
        self.v_ik2pi = 2j * np.pi * rfftfreq(Nv, 2*V/Nv)

    @staticmethod
    def gaussian(v):
        return np.exp(-0.5*v**2)/np.sqrt(2*np.pi)

    def f0(self, x, v):
        x = x[:,None]
        return self.gaussian(v) * (1 + 0.5 * np.cos(2*np.pi*x))

    def compute_E(self, f):
        """
        Computing (Fourier transformed) electric field.
        """
        rho = np.maximum(trapezoid(f, dx=2*self.V/self.Nv, axis=-1), 1e-8)
        m = rho.mean(-1)
        Ehat = np.hstack([0, rfft(rho-m)[1:] / self.x_ik2pi[1:self.Nx//2+1]])
        E = irfft(Ehat, self.Nx)
        return E

    @partial(jax.jit, static_argnums=(0,))
    def eqn(self, t, fhat):
        """
        Discretized Equation.
        This will be passed to ODE integrators (such as RK45).
        The variable t is necessary to be passed to.

        ToDo: applying rfft with 2/3 anti-aliasing rule.
        """
        fhat = fhat.reshape((self.Nx, -1)) # this may be omitted by using the "tree_math" package.
        f = irfft(ifft(fhat, axis=0), self.Nv+1)
        transport_x = - self.x_ik2pi[:,None] * fft(rfft(f * self.vv), axis=0)
        E = self.compute_E(f)
        transport_v = - self.v_ik2pi * fft(rfft(f * E[:,None]), axis=0)
        return (transport_x + transport_v).ravel()
    
    def solve(self):
        """
        ToDo: status bar.
        """
        f0hat = fft(rfft(self.f0(self.xx, self.vv)), axis=0).ravel()
        solver = RK45(self.eqn, 0., f0hat, self.T)
        while solver.status == "running":
            solver.step()
        t, fhat = solver.t, solver.y
        f = irfft(ifft(fhat, axis=0), self.Nv+1)
        return t, f
    

class kuramoto_sivashinsky:
    """
    Kuramoto-Sivashinsky equation on the periodic domain [0, X].
    u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx = 0.
    """
    dim: int = 1

    def __init__(self,
                 T: float = 1.,
                 dt: float = None,
                 X: float = 2*np.pi,
                 Nx: int = 256,
                 alpha: float = 100/16,
                 beta: float = 100/16**2,
                 gamma: float = 100/16**4,
                 ):
        self.T = T
        self.dt = dt
        self.X = X
        self.Nx = Nx
        self.xx = np.linspace(0., X, Nx, endpoint=False)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ik2pi = 2j * np.pi * rfftfreq(Nx, X/Nx)
        

    def u0(self, x):
        return np.cos(x) * (1 + np.sin(x))

    @partial(jax.jit, static_argnums=(0,))
    def eqn(self, t, uhat):
        u = irfft(uhat, self.Nx)
        uux = self.ik2pi * rfft(0.5 * u**2)
        uxx = self.ik2pi**2 * uhat
        uxxxx = self.ik2pi**4 * uhat
        return - self.alpha * uux - self.beta * uxx - self.gamma * uxxxx
    
    def solve(self):
        u0 = self.u0(self.xx)
        u0hat = rfft(u0)
        solver = RK45(self.eqn, 0., u0hat, self.T, max_step=self.dt)
        uhat_list = [u0hat]
        while solver.status == "running":
            solver.step()
            uhat_list.append(solver.y)
        return solver.t, uhat_list


if __name__ == "__main__":
    model = kuramoto_sivashinsky(T=1.5, Nx=1024, dt=5e-4)
    t, uhat_list = model.solve()
    uhat = np.stack(uhat_list)
    u = irfft(uhat, model.Nx)
    plt.imshow(u.T, origin='lower', aspect='auto', cmap='jet', extent=[0, model.T, model.X, 0])
    plt.title(f"time: {t}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("test.png")