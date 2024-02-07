# Kuramoto-Sivashinsky Equation.

The equation reads:
$$ u_t = - u_{xx} - u_{xxxx} - uu_{x}, $$
where $(t, x) \in (0, 150] \times (0, 32\pi)$ with the periodic boundary condition.
This equation is classified into stiff PDEs, since it has a form of 
$$ u_t = \mathcal{L} u + \mathcal{N}[u],$$
where $\mathcal{L}$ is a linear operator with high order (spatial) derivatives, and $\mathcal{N}$ is a nonlinear operator with lower order derivatives.
A lot of studies has been performed to analyze equations of such formulation.
Please refer to Kassam & Trefethen 2005, SISC.

In fact, the equation can be solved within $1$ second in my laptop, by combination of fast fourier transform and exponential time differencing Runge-Kutta time-stepping of order $4$.
However, it is notoriously difficult to solve with Physics-informed neural networks (PINNs).
There was a result in "Causality PINN" by S. Wang, yet the computational cost was exceedingly high compared to other problems.
He used $10$-layer (modified) multi-layer perceptron with $256$ hidden units, decomposed time domain into $10$ sub-pieces, and utilized multiple GPUs.