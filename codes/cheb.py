import numpy as np


def cheb(N):
    if N==0:
        return 0, 1
    
    x = np.cos(
        np.pi * np.linspace(0, 1, N+1)
    )
    c = np.array([2] + [1]*(N-1) + [2]) * (-1 * np.ones(N+1))**np.arange(N+1)
    dX = x[:,None] - x
    D = c[:,None] / c / (dX + np.eye(N+1))
    D = D - np.diag(D.sum(1))
    return D, x