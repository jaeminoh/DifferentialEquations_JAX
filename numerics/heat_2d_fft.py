import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftfreq, ifft2
from scipy.integrate import solve_ivp


def main(N: int = 30, T: float = 1.0):
    d = 2
    scale = 2.5

    sample_mean = np.array([0.0, 0.0]).reshape(1, d, 1) * scale

    tx = np.linspace(-1, 1, N) * scale
    ty = np.linspace(-1, 1, N) * scale
    gx, gy = np.meshgrid(tx, ty)
    samples = np.stack((gx, gy), axis=-1).reshape(N * N, d, 1)
    samples_ = samples - sample_mean
    conics = (
        np.linalg.inv(np.eye(d) * 0.05 * scale * scale)
        .reshape(1, d, d)
        .repeat(N * N, 0)
    )
    powers = -0.5 * np.matmul(samples_.transpose(0, 2, 1), np.matmul(conics, samples_))
    data = np.exp(powers).squeeze().reshape(N, N)  # * 0.5

    print(data.shape)

    u0 = data[:-1, :-1]
    k = fftfreq(N - 1, 2 * scale / (N - 1)) * 2j * np.pi
    k2 = k**2
    u0hat = fft2(u0)

    def heat(t, uhat):
        return ((k2 + k2.reshape(-1, 1)) * uhat.reshape(N - 1, N - 1)).ravel()
    
    print("solving...")
    tic = time.time()
    sol = solve_ivp(
        heat,
        [0, 1],
        u0hat.ravel(),
        method="RK45",
        t_eval=np.linspace(0, 1, 10),
    )
    toc = time.time()
    print(f"Done! Elapsed time: {toc - tic:.2f}s")

    u = ifft2(sol.y[:, -1].reshape(N - 1, N - 1)).real
    print(data[0])

    _, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
    ax0.imshow(data, vmin=0, vmax=1)
    ax1.imshow(u, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(f"heat_fft_N{N}_t{T}.pdf")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
