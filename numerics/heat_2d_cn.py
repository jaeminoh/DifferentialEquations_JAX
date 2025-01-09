import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def tri_disc(N, a):
    M0r = np.zeros(N)
    M0r[0] = 1 + a
    M0r[1] = -a

    M1r = np.zeros(N)
    M1r[0] = -a
    M1r[1] = 1 + 2 * a
    M1r[2] = -a

    Mlr = np.zeros(N)
    Mlr[-2] = -a
    Mlr[-1] = 1 + a

    M = np.zeros((N, N))
    M[0] = M0r
    M[1] = M1r
    M[-1] = Mlr

    for i in range(2, N - 1):
        M[i] = np.roll(M[i - 1], 1)

    return M


def conv_mat(N, a, b):
    M1r = np.zeros(N * N)
    M1r[1] = a
    M1r[N] = a
    M1r[N + 1] = b
    M1r[N + 2] = a
    M1r[2 * N + 1] = a

    M = np.zeros(((N - 2) ** 2, N**2))
    M[0] = M1r
    jump = (N - 2) * (np.arange(1, (N - 2) ** 2) - 1) + 1
    for i in range(1, (N - 2) * (N - 2)):
        if i + 1 in jump[1:]:
            M[i] = np.roll(M[i - 1], 3)
        else:
            M[i] = np.roll(M[i - 1], 1)

    return M


def main(N: int = 50, dt: float = 1e-3, maxiter: int = 1000):
    # grid
    print(f"N: {N}, dt: {dt}, iters: {maxiter}")
    scale = 2.5

    x = scale * np.linspace(-1, 1, N) 
    y = scale * np.linspace(-1, 1, N) 

    h = 2 * scale / N
    k = dt / (2 * h**2)

    # initial condition - Gaussian
    d = 2
    sample_mean = np.array([0.0, 0.0]).reshape(1, d, 1) * scale

    tx = x
    ty = y
    gx, gy = np.meshgrid(tx, ty)
    samples = np.stack((gx, gy), axis=-1).reshape(N**2, d, 1)
    samples_ = samples - sample_mean
    conics = (
        np.linalg.inv(np.eye(d) * 0.05 * scale * scale).reshape(1, d, d).repeat(N**2, 0)
    )
    powers = -0.5 * np.matmul(samples_.transpose(0, 2, 1), np.matmul(conics, samples_))
    u0 = np.exp(powers).squeeze().reshape(N, N)  # * 0.5

    # discretization
    A = tri_disc(N, k)
    B = conv_mat(N + 2, k, 1 - 4 * k)
    C = tri_disc(N, k)
    print("2. Generate discretization matrices")

    def padding(u):
        u0_ = np.zeros_like(u[0])
        uN_ = np.zeros_like(u[-1])
        u = np.vstack([u0_, u, uN_])
        u_0 = np.zeros_like(u[:, 0])
        u_N = np.zeros_like(u[:, -1])
        u = np.concatenate([u_0[:, None], u, u_N[:, None]], 1)
        return u

    # Crank-Nicolson Method
    A_inv = np.linalg.inv(A)
    C_inv = np.linalg.inv(C)

    u_pred = [u0]
    u = u0
    u_star = np.zeros((N, N))

    print("3. Start iteration session")
    for it in tqdm(range(maxiter - 1)):
        # step 1
        # u_p = pad(u[None,None].ten).reshape(-1, 1)
        u_p = padding(u).reshape(-1, 1)
        S = B @ u_p
        S = S.reshape(N, N)
        u_star[1:-1] = S[1:-1] @ A_inv.T

        u_star[0, 0] = k * (u[1, 0] - 2 * u[0, 0] + u[0, 1]) + u[0, 0]
        u_star[0, -1] = k * (u[1, -1] - 2 * u[0, -1] + u[0, -2]) + u[0, -1]
        u_star[-1, 0] = k * (u[-2, 0] - 2 * u[-1, 0] + u[-1, 1]) + u[-1, 0]
        u_star[-1, -1] = k * (u[-2, -1] - 2 * u[-1, -1] + u[-1, -2]) + u[-1, -1]
        u_star[0, 1:-1] = (
            k * (-3 * u[0, 1:-1] + u[1, 1:-1] + u[0, :-2] + u[0, 2:]) + u[0, 1:-1]
        )
        u_star[-1, 1:-1] = (
            k * (-3 * u[-1, 1:-1] + u[-2, 1:-1] + u[-1, :-2] + u[-1, 2:]) + u[-1, 1:-1]
        )

        # step 2
        u = C_inv @ u_star.T

        u_pred.append(u)

    _, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
    print(u.shape)
    ax0.imshow(u_pred[0], vmin=0, vmax=1)
    ax1.imshow(u_pred[-1], vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(f"heat_cn_N{N}_t{maxiter * dt}.pdf")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
