import numpy as np


def first_derivative(grid):
    N = len(grid)
    a = grid[0]
    b = grid[-1]
    dx = 1 / ((float(N)) / (b - a))

    Tii = np.zeros(N)

    T_initial = np.diag(Tii)
    for i in range(1, N):
        for j in range(i):
            T_initial[i, j] = 1/dx * ((-1)**(i-j))/(i-j)
    T_final = T_initial - T_initial.T
    return T_final


def second_derivative(grid):
    N = len(grid)
    a = grid[0]
    b = grid[-1]
    coeff = (1. / ((-1) / (((float(N)) / (b - a)) ** 2)))

    Tii = np.zeros(N)

    Tii += coeff * ((np.pi ** 2.) / 3.)
    T_initial = np.diag(Tii)
    for i in range(1, N):
        for j in range(i):
            T_initial[i, j] = coeff * ((-1.) ** (i - j)) * (2. / ((i - j) ** 2))
    T_final = T_initial + T_initial.T - np.diag(Tii)
    return T_final


def first_derivative_fd(grid):
    import scipy.sparse as sp
    dx = (grid[-1] - grid[0])/(len(grid))
    coeffs = np.array([1/12, -2/3, 0, 2/3, -1/12])/dx

    fd_matrix = sp.diags(coeffs, np.arange(-2, 3, 1), shape=(len(grid), len(grid)))

    return fd_matrix.toarray()


def second_derivative_fd(grid):
    import scipy.sparse as sp
    dx = (grid[-1] - grid[0]) / (len(grid))
    coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])/(dx**2)

    fd_matrix = sp.diags(coeffs, np.arange(-2, 3, 1), shape=(len(grid), len(grid)))

    return fd_matrix.toarray()


def kron_sum(b, a):
    import scipy.sparse as sp
    '''Computes a Kronecker sum to build our Kronecker-Delta tensor product expression'''
    n_1 = a.shape[0]  # len of grid 1
    n_2 = b.shape[0]  # len of grid 2
    ident_1 = sp.identity(n_1)  # the identity matrix of grid 1
    ident_2 = sp.identity(n_2)  # the identity matrix of grid 2

    return sp.kron(a, ident_2) + sp.kron(ident_1, b)


def derivatives_2d(grid1, grid2, order):
    import scipy.sparse as sp
    grids = [grid1, grid2]

    if order == 1:
        der = [first_derivative(g) for g in grids]
    elif order == 2:
        der = [second_derivative(g) for g in grids]
    der_map = map(sp.csr_matrix, der)

    from functools import reduce
    d = reduce(kron_sum, der_map)
    return d


def fd_derivatives_2d(grid1, grid2, order):
    import scipy.sparse as sp
    grids = [grid1, grid2]

    if order == 1:
        der = [first_derivative_fd(g) for g in grids]
    elif order == 2:
        der = [second_derivative_fd(g) for g in grids]
    der_map = map(sp.csr_matrix, der)

    from functools import reduce
    d = reduce(kron_sum, der_map)
    return d


wvfn = np.load('2d_h3o2_new_def_600_points_no_cutoff.npz')['wvfns']

sp = np.linspace(-65, 65, 600)
roo = np.linspace(3.9, 5.8, 600)
X, Y = np.meshgrid(sp, roo)

derivative1_y = derivatives_2d(roo, sp, 1)
derivative1_x = derivatives_2d(sp, roo, 1)
derivative1_fd_x = fd_derivatives_2d(roo, sp, 1)
derivative1_fd_y = fd_derivatives_2d(sp, roo, 1)

import scipy.sparse as sp

derivative1_wvfn = sp.csr_matrix.dot(derivative1_y, sp.csr_matrix.dot(derivative1_x, wvfn[:, 0]))
derivative1_fd_wvfn = sp.csr_matrix.dot(derivative1_fd_y, sp.csr_matrix.dot(derivative1_fd_x, wvfn[:, 0]))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(4)
im1 = axes[0].contourf(X, Y, wvfn[:, 0].reshape((600, 600)))
im2 = axes[1].contourf(X, Y, derivative1_wvfn.reshape((600, 600)))
im3 = axes[2].contourf(X, Y, derivative1_fd_wvfn.reshape((600, 600)))
im4 = axes[3].contourf(X, Y, derivative1_wvfn.reshape((600, 600)) - derivative1_fd_wvfn.reshape((600, 600)))
fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
fig.colorbar(im3, ax=axes[2])
fig.colorbar(im4, ax=axes[3])
# plt.plot(x, eig[:, 0], label='wave function')
# plt.plot(x, np.dot(first_derivative(x), eig[:, 0]), label='1st derivative')
# plt.plot(x, first_derivative_fd(x, eig[:, 0]), label='1st derivative fd')
# plt.plot(x, np.dot(second_derivative(x), eig[:, 0]), label='2nd derivative')
# plt.plot(x, second_derivative_fd(x, eig[:, 0]), label='2nd derivative fd')
# plt.legend()
plt.show()


