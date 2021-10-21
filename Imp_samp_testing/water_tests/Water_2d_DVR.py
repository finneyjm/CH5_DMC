import numpy as np
from Coordinerds.CoordinateSystems import *

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
water = np.load('monomer_coords.npy')
order_w = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0]]
hydronium = np.flip(np.load('../../lets_go_girls/jobs/Prot_water_params/monomer_coords.npy'))
order_h = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2]]
order_t = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 4, 1, 2], [6, 4, 1, 2],
           [7, 0, 1, 2], [8, 7, 1, 2], [9, 7, 1, 2]]
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
OH_red = (m_O*m_H) / (m_O + m_H)
OD_red = (m_O*m_D) / (m_O + m_D)


def linear_combo_stretch_grid(r1, r2, coords):
    # re = np.linalg.norm(coords[0]-coords[1])
    # re2 = np.linalg.norm(coords[0]-coords[2])
    # re = 0.9616036495623883 * ang2bohr
    # re2 = 0.9616119936423067 * ang2bohr
    # re2 = re
    coords = np.array([coords] * 1)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).squeeze()
    zmat[:, 0, 1] = r1
    zmat[:, 1, 1] = r2
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


def potential_bare_water(grid):
    from Water_monomer_pot_fns import PatrickShinglePotential as pot
    V = pot(grid)
    return np.array(V)


def angle(coords):
    dists = oh_dists(coords)
    v1 = (coords[:, 1] - coords[:, 0]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
    v2 = (coords[:, 2] - coords[:, 0]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))

    ang1 = np.arccos(np.matmul(v1[:, None, :], v2[..., None]).squeeze())

    return ang1.T


def oh_dists(coords):
    bonds = [[1, 2], [1, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def Kinetic_Calc(grid1, grid2, red_m1, red_m2):
    import scipy.sparse as sp
    grids = [grid1, grid2]
    masses = [red_m1, red_m2]
    print('starting the kinetic energy calculation')

    def kin(grid, red_m):
        N = len(grid)
        a = grid[0]
        b = grid[-1]
        coeff = (1. / ((2. * red_m) / (((float(N) - 1.) / (b - a)) ** 2)))

        Tii = np.zeros(N)

        Tii += coeff * ((np.pi ** 2.) / 3.)
        T_initial = np.diag(Tii)
        for i in range(1, N):
            for j in range(i):
                T_initial[i, j] = coeff * ((-1.) ** (i - j)) * (2. / ((i - j) ** 2))
        T_final = T_initial + T_initial.T - np.diag(Tii)
        return T_final

    kinetic = [kin(g, m) for g, m in zip(grids, masses)]  # get a list of the two kinetic energy matrices
    print('done calculating kinetic energy')

    def kron_sum(der):
        '''Computes a Kronecker sum to build our Kronecker-Delta tensor product expression'''
        n_1 = len(der[1])  # len of grid 1
        ident_1 = sp.eye(n_1)  # the identity matrix of grid 1
        return sp.kron(sp.csr_matrix(der[0]), ident_1) + sp.kron(ident_1, sp.csr_matrix(der[1]))

    from functools import reduce
    T = kron_sum(kinetic)
    print('threw those matrices into our sparse matrix')
    return T


def get_pot(grid, grid1, grid2):
    pot = potential_bare_water(grid)
    # pot[pot>32000/har2wave] = 32000/har2wave
    import scipy.sparse as sp
    return sp.diags([pot], [0]), pot.reshape((len(grid1), len(grid2)))


def Energy(T, V, num_wvfns=5):
    import scipy.sparse as sp
    H = (T + V)
    print('starting the diagonalization')
    import scipy.sparse.linalg as la
    En, Eigv = la.eigsh(H, num_wvfns, which='SM')
    ind = np.argsort(En)
    En = En[ind]
    Eigv = Eigv[:, ind]
    return En, Eigv


def run(anti, sym, anti_mass, sym_mass, structure):
    A = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
    X, Y = np.meshgrid(anti, sym, indexing='ij')
    eh = np.matmul(np.linalg.inv(A), np.vstack((X.flatten(), Y.flatten())))
    r1 = eh[0]
    r2 = eh[1]
    grid = linear_combo_stretch_grid(r1, r2, structure)

    V, extraV = get_pot(grid, anti, sym)
    T = Kinetic_Calc(anti, sym, anti_mass, sym_mass)

    En, Eig = Energy(T, V)

    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    for i in range(Eig.shape[1]):
        Eig[:, i] = Eig[:, i].reshape((len(anti), len(sym))).T.flatten()
    return En, Eig, extraV


num_points = 100
re = 0.95784 * ang2bohr
re2 = 0.95783997 * ang2bohr
anti = np.linspace(-0.85, 0.85, num_points)
sym = np.linspace(-0.55, 1.15, num_points) + (re + re2)/np.sqrt(2)
ang = np.deg2rad(104.1747712)
anti_gmat_one_over = 1/(1/OH_red - np.cos(ang)/m_O)
sym_gmat_one_over = 1/(1/OH_red + np.cos(ang)/m_O)
en_wat, eig_wat, v = run(anti, sym, anti_gmat_one_over, sym_gmat_one_over, water)

print(f'ground state energy = {en_wat[0]*har2wave}')

print(f'freq 1 = {(en_wat[1]-en_wat[0])*har2wave}')

print(f'freq 2 = {(en_wat[2]-en_wat[0])*har2wave}')


np.savez('2d_anti_sym_stretch_water_wvfns', grid=[anti, sym], ground=eig_wat[:, 0],
         excite_anti=eig_wat[:, 1], excite_sym=eig_wat[:, 2])

import matplotlib.pyplot as plt

X, Y = np.meshgrid(anti, sym)
fig, axes = plt.subplots(2, 2)
yeet0 = axes[0, 0].contourf(X, Y, v.reshape((len(anti), len(sym))).T*har2wave)
axes[0, 0].set_xlabel('a (Bohr)')
axes[0, 0].set_ylabel('s (Bohr)')
fig.colorbar(yeet0, ax=axes[0, 0])

yeet1 = axes[0, 1].contourf(X, Y, eig_wat[:, 0].reshape((len(anti), len(sym))))
axes[0, 1].set_xlabel('a (Bohr)')
axes[0, 1].set_ylabel('s (Bohr)')
fig.colorbar(yeet1, ax=axes[0, 1])

yeet2 = axes[1, 0].contourf(X, Y, eig_wat[:, 1].reshape((len(anti), len(sym))))
axes[1, 0].set_xlabel('a (Bohr)')
axes[1, 0].set_ylabel('s (Bohr)')
fig.colorbar(yeet2, ax=axes[1, 0])

yeet3 = axes[1, 1].contourf(X, Y, eig_wat[:, 2].reshape((len(anti), len(sym))))
axes[1, 1].set_xlabel('a (Bohr)')
axes[1, 1].set_ylabel('s (Bohr)')
fig.colorbar(yeet3, ax=axes[1, 1])

plt.tight_layout()

plt.savefig('2d_anti_sym_wvfns')

plt.show()


