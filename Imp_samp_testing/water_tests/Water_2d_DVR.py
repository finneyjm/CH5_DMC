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

    T = kron_sum(kinetic)
    print('threw those matrices into our sparse matrix')
    return T


def get_pot(grid, grid1, grid2):
    pot = potential_bare_water(grid)
    # pot[pot>100000/har2wave] = 100000/har2wave
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
    return En, Eig, extraV, grid


from Potential.Water_monomer_pot_fns import dipole_h2o
num_points = 100
re = 0.95784 * ang2bohr
re2 = 0.95783997 * ang2bohr
anti = np.linspace(-0.85, 0.85, num_points)
sym = np.linspace(-0.85, 1.15, num_points) + (re + re2)/np.sqrt(2)
ang = np.deg2rad(104.1747712)
anti_gmat_one_over = 1/(1/OH_red - np.cos(ang)/m_O)
sym_gmat_one_over = 1/(1/OH_red + np.cos(ang)/m_O)
X, Y = np.meshgrid(anti, sym, indexing='ij')

en_wat, eig_wat, v, coords = run(anti, sym, anti_gmat_one_over, sym_gmat_one_over, water)

print(f'ground state energy = {en_wat[0]*har2wave}')

print(f'freq 1 = {(en_wat[1]-en_wat[0])*har2wave}')

print(f'freq 2 = {(en_wat[2]-en_wat[0])*har2wave}')


anti2 = np.zeros(300)
sym2 = np.linspace(-0.5, 0.85, 300) + (re + re2)/np.sqrt(2)
A = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
eh = np.matmul(np.linalg.inv(A), np.vstack((anti2.flatten(), sym2.flatten())))
r1 = eh[0]
r2 = eh[1]
structures = linear_combo_stretch_grid(r1, r2, water)
np.savez('test_water_structures_sym', grid=sym2, coords=structures)


import DMC_Tools as dt

mass = np.array([m_O, m_H, m_H])
ref = np.load('monomer_coords.npy')
MOM = dt.MomentOfSpinz(ref, mass)
ref = MOM.coord_spinz()
eck = dt.EckartsSpinz(ref, coords, mass, planar=True)
coords = np.ma.masked_invalid(eck.get_rotated_coords())


dips = dipole_h2o(coords)
freq = 3752.632316881249
freq_std = 1.0858576310252577

au_to_Debye = 1 / 0.3934303
conv_fac = 4.702e-7
km_mol = 5.33e6
conversion = conv_fac * km_mol

t_mom = np.zeros(3)
for dip in range(3):
    t_mom[dip] = np.dot(eig_wat[:, 2], dips[:, dip]*eig_wat[:, 0]*au_to_Debye)

intens = np.linalg.norm(t_mom)
print(f'intensity = {intens**2*conversion*freq} +/- {freq_std*intens**2*conversion}')

t_mom = np.zeros(3)
for dip in range(3):
    t_mom[dip] = np.dot(eig_wat[:, 1], dips[:, dip]*eig_wat[:, 0]*au_to_Debye)

intens = np.linalg.norm(t_mom)
freq = (en_wat[1]-en_wat[0])*har2wave
print(f'intensity = {intens**2*conversion*freq}')

print(np.dot(eig_wat[:, 1], Y.flatten()*eig_wat[:, 0]))

print(np.dot(eig_wat[:, 2], X.flatten()*eig_wat[:, 0]))
# print(f'intensity_full = {}')


np.savez('2d_anti_sym_stretch_water_wvfns', grid=[anti, sym], ground=eig_wat[:, 0],
         excite_anti=eig_wat[:, 2], excite_sym=eig_wat[:, 1])

import matplotlib.pyplot as plt

X, Y = np.meshgrid(anti, sym, indexing='ij')
fig, axes = plt.subplots(2, 2)
yeet0 = axes[0, 0].contourf(X, Y, v*har2wave)
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


