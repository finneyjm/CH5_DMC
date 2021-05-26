import numpy as np
from ProtWaterPES import *
from Coordinerds.CoordinateSystems import *

oxy_pos = 4.70153912
new_pos = oxy_pos

struct = np.array([
    [2.06095307, 0.05378083, 0.],
    [0., 0., 0.],
    [-0.32643038, -1.70972841, 0.52193868],
    [new_pos, 0., 0.],
    [5.20071798-oxy_pos+new_pos, 0.80543847, 1.55595785]
])

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
m_red_D = (m_O*m_D)/(m_O+m_D)
m_red_sp = 1/(1/m_H + 1/(2*m_O + 2*m_H))
m_red_OO = (m_O**2)/(2*m_O)
omega = 3600./har2wave

new_struct = np.array([
    [2.30803545e+00, -3.02071334e-03, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [-4.65570340e-01, 1.67058080e+00, -5.46666468e-01],
    [4.61607485e+00, 0.00000000e+00, 0.00000000e+00],
    [5.12936209e+00, -8.18802009e-01, -1.54030505e+00]
])

new_struct = np.array([
    [0.000000000000000, 0.000000000000000, 0.000000000000000],
    [-2.304566686034061, 0.000000000000000, 0.000000000000000],
    [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
    [2.304566686034061, 0.000000000000000, 0.000000000000000],
    [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233]
])
new_struct[:, 0] = new_struct[:, 0] + 2.304566686034061


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def asym_grid(coords, r1, a):
    coords = np.array([coords]*1)
    coords = coords[:, (1, 3, 0, 2, 4)]
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0], [3, 0, 1, 2],
                                                                                   [4, 1, 0, 2]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).reshape((N, 4, 6))
    zmat[:, 2, 1] = r1
    zmat[:, 3, 1] = r1 - a
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    coords = new_coords[:, (2, 0, 3, 1, 4)]
    return coords


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    s_oh = 1/np.sqrt(2)*(dis[:, 0]+dis[:, 1])
    mid = dis[:, 2]/2
    sp = mid - dis[:, -1]*np.cos(roh_roo_angle(coords, dis[:, -2], dis[:, -1]))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], s_oh, dis[:, -2], sp)).T


def roh_roo_angle(coords, roo_dist, roh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (coords[:, 1]-coords[:, 0])/np.broadcast_to(roh_dist[:, None], (len(roh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


def shared_prot_grid(coords, sp):
    # coords = np.array([coords] * len(sp))
    mid = (coords[:, 3, 0] - coords[:, 1, 0])/2
    coords[:, 0, 0] = mid-sp
    return coords


def oo_grid(coords, Roo):
    coords = np.array([coords] * len(Roo))
    equil_roo_roh_x = coords[0, 3, 0] - coords[0, 4, 0]
    # equil_roh_x = coords[0, 4, 0]
    coords[:, 3, 0] = Roo
    coords[:, 4, 0] = Roo - equil_roo_roh_x
    # coords = coords[:, (1, 3, 0, 2, 4)]
    # zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
    #                                                                     ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
    #                                                                                [2, 0, 1, 0], [3, 0, 1, 2],
    #                                                                                [4, 1, 0, 2]])).coords
    # N = len(Roo)
    # zmat = np.array([zmat] * N).reshape((N, 4, 6))
    # zmat[:, 0, 1] = Roo
    # new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    # coords = new_coords[:, (2, 0, 3, 1, 4)]
    return coords


def pot(coords, grid1, grid2):
    print('started making our grid')
    mesh = np.array(np.meshgrid(grid1, grid2))
    gridz = np.reshape(mesh, (2, len(grid1)*len(grid2)))
    roo_coords = oo_grid(coords, gridz[1])
    full_coords = shared_prot_grid(roo_coords, gridz[0])
    print('finished making the grid, now to start the potential')
    mid = (full_coords[:, 3, 0] - full_coords[:, 1, 0])/2
    full_coords[:, :, 0] -= mid[:, None]
    pot = get_pot(full_coords)
    # pot[pot > 12000/har2wave] = 12000/har2wave
    print('finished evaluating the potential')
    import scipy.sparse as sp
    return sp.diags([pot], [0]), pot.reshape((len(grid1), len(grid2)))


def HO_pots(mass, grid1, grid2):
    mesh = np.array(np.meshgrid(grid1, grid2))
    gridz = np.reshape(mesh, (2, len(grid1) * len(grid2)))
    pot = 1/2*mass*omega**2*gridz[0]**2 + 1/2*mass*omega**2*gridz[1]**2
    coupling = 1/2*mass*(550/har2wave)**2*(gridz[0]*gridz[1])
    pot = pot + coupling
    import scipy.sparse as sp
    return sp.diags([pot], [0]), pot.reshape((len(grid1), len(grid2)))


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
    kinetic_map = map(sp.csr_matrix, kinetic)  # provide a map iterable for the sparse matrix

    def kron_sum(b, a):
        '''Computes a Kronecker sum to build our Kronecker-Delta tensor product expression'''
        n_1 = a.shape[0]  # len of grid 1
        n_2 = b.shape[0]  # len of grid 2
        ident_1 = sp.identity(n_1)  # the identity matrix of grid 1
        ident_2 = sp.identity(n_2)  # the identity matrix of grid 2

        # returns the sum of a sparse kin matrix 1 with a completely block diagonal kin matrix 2
        return sp.kron(a, ident_2) + sp.kron(ident_1, b)

    from functools import reduce
    T = reduce(kron_sum, kinetic_map)
    print('threw those matrices into our sparse matrix')
    return T


def Energy(T, V, num_wvfns=20):
    H = (T + V)
    print('starting the diagonalization')
    import scipy.sparse.linalg as la
    En, Eigv = la.eigsh(H, num_wvfns, which='SM')
    ind = np.argsort(En)
    En = En[ind]
    Eigv = Eigv[:, ind]
    return En, Eigv


def run(grid1, grid2, mass1, mass2):
    print('starting DVR run')
    V, extraV = pot(new_struct, grid1, grid2)
    # V, extraV = HO_pots(mass1, grid1, grid2)
    print('put the potential in a sparse matrix')
    T = Kinetic_Calc(grid1, grid2, mass1, mass2)
    En, Eig = Energy(T, V)
    print('whew! done with DVR!')
    print(f'ground state energy = {En[0] * har2wave}')
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    print(f'frequency of first transition = {(En[1] - En[0]) * har2wave}')
    return En, Eig, extraV


num_points1 = 350
num_points2 = 350
small_grid_points = 100
blah = np.loadtxt('wf_Rz_60_60')
xh_a = -1.492883667487176
xh_b = 1.492883667487176
roo_a = 3.892916571908299
roo_b = 5.801579551339067
xh_a = blah[0, 1]
xh_b = blah[-1, 1]
roo_a = blah[0, 0]
roo_b = blah[-1, 0]
xh = blah[:, 1].reshape((60, 60))[:, 0]
roo = blah[:, 0].reshape((60, 60))[0]
roo = np.linspace(3.9, 5.8, 100)
xh = np.linspace(-1.5, 1.5, 100)
# en, eig, V = run(np.linspace(-1, 1, num=num_points1), np.linspace(-1, 1, num=num_points2), m_red, m_red)
en, eig, V = run(xh, roo, m_red_sp, m_red_OO)
# np.savez('small_grid_2d_h3o2', energies=en, wvfns=eig, pot=V)
np.savez('small_grid_2d_h3o2_bigger_grid', energies=en, wvfns=eig, pot=V)
# np.savez('test_2d_HO', energies=en, wvfns=eig, pot=V)
# two_d = np.load('first_2d_h3o2.npz')
small_grid = np.load('small_grid_2d_h3o2_bigger_grid.npz')
# two_d = np.load('test_2d_HO.npz')
# energies = two_d['energies']*har2wave
small_energies = small_grid['energies']*har2wave
# freq1 = energies[1] - energies[0]
# freq2 = energies[2] - energies[0]
# freq3 = energies[3] - energies[2]

small_freq1 = small_energies[1] - small_energies[0]
small_freq2 = small_energies[2] - small_energies[0]
small_freq3 = small_energies[3] - small_energies[0]

# print(f'freq1 diff = {freq1 - small_freq1}')
# print(f'freq2 diff = {freq2 - small_freq2}')
# print(f'freq3 diff = {freq3 - small_freq3}')
print(f'freq1 = {small_freq1}')
print(f'freq2 = {small_freq2}')
print(f'freq3 = {small_freq3}')
print(f'energies = {small_energies[0]}')
print(f'energies = {small_energies[1]}')
print(f'energies = {small_energies[2]}')
print(f'energies = {small_energies[3]}')
print(f'energies = {small_energies[6]}')
print(f'freq 6 = {small_energies[6]-small_energies[0]}')


# wvfns = two_d['wvfns']
# V = two_d['pot']*har2wave
import matplotlib.pyplot as plt
# sp = np.linspace(xh_a, xh_b, num=num_points1)/ang2bohr
# Roo = np.linspace(roo_a, roo_b, num=num_points2)/ang2bohr
# X, Y = np.meshgrid(sp, Roo)

# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, V)
# fig.colorbar(tcc)
# ax.set_ylabel(r'R$_{\rm{OO}} \AA$')
# ax.set_xlabel(r'sp $\AA$')
# plt.show()
# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, wvfns[:, 3].reshape(num_points1, num_points2))
# ax.set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$')
# ax.set_xlabel(r'XH $\rm\AA$')
# fig.colorbar(tcc)
# plt.show()


wvfns2 = small_grid['wvfns']
V2 = small_grid['pot']*har2wave
# import matplotlib.pyplot as plt
sp = np.linspace(xh_a, xh_b, num=small_grid_points)/ang2bohr
Roo = np.linspace(roo_a, roo_b, num=small_grid_points)/ang2bohr
X, Y = np.meshgrid(sp, Roo)


fig, ax = plt.subplots()
# ind = np.argwhere(V2 > 15000)
V2[V2 > 15000] = 15000
tcc = ax.contourf(X, Y, V2)
fig.colorbar(tcc)
ax.set_ylabel(r'R$_{\rm{OO}} \AA$')
ax.set_xlabel(r'sp $\AA$')
plt.show()

fig, ax = plt.subplots()
tcc = ax.contourf(X, Y, wvfns2[:, 0].reshape(small_grid_points, small_grid_points))
ax.set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax.set_xlabel(r'XH $\rm\AA$', fontsize=16)
plt.tight_layout()
fig.colorbar(tcc)
plt.show()

fig, ax = plt.subplots(3, 3)
tcc = ax[0, 0].contourf(X, Y, wvfns2[:, 0].reshape(small_grid_points, small_grid_points))
ax[0, 0].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[0, 0].set_xlabel(r'XH $\rm\AA$', fontsize=16)

tcc = ax[0, 1].contourf(X, Y, wvfns2[:, 1].reshape(small_grid_points, small_grid_points))
ax[0, 1].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[0, 1].set_xlabel(r'XH $\rm\AA$', fontsize=16)

tcc = ax[0, 2].contourf(X, Y, wvfns2[:, 2].reshape(small_grid_points, small_grid_points))
ax[0, 2].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[0, 2].set_xlabel(r'XH $\rm\AA$', fontsize=16)

tcc = ax[1, 0].contourf(X, Y, wvfns2[:, 3].reshape(small_grid_points, small_grid_points))
ax[1, 0].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[1, 0].set_xlabel(r'XH $\rm\AA$', fontsize=16)

tcc = ax[1, 1].contourf(X, Y, wvfns2[:, 4].reshape(small_grid_points, small_grid_points))
ax[1, 1].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[1, 1].set_xlabel(r'XH $\rm\AA$', fontsize=16)

tcc = ax[1, 2].contourf(X, Y, wvfns2[:, 5].reshape(small_grid_points, small_grid_points))
ax[1, 2].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[1, 2].set_xlabel(r'XH $\rm\AA$', fontsize=16)

tcc = ax[2, 0].contourf(X, Y, wvfns2[:, 6].reshape(small_grid_points, small_grid_points))
ax[2, 0].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[2, 0].set_xlabel(r'XH $\rm\AA$', fontsize=16)

tcc = ax[2, 1].contourf(X, Y, wvfns2[:, 7].reshape(small_grid_points, small_grid_points))
ax[2, 1].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[2, 1].set_xlabel(r'XH $\rm\AA$', fontsize=16)

tcc = ax[2, 2].contourf(X, Y, wvfns2[:, 8].reshape(small_grid_points, small_grid_points))
ax[2, 2].set_ylabel(r'R$_{\rm{OO}}$ $\rm\AA$', fontsize=16)
ax[2, 2].set_xlabel(r'XH $\rm\AA$', fontsize=16)
plt.tight_layout()
fig.colorbar(tcc)
plt.show()


