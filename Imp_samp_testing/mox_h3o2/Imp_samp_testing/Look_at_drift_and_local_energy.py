import numpy as np
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp
from scipy import interpolate
from ProtWaterPES import *

A = np.array([[42.200232187251913, -0.60594644269321474], [1.0206303697659393, 41.561937672470521]])
z_p = np.linspace(-45, 45, 200)
# z_p = np.zeros(200)
a_p = np.linspace(-25, 25, 200)
a_p = np.zeros(200)

eh = np.matmul(np.linalg.inv(A), np.vstack((a_p, z_p)))
grid_sp = eh[1]
# grid_sp = grid_sp - grid_sp
grid_a = eh[0]

m_OH = 1
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
dtau = 1
alpha = 1./(2.*dtau)
sigmaH = np.sqrt(dtau/m_H)
sigmaO = np.sqrt(dtau/m_O)
sigmaD = np.sqrt(dtau/m_D)
sigma = np.broadcast_to(np.array([sigmaH, sigmaO, sigmaH, sigmaO, sigmaH])[:, None], (5, 3))
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
omega_asym = 3815.044564/har2wave

small_grid_points = 600
Roo_grid = np.linspace(3.9, 5.8, small_grid_points)
sp_grid = np.linspace(-65, 65, small_grid_points)
# sp_grid = np.linspace(-1.5, 1.5, small_grid_points)
big_Roo_grid = np.linspace(4, 5.4, 1000)
big_sp_grid = np.linspace(-1.2, 1.2, 1000)
big_sp_grid = np.linspace(-50, 50, 1000)
X, Y = np.meshgrid(big_sp_grid, big_Roo_grid)
X2, Y2 = np.meshgrid(sp_grid, Roo_grid)
# two_d_wvfns = np.load('../../2d_h3o2_new_def_600_points.npz')['wvfns']

z_ground_no_der = np.load('z_ground_no_der_big_no_cutoff.npy')

ground_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
                                                       z_ground_no_der.flatten())
# wvfn = two_d_wvfns[:, 2].reshape((len(Roo_grid), len(sp_grid)))
# wvfn[:, int(small_grid_points/2):] = np.abs(wvfn[:, int(small_grid_points/2):])
# wvfn[:, :int(small_grid_points/2)] = -np.abs(wvfn[:, :int(small_grid_points/2)])
z_excite_xh_no_der = np.load('z_excite_xh_no_der_big_no_cutoff.npy')

excite_xh_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
                                                          z_excite_xh_no_der.flatten())


Tpsi_o_psi1 = np.load('kinetic_energy_matrix_zp_Roo_ground_no_cutoff.npy')

# wvfn = np.load('2d_h3o2_new_def_600_points_no_cutoff.npz')['wvfns'][:, 0]
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(3)
# ax[0].plot(X2[280, :], Tpsi_o_psi1.reshape((600, 600))[280, :])
# ax[1].plot(X2[280, :], (Tpsi_o_psi1*wvfn).reshape((600, 600))[280, :])
# ax[2].plot(X2[280, :], wvfn.reshape((600, 600))[280, :])
# plt.xlabel("z'")
# # plt.colorbar()
# plt.show()

ground_T = interpolate.CloughTocher2DInterpolator(list(zip(X2.flatten(), Y2.flatten())),
                                                  Tpsi_o_psi1)

Tpsi_o_psi2 = np.load('kinetic_energy_matrix_zp_Roo_excite_no_cutoff.npy')

excite_xh_T = interpolate.CloughTocher2DInterpolator(list(zip(X2.flatten(), Y2.flatten())),
                                                     Tpsi_o_psi2)


def shared_prot_grid(coords, sp):
    mid = (coords[:, 3, 0] - coords[:, 1, 0])/2
    coords[:, 0, 0] = mid+sp
    return coords


def linear_combo_grid(coords, grid1, grid2):
    re_a = np.linalg.norm(coords[2]-coords[1])
    coords = np.array([coords] * 1)
    coords = coords[:, (1, 3, 0, 2, 4)]
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0], [3, 0, 1, 2],
                                                                                   [4, 1, 0, 2]])).coords
    N = len(grid1)
    zmat = np.array([zmat] * N).reshape((N, 4, 6))
    zmat[:, 2, 1] = re_a + np.sqrt(2) / 2 * grid2
    zmat[:, 3, 1] = re_a - np.sqrt(2) / 2 * grid2
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    coords = new_coords[:, (2, 0, 3, 1, 4)]
    coords = shared_prot_grid(coords, grid1)
    coords[:, :, 1] = coords[:, :, 2]
    coords[:, :, 2] = np.zeros(coords[:, :, 2].shape)
    return coords


def a_prime(a, z):
    return -0.60594644269321474*z + 42.200232187251913*a


def z_prime(a, z):
    return 41.561937672470521*z + 1.0206303697659393*a


def get_da_psi(coords, excite):
    psi = np.ones((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    # dists[:, 0] = dists[:, 0] - shift[0]
    dead = -0.60594644269321474*dists[:, -1] + 42.200232187251913*dists[:, 0]
    dead2 = 41.561937672470521*dists[:, -1] + 1.0206303697659393*dists[:, 0]
    # dead2 = dists[:, -1]
    # dead = dists[:, 0]
    if excite == 'sp':
        # psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dead ** 2))
        psi[:, 1] = excite_xh_no_der(dead2, dists[:, -2])
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dead ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dead
        psi[:, 1] = ground_no_der(dead2, dists[:, -2])
    else:
        # psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dead ** 2))
        psi[:, 1] = ground_no_der(dead2, dists[:, -2])
    return psi


def all_da_psi(coords, excite):
    dx = 1e-3
    psi = np.zeros((len(coords), 3, 5, 3))
    psi[:, 1] = np.broadcast_to(np.prod(get_da_psi(coords, excite), axis=1)[:, None, None], (len(coords), 5, 3))
    for atom in range(5):
        for xyz in range(3):
            coords[:, atom, xyz] -= dx
            psi[:, 0, atom, xyz] = np.prod(get_da_psi(coords, excite), axis=1)
            coords[:, atom, xyz] += 2*dx
            psi[:, 2, atom, xyz] = np.prod(get_da_psi(coords, excite), axis=1)
            coords[:, atom, xyz] -= dx
    return psi


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0], [3, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    s_oh = 1/np.sqrt(2)*(dis[:, 0]+dis[:, 1])
    mid = dis[:, 2]/2
    sp = -mid + dis[:, -2]*np.cos(roh_roo_angle(coords, dis[:, -3], dis[:, -2]))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], s_oh, dis[:, -3], sp)).T


def roh_roo_angle(coords, roo_dist, roh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (coords[:, 1]-coords[:, 0])/np.broadcast_to(roh_dist[:, None], (len(roh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


def drift(coords, excite):
    dx = 1e-3
    psi = all_da_psi(coords, excite)
    der = (psi[:, 2] - psi[:, 0])/dx/psi[:, 1]
    return der, psi


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def pot(coordz):
    coords = np.array_split(coordz, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    V = np.concatenate(V)
    return V


pool = mp.Pool(mp.cpu_count()-1)


def local_kinetic(psit):
    dx = 1e-3
    d2psidx2 = ((psit[:, 0] - 2. * psit[:, 1] + psit[:, 2]) / dx ** 2) / psit[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin


import matplotlib.pyplot as plt
new_struct = np.array([
    [0.000000000000000, 0.000000000000000, 0.000000000000000],
    [-2.304566686034061, 0.000000000000001, 0.000000000000000],
    [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
    [2.304566686034061, 0.000000000000001, 0.000000000000000],
    [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233]
])
new_struct[:, 0] = new_struct[:, 0] + 2.304566686034061

print(all_dists(np.array([new_struct]*1))[:, -2])
coordz = np.array([[2.9438700000,     -0.1612600000,      0.3570000000],
       [ 0.1400000000,     -0.3178000000,     -0.0612000000],
       [-0.3573000000,     -2.1260000000,     -0.0914000000],
        [4.9771000000,      0.1289000000,     -0.2338000000],
        [6.1124000000,     -0.1762000000,      1.2233000000]])
# new_struct = coordz
mesh = np.array(np.meshgrid(grid_a, grid_sp))
gridz = np.reshape(mesh, (2, len(grid_a) * len(grid_sp)))
coords = linear_combo_grid(new_struct, grid_sp, grid_a)
d, psi = drift(coords, 'sp')
dists = all_dists(coords)
better_T = excite_xh_T(z_prime(dists[:, 0], dists[:, -1]), dists[:, -2])
# d2, psi2 = drift(coordz.reshape((1, 5, 3)), 'sp')
# dis = all_dists(coordz.reshape((1, 5, 3)))
# dis_track = all_dists(coords)
# avg_z_p_diff = np.average(z_p - (41.561937672470521*dis_track[:, -1] + 1.0206303697659393*dis_track[:, 0]))
# avg_a_p_diff = np.average(a_p - (-0.60594644269321474*dis_track[:, -1] + 42.200232187251913*dis_track[:, 0]))
# print(d2)
# print(dis)
# print(-0.60594644269321474*dis[:, -1] + 42.200232187251913*dis[:, 0])
# print(41.561937672470521*dis[:, -1] + 1.0206303697659393*dis[:, 0])
# print(avg_a_p_diff)
# print(avg_z_p_diff)
# d = np.sum(np.sum(d, axis=1), axis=1)
# zyx = ['x', 'y', 'z']
# atom = ['H1', 'O1', 'H2', 'O2', 'H3']
# fig, ax = plt.subplots(5, 3)
# for i in range(5):
#     for xyz in range(3):
#         ax[i, xyz].plot(grid_a, d[:, i, xyz]/ang2bohr, label=f'{atom[i]} {zyx[xyz]}')
#         # ax[i, xyz].plot(grid_a, d[:, i, xyz]/ang2bohr, label=f'O1 {zyx[xyz]}')
#         # ax[i, xyz].plot(grid_a, d[:, i, xyz]/ang2bohr, label=f'H2 {zyx[xyz]}')
#         # ax[i, xyz].plot(grid_a, d[:, i, xyz]/ang2bohr, label=f'O2 {zyx[xyz]}')
#         # ax[i, xyz].plot(grid_a, d[:, i, xyz]/ang2bohr, label=f'H3 {zyx[xyz]}')
#         ax[i, xyz].set_xlabel(r"a")
#         ax[i, xyz].set_ylabel(r'drift $\AA$')
#         ax[i, xyz].legend()
# # plt.legend()
# plt.show()
kin = local_kinetic(psi)
v = pot(coords)

e_loc = kin+v
print(v*har2wave)
print(e_loc*har2wave)
plt.plot(z_p, e_loc*har2wave, label='Local Energy')
plt.plot(z_p, v*har2wave, label='Potential Energy')
plt.plot(z_p, kin*har2wave, label='Local Kinetic Energy')
plt.plot(z_p, better_T*har2wave, label='Better Local Kinetic Energy')
plt.xlabel(r"z'")
plt.ylabel(r'Energy cm$^-1$')
plt.tight_layout()
plt.ylim(-40000, 40000)

plt.legend()
plt.savefig('Local_energy_ground_sp_tpsi')
plt.show()
# e_loc = e_loc.reshape((len(grid_a), len(grid_sp)))
# mesh = np.array(np.meshgrid(a_p, z_p))


# fig, ax = plt.subplots()
# tcc = plt.contourf(mesh[0], mesh[1], d/ang2bohr)
# fig.colorbar(tcc)
# plt.show()
# fig, ax = plt.subplots()
# tcc = plt.contourf(mesh[0], mesh[1], e_loc*har2wave)
# fig.colorbar(tcc)
# plt.show()





