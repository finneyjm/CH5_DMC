import numpy as np
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp
from scipy import interpolate
from ProtWaterPES import *

A = np.array([[42.200232187251913, -0.60594644269321474], [1.0206303697659393, 41.561937672470521]])
z_p = np.linspace(-45, 45, 200)
# z_p = np.zeros(200)
a_p = np.linspace(-5, 5, 200)
a_p = np.zeros(200)

eh = np.matmul(np.linalg.inv(A), np.vstack((a_p, z_p)))
grid_sp = eh[1]
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

small_grid_points = 400
Roo_grid = np.linspace(3.9, 5.8, small_grid_points)
sp_grid = np.linspace(-65, 65, small_grid_points)
X, Y = np.meshgrid(sp_grid, Roo_grid)
two_d_wvfns = np.load('../../2d_h3o2_new_def.npz')['wvfns']

# z_ground_no_der = np.load('z_ground_no_der_new_def.npy')

ground_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
                                                       np.abs(two_d_wvfns[:, 0].reshape((len(Roo_grid),
                                                                                         len(sp_grid)))).flatten())
wvfn = two_d_wvfns[:, 2].reshape((len(Roo_grid), len(sp_grid)))
wvfn[:, int(small_grid_points/2):] = np.abs(wvfn[:, int(small_grid_points/2):])
wvfn[:, :int(small_grid_points/2)] = -np.abs(wvfn[:, :int(small_grid_points/2)])
# z_excite_xh_no_der = np.load('z_excite_xh_no_der_new_def.npy')

excite_xh_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
                                                          wvfn.flatten())


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


def get_da_psi(coords, excite):
    psi = np.ones((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    # dists[:, 0] = dists[:, 0] - shift[0]
    dead = -0.60594644269321474*dists[:, -1] + 42.200232187251913*dists[:, 0]
    dead2 = 41.561937672470521*dists[:, -1] + 1.0206303697659393*dists[:, 0]
    if excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dead ** 2))
        psi[:, 1] = excite_xh_no_der(dead2, dists[:, -2])
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dead ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dead
        # psi[:, 1] = ground_no_der(dead2, dists[:, -2])
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
mesh = np.array(np.meshgrid(grid_a, grid_sp))
gridz = np.reshape(mesh, (2, len(grid_a) * len(grid_sp)))
coords = linear_combo_grid(new_struct, grid_sp, grid_a)
d, psi = drift(coords, 'sp')
# d = np.sum(np.sum(d, axis=1), axis=1)
plt.plot(grid_a, d[:, 0, 0]/ang2bohr, label='H1')
plt.plot(grid_a, d[:, 1, 0]/ang2bohr, label='O1')
plt.plot(grid_a, d[:, 2, 0]/ang2bohr, label='H2')
plt.plot(grid_a, d[:, 3, 0]/ang2bohr, label='O2')
plt.plot(grid_a, d[:, 4, 0]/ang2bohr, label='H3')
plt.xlabel(r'z')
plt.ylabel(r'drift $\AA$')
plt.legend()
plt.show()
kin = local_kinetic(psi)
v = pot(coords)

e_loc = kin+v
plt.plot(z_p, e_loc*har2wave, label='Local Energy')
plt.plot(z_p, v*har2wave, label='Potential Energy')
plt.plot(z_p, kin*har2wave, label='Local Kinetic Energy')

plt.legend()
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





