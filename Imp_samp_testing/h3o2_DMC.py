import copy
from scipy import interpolate
import numpy as np
import multiprocessing as mp
from ProtWaterPES import *

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_OD = (m_D*m_O)/(m_D+m_O)
m_OH = (m_H*m_O)/(m_H+m_O)
dtau = 1
alpha = 1./(2.*dtau)
sigmaH = np.sqrt(dtau/m_H)
sigmaO = np.sqrt(dtau/m_O)
sigmaD = np.sqrt(dtau/m_D)
sigma = np.broadcast_to(np.array([sigmaH, sigmaO, sigmaH, sigmaO, sigmaH])[:, None], (5, 3))
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

omega_asym = 3070.648654929466/har2wave
omega_asym_D = 2235.4632530938925/har2wave

struct = np.array([
    [2.06095307, 0.05378083, 0.],
    [0., 0., 0.],
    [-0.32643038, -1.70972841, 0.52193868],
    [4.70153912, 0., 0.],
    [5.20071798, 0.80543847, 1.55595785]
])

Roo_grid = np.linspace(4.2, 5.2, 50)
sp_grid = np.linspace(-1, 1, 50)
sp_grid, Roo_grid = np.meshgrid(sp_grid, Roo_grid)
two_d_wvfns = np.load('small_grid_2d_h3o2.npz')['wvfns']
big_Roo_grid = np.linspace(4.4, 5.0, 2000)
big_sp_grid = np.linspace(-0.75, 0.75, 2000)

interp_ground = interpolate.bisplrep(sp_grid, Roo_grid, two_d_wvfns[:, 0].reshape((len(Roo_grid), len(sp_grid))), s=0)
z_ground_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground)
z_ground_dx1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dx=1)
z_ground_dx2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dx=2)
z_ground_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dy=1)
z_ground_dy2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dy=2)
z_ground_dx1_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dx=1, dy=1)

ground_no_der = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_ground_no_der)
ground_dx1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_ground_dx1)
ground_dx2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_ground_dx2)
ground_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_ground_dy1)
ground_dy2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_ground_dy2)
ground_dx1_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_ground_dx1_dy1)

interp_excite_xh = interpolate.bisplrep(sp_grid, Roo_grid, two_d_wvfns[:, 1].reshape((len(Roo_grid), len(sp_grid))), s=0)
z_excite_xh_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh)
z_excite_xh_dx1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dx=1)
z_excite_xh_dx2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dx=2)
z_excite_xh_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dy=1)
z_excite_xh_dy2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dy=2)
z_excite_xh_dx1_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dx=1, dy=1)

excite_xh_no_der = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_xh_no_der)
excite_xh_dx1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_xh_dx1)
excite_xh_dx2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_xh_dx2)
excite_xh_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_xh_dy1)
excite_xh_dy2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_xh_dy2)
excite_xh_dx1_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_xh_dx1_dy1)

interp_excite_roo = interpolate.bisplrep(sp_grid, Roo_grid, two_d_wvfns[:, 2].reshape((len(Roo_grid), len(sp_grid))), s=0)
z_excite_roo_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo)
z_excite_roo_dx1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dx=1)
z_excite_roo_dx2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dx=2)
z_excite_roo_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dy=1)
z_excite_roo_dy2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dy=2)
z_excite_roo_dx1_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dx=1, dy=1)

excite_roo_no_der = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_no_der)
excite_roo_dx1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dx1)
excite_roo_dx2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dx2)
excite_roo_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dy1)
excite_roo_dy2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dy2)
excite_roo_dx1_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dx1_dy1)

interp_excite_both = interpolate.bisplrep(sp_grid, Roo_grid, two_d_wvfns[:, 3].reshape((len(Roo_grid), len(sp_grid))), s=0)
z_excite_both_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both)
z_excite_both_dx1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dx=1)
z_excite_both_dx2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dx=2)
z_excite_both_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dy=1)
z_excite_both_dy2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dy=2)
z_excite_both_dx1_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dx=1, dy=1)

excite_both_no_der = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_no_der)
excite_both_dx1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dx1)
excite_both_dx2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dx2)
excite_both_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dy1)
excite_both_dy2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dy2)
excite_both_dx1_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dx1_dy1)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct, excite, initial_shifts):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([initial_struct]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.excite = excite
        self.shift = initial_shifts


def psi_t(coords, excite, shift):
    psi = np.zeros((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    dists[:, 0] = dists[:, 0] - shift
    if excite == 'all':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_both_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp & roo':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_both_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'roo & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_roo_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'roo':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_roo_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    return psi


def dpsidx(coords, excite, shift):
    dists = all_dists(coords)
    daroox = daroodx(coords, dists[:, 1:4])
    dspx = dspdx(coords)
    dr = np.concatenate((daroox, dspx[..., None]), axis=-1)
    collect = dpsidasp(coords, excite, dists, shift)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def d2psidx2(coords, excite, shift):
    dists = all_dists(coords)
    daroox = daroodx(coords, dists[:, 1:4])
    dspx = dspdx(coords)
    dr1 = np.concatenate((daroox, dspx[..., None]), axis=-1)
    daroox2 = daroodx2(coords, dists[:, 1:4])
    dspx2 = d2spdx2(coords, dists[:, -1])
    dr2 = np.concatenate((daroox2, dspx2[..., None]), axis=-1)
    first_dir = dpsidasp(coords, excite, dists, shift)
    second_dir = d2psidasp(coords, excite, dists, shift)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1**2, second_dir[:, None, 0:3, None]).squeeze()
    part3 = dr1[..., 1]*dr1[..., 2]*np.broadcast_to(second_dir[:, -1, None, None], (len(dr1), 5, 3))
    part4 = np.matmul(dr1[..., 0:2]*np.broadcast_to(dr1[..., 2, None], (len(dr1), 5, 3, 2)),
              np.broadcast_to(second_dir[:, -1, None], (len(dr1), 2))[:, None, :, None]).squeeze()
    return part1 + part2 + part3 + 2*part4


def dpsidasp(coords, excite, dists, shift):
    collect = np.zeros((len(coords), 3))
    mw_h = m_OH * omega_asym
    dists[:, 0] = dists[:, 0] - shift
    if excite == 'all':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = excite_both_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_both_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp & roo':
        collect[:, 0] = -mw_h*dists[:, 1]
        collect[:, 2] = excite_both_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_both_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp & a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = excite_xh_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_xh_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp':
        collect[:, 0] = -mw_h*dists[:, 1]
        collect[:, 2] = excite_xh_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_xh_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'roo & a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = excite_roo_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_roo_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'roo':
        collect[:, 0] = -mw_h*dists[:, 1]
        collect[:, 2] = excite_roo_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_roo_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = ground_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = ground_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    else:
        collect[:, 0] = -mw_h*dists[:, 1]
        collect[:, 2] = ground_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = ground_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    return collect


def d2psidasp(coords, excite, dists, shift):
    collect = np.zeros((len(coords), 4))
    mw_h = m_OH * omega_asym
    dists[:, 0] = dists[:, 0] - shift
    if excite == 'all':
        collect[:, 0] = mw_h*(mw_h*dists[:, 1]**2 - 3)
        collect[:, 2] = excite_both_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_both_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = excite_both_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp & roo':
        collect[:, 0] = mw_h**2*dists[:, 1]**2 - mw_h
        collect[:, 2] = excite_both_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_both_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = excite_both_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp & a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 1]**2 - 3)
        collect[:, 2] = excite_xh_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_xh_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = excite_xh_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'sp':
        collect[:, 0] = mw_h**2*dists[:, 1]**2 - mw_h
        collect[:, 2] = excite_xh_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_xh_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = excite_xh_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'roo & a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 1]**2 - 3)
        collect[:, 2] = excite_roo_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_roo_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = excite_roo_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'roo':
        collect[:, 0] = mw_h**2*dists[:, 1]**2 - mw_h
        collect[:, 2] = excite_roo_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = excite_roo_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = excite_roo_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    elif excite == 'a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 1]**2 - 3)
        collect[:, 2] = ground_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = ground_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = ground_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    else:
        collect[:, 0] = mw_h**2*dists[:, 1]**2 - mw_h
        collect[:, 2] = ground_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = ground_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = ground_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    return collect


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    mid = dis[:, 2]/2
    sp = mid - dis[:, -1]*np.cos(roh_roo_angle(coords, dis[:, -2], dis[:, -1]))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], dis[:, -2], sp)).T


def sp_calc_for_fd(coords):
    bonds = [[1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    mid = dis[:, 0] / 2
    sp = mid - dis[:, -1] * np.cos(roh_roo_angle(coords, dis[:, -2], dis[:, -1]))
    return sp


def roh_roo_angle(coords, roo_dist, roh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (coords[:, 1]-coords[:, 0])/np.broadcast_to(roh_dist[:, None], (len(roh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


def dspdx(coords):
    chain = np.zeros((len(coords), 5, 3, 4))
    dx = 1e-3  #Bohr
    coeffs = np.array([1/12, -2/3, 2/3, -1/12])/dx
    atoms = [0, 1, 3]  # the only atoms that affect the derivative of sp
    for atom in atoms:
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 2] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 3] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)


def d2spdx2(coords, sp):
    chain = np.zeros((len(coords), 5, 3, 5))
    chain[:, :, :, 2] = np.broadcast_to(sp[..., None, None], (len(coords), 5, 3))
    dx = 1e-3  #Bohr
    coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])/(dx**2)
    atoms = [0, 1, 3]  # the only atoms that affect the derivative of sp
    for atom in atoms:
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 3] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 4] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)


def daroodx(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = 1/np.sqrt(2)*((coords[:, 2*bond + 1] - coords[:, 2*bond + 2]) / dists[:, bond, None])
        chain[:, 2*bond + 2, :, 0] = 1/np.sqrt(2)*((coords[:, 2*bond + 2] - coords[:, 2*bond + 1]) / dists[:, bond, None])
    chain[:, 1, :, 1] = ((coords[:, 1] - coords[:, 3]) / dists[:, -1, None])
    chain[:, 3, :, 1] = ((coords[:, 3] - coords[:, 1]) / dists[:, -1, None])
    return chain


def daroodx2(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = 1/np.sqrt(2)*(1./dists[:, bond, None] - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2/dists[:, bond, None]**3)
        chain[:, 2*bond + 2, :, 0] = 1/np.sqrt(2)*(1./dists[:, bond, None] - (coords[:, 2*bond + 2]-coords[:, 2*bond + 1])**2/dists[:, bond, None]**3)
    chain[:, 1, :, 1] = (1./dists[:, -1, None] - (coords[:, 1]-coords[:, 3])**2/dists[:, -1, None]**3)
    chain[:, 3, :, 1] = (1./dists[:, -1, None] - (coords[:, 1]-coords[:, 3])**2/dists[:, -1, None]**3)
    return chain


def drift(coords, excite, shift):
    return 2*dpsidx(coords, excite, shift)


def metropolis(Fqx, Fqy, x, y, excite, shift):
    psi_1 = parralel_psi(x, excite, shift)
    psi_2 = parralel_psi(y, excite, shift)
    psi_ratio = np.prod((psi_2/psi_1)**2, axis=1)
    a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
    a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
    remove = np.argwhere(psi_2 * psi_1 < 0)
    a[remove] = 0.
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx):
    Drift = sigma**2/2.*Fqx
    randomwalk = np.random.normal(0.0, sigma, size=(len(Psi.coords), sigma.shape[0], sigma.shape[1]))
    y = randomwalk + Drift + np.array(Psi.coords)
    Fqy = parrelel_drift(y, Psi.excite, Psi.shift)
    a = metropolis(Fqx, Fqy, Psi.coords, y, Psi.excite, Psi.shift)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqx, acceptance


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def pot(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


def parralel_psi(coords, excite, shift):
    from functools import partial
    coords = np.array_split(coords, mp.cpu_count()-1)
    psi = pool.map(partial(psi_t, excite=excite, shift=shift), coords)
    full_psi = np.concatenate(psi)
    return full_psi


def parrelel_drift(coords, excite, shift):
    from functools import partial
    coords = np.array_split(coords, mp.cpu_count()-1)
    f = pool.map(partial(drift, excite=excite, shift=shift), coords)
    F = np.concatenate(f)
    return F


def parrelel_second_dir(coords, excite, shift):
    from functools import partial
    coords = np.array_split(coords, mp.cpu_count()-1)
    s_dir = pool.map(partial(d2psidx2, excite=excite, shift=shift), coords)
    S_dir = np.concatenate(s_dir)
    return S_dir


def local_kinetic(Psi):
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * parrelel_second_dir(Psi.coords, Psi.excite, Psi.shift), axis=1), axis=1)
    return kin


def E_loc(Psi):
    Psi.El = local_kinetic(Psi) + Psi.V
    return Psi


def E_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/P0)
    return E_ref


def Weighting(Eref, Psi, DW, Fqx):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    threshold = 0.01
    max_thresh = 20
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        if DW is True:
            Biggo_num = int(Psi.walkers[ind])
            Psi.walkers[i[0]] = Biggo_num
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_el = float(Psi.El[ind])
        Biggo_force = np.array(Fqx[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Fqx[i[0]] = Biggo_force

    death = np.argwhere(Psi.weights > max_thresh)
    for i in death:
        ind = np.argmin(Psi.weights)
        if DW is True:
            Biggo_num = float(Psi.walkers[i[0]])
            Psi.walkers[ind] = Biggo_num
        Biggo_weight = float(Psi.weights[i[0]])
        Biggo_pos = np.array(Psi.coords[i[0]])
        Biggo_pot = float(Psi.V[i[0]])
        Biggo_el = float(Psi.El[i[0]])
        Biggo_force = np.array(Fqx[i[0]])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[ind] = Biggo_pos
        Psi.V[ind] = Biggo_pot
        Psi.El[ind] = Biggo_el
        Fqx[ind] = Biggo_force
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


def run(N_0, time_steps, propagation, excite, initial_struct, initial_shifts, shift_rate):
    DW = False
    psi = Walkers(N_0, initial_struct, excite, initial_shifts)
    Fqx = parrelel_drift(psi.coords, psi.excite, psi.shift)
    Psi, Fqx, acceptance = Kinetic(psi, Fqx)
    Psi = pot(Psi)
    Psi = E_loc(Psi)
    time = np.array([])
    weights = np.array([])
    accept = np.array([])
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, DW, Fqx)
    time = np.append(time, 1)
    weights = np.append(weights, np.sum(new_psi.weights))
    accept =np.append(accept, acceptance)

    Psi_tau = 0
    shift = np.zeros((time_steps + 1, 3))
    shift[0] = Psi.shift
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)

        Psi, Fqx, acceptance = Kinetic(new_psi, Fqx)
        Psi = pot(Psi)
        Psi = E_loc(Psi)
        shift[i + 1] = Psi.shift

        if i >= 5000:
            Psi.shift = Psi.shift + shift_rate

        if DW is False:
            prop = float(propagation)
        elif DW is True:
            prop -= 1.
            if Psi_tau == 0:
                Psi_tau = copy.deepcopy(Psi)
        new_psi = Weighting(Eref, Psi, DW, Fqx)

        Eref = E_ref_calc(new_psi)

        Eref_array = np.append(Eref_array, Eref)
        time = np.append(time, 2 + i)
        weights = np.append(weights, np.sum(new_psi.weights))
        accept = np.append(accept, acceptance)

        if i >= (time_steps - 1. - float(propagation)) and prop > 0.:
            DW = True
        elif i >= (time_steps - 1. - float(propagation)) and prop == 0.:
            d_values = descendants(new_psi)
    return Eref_array, weights, shift, d_values, Psi_tau.coords


pool = mp.Pool(mp.cpu_count()-1)

new_struct = np.array([
    [0.000000000000000, 0.000000000000000, 0.000000000000000],
    [-2.304566686034061, 0.000000000000000, 0.000000000000000],
    [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
    [2.304566686034061, 0.000000000000000, 0.000000000000000],
    [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233]
])
new_struct[:, 0] = new_struct[:, 0] + 2.304566686034061
import matplotlib.pyplot as plt
psi = Walkers(50, new_struct, None, [0])
psi.coords[:, 0, 0] = np.linspace(1.8, 2.8, 50)
psi = pot(psi)
psi = E_loc(psi)
plt.plot(psi.coords[:, 0, 0], psi.El*har2wave, label='Local Energy')
plt.plot(psi.coords[:, 0, 0], psi.El*har2wave-psi.V*har2wave, label='Local Kinetic Energy')
plt.plot(psi.coords[:, 0, 0], psi.V*har2wave, label='Potential Energy')
plt.legend()
plt.show()


# eref, weights, shift, d, coords = run(1000, 5000, 250, None, new_struct, [0], [0])
# plt.plot(eref*har2wave)
#
# np.save('h3o2_test_ground_eref', eref)
# plt.show()
