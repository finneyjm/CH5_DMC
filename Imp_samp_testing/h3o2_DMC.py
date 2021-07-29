import matplotlib.pyplot as plt
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

omega_sym = 2704.021674298211/har2wave
omega_sym_D = 1968.55510602268/har2wave

new_struct = np.array([
    [2.06095307, 0.05378083, 0.],
    [0., 0., 0.],
    [-0.32643038, -1.70972841, 0.52193868],
    [4.70153912, 0., 0.],
    [5.20071798, 0.80543847, 1.55595785]
])

small_grid_points = 200
Roo_grid = np.linspace(3.9, 5.8, small_grid_points)
sp_grid = np.linspace(-1.5, 1.5, small_grid_points)
sp_grid = np.linspace(-65, 65, small_grid_points)
sp_grid, Roo_grid = np.meshgrid(sp_grid, Roo_grid)
two_d_wvfns = np.load('small_grid_2d_h3o2_bigger_grid.npz')['wvfns']
two_d_wvfns = np.load('2d_h3o2_new_def.npz')['wvfns']
big_Roo_grid = np.linspace(4, 5.4, 1000)
big_sp_grid = np.linspace(-1.2, 1.2, 1000)
big_sp_grid = np.linspace(-50, 50, 1000)
# big_Roo_grid = np.linspace(3.9, 5.8, 100)
# big_sp_grid = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(big_sp_grid, big_Roo_grid)
# wvfn = two_d_wvfns[:, 0].reshape((len(Roo_grid), len(sp_grid)))
# wvfn = np.abs(wvfn)

interp_ground = interpolate.bisplrep(sp_grid, Roo_grid, two_d_wvfns[:, 0].reshape((len(Roo_grid), len(sp_grid))), s=1e-6)
z_ground_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground).T
# z_ground_dx1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dx=1).T
# z_ground_dx2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dx=2).T
# z_ground_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dy=1).T
# z_ground_dy2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dy=2).T
# z_ground_dx1_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground, dx=1, dy=1).T
#
# np.savez('interp_ground', a=interp_ground[0], b=interp_ground[1], c=interp_ground[2], d=interp_ground[3], e=interp_ground[4])
# blah = np.load('interp_ground.npz')
# interp_ground = [blah['a'], blah['b'], blah['c'], blah['d'], blah['e']]
# z_ground_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_ground).T
# np.save('z_ground_no_der', z_ground_no_der)
# ind = np.argwhere(z_ground_no_der < 1e-5)
z_ground_no_der[z_ground_no_der < 1e-6] = 0
np.save('z_ground_no_der_new_def', z_ground_no_der)
# np.load('z_ground_no_der.npy')

# z_ground_dx1 = z_ground_dx1/z_ground_no_der
# z_ground_dx2 = z_ground_dx2/z_ground_no_der
# z_ground_dy1 = z_ground_dy1/z_ground_no_der
# z_ground_dy2 = z_ground_dy2/z_ground_no_der
# z_ground_dx1_dy1 = z_ground_dx1_dy1/z_ground_no_der
#
# np.save('z_ground_dx1', z_ground_dx1)
# np.save('z_ground_dx2', z_ground_dx2)
# np.save('z_ground_dy1', z_ground_dy1)
# np.save('z_ground_dy2', z_ground_dy2)
# np.save('z_ground_dx1_dy1', z_ground_dx1_dy1)

# z_ground_no_der = np.load('z_ground_no_der.npy')
# z_ground_dx1 = np.load('z_ground_dx1.npy')
# z_ground_dx2 = np.load('z_ground_dx2.npy')
# z_ground_dy1 = np.load('z_ground_dy1.npy')
# z_ground_dy2 = np.load('z_ground_dy2.npy')
# z_ground_dx1_dy1 = np.load('z_ground_dx1_dy1.npy')
#
# ind = np.argwhere(z_ground_no_der < 5e-5)
# z_ground_dx1[ind[:, 0], ind[:, 1]] = 0
# z_ground_dx2[ind[:, 0], ind[:, 1]] = 0
# z_ground_dx1_dy1[ind[:, 0], ind[:, 1]] = 0
# z_ground_dy1[ind[:, 0], ind[:, 1]] = 0
# z_ground_dy2[ind[:, 0], ind[:, 1]] = 0

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, z_ground_dx1_dy1)
# fig.colorbar(tcc)
# plt.xlabel('XH')
# plt.ylabel('Roo')
# plt.show()

# ground_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_no_der.T.flatten())
# ground_dx1 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_dx1.T.flatten())
# ground_dx2 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_dx2.T.flatten())
# ground_dy1 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_dy1.T.flatten())
# ground_dy2 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_dy2.T.flatten())
# ground_dx1_dy1 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_dx1_dy1.T.flatten())

wvfn = two_d_wvfns[:, 2].reshape((len(Roo_grid), len(sp_grid)))
wvfn[:, int(small_grid_points/2):] = np.abs(wvfn[:, int(small_grid_points/2):])
wvfn[:, :int(small_grid_points/2)] = -np.abs(wvfn[:, :int(small_grid_points/2)])


interp_excite_xh = interpolate.bisplrep(sp_grid, Roo_grid, wvfn, s=1e-6)
z_excite_xh_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh).T
# ind = np.argwhere(z_excite_xh_no_der < 1e-6)
z_excite_xh_no_der[np.abs(z_excite_xh_no_der) < 1e-6] = 0
np.save('z_excite_xh_no_der_new_def', z_excite_xh_no_der)
# np.save('z_excite_xh_no_der', z_excite_xh_no_der)


z_excite_xh_dx1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dx=1).T
z_excite_xh_dx2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dx=2).T
z_excite_xh_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dy=1).T
z_excite_xh_dy2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dy=2).T
z_excite_xh_dx1_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_xh, dx=1, dy=1).T
#
z_excite_xh_dx1 = z_excite_xh_dx1/z_excite_xh_no_der
z_excite_xh_dx2 = z_excite_xh_dx2/z_excite_xh_no_der
z_excite_xh_dy1 = z_excite_xh_dy1/z_excite_xh_no_der
z_excite_xh_dy2 = z_excite_xh_dy2/z_excite_xh_no_der
z_excite_xh_dx1_dy1 = z_excite_xh_dx1_dy1/z_excite_xh_no_der

# np.save('z_excite_xh_dx1', z_excite_xh_dx1)
# np.save('z_excite_xh_dx2', z_excite_xh_dx2)
# np.save('z_excite_xh_dy1', z_excite_xh_dy1)
# np.save('z_excite_xh_dy2', z_excite_xh_dy2)
# np.save('z_excite_xh_dx1_dy1', z_excite_xh_dx1_dy1)

raise Exception(wvfn)
#
# #
# excite_xh_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_excite_xh_no_der.T.flatten())
# excite_xh_dx1 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_excite_xh_dx1.T.flatten())
# excite_xh_dx2 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_excite_xh_dx2.T.flatten())
# excite_xh_dy1 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_excite_xh_dy1.T.flatten())
# excite_xh_dy2 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_excite_xh_dy2.T.flatten())
# excite_xh_dx1_dy1 = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_excite_xh_dx1_dy1.T.flatten())

# print(wvfn[56, 56])
# print(wvfn[56, 42])
# print(wvfn[42, 56])
# print(z_excite_xh_no_der[56, 56])
# print(z_excite_xh_no_der[56, 42])
# print(z_excite_xh_no_der[42, 56])
# sps = np.array([big_sp_grid[56], big_sp_grid[56], big_sp_grid[42]])
# roos = np.array([big_Roo_grid[56], big_Roo_grid[42], big_Roo_grid[56]])
# # # print(interpolate.griddata((big_sp_grid, big_Roo_grid), z_ground_no_der, (sps, roos), method='cubic'))
# print(excite_xh_no_der(sps, roos))
# print(ground_no_der(sps, roos)[np.argsort(np.argsort(roos)), np.argsort(np.argsort(sps))])

# Roo_grid1 = np.linspace(4.4, 5., 100)
# sp_grid1 = np.linspace(-0.5, 0.5, 100)
# fig, ax = plt.subplots()
# tcc = ax.contourf(Roo_grid, sp_grid, excite_xh_no_der(Roo_grid1, sp_grid1))
# fig.colorbar(tcc)
# plt.show()
#
# interp_excite_roo = interpolate.bisplrep(sp_grid, Roo_grid, two_d_wvfns[:, 2].reshape((len(Roo_grid), len(sp_grid))), s=0)
# z_excite_roo_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo)
# z_excite_roo_dx1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dx=1)
# z_excite_roo_dx2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dx=2)
# z_excite_roo_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dy=1)
# z_excite_roo_dy2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dy=2)
# z_excite_roo_dx1_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_roo, dx=1, dy=1)
#
# excite_roo_no_der = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_no_der)
# excite_roo_dx1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dx1)
# excite_roo_dx2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dx2)
# excite_roo_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dy1)
# excite_roo_dy2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dy2)
# excite_roo_dx1_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_roo_dx1_dy1)
#
# interp_excite_both = interpolate.bisplrep(sp_grid, Roo_grid, two_d_wvfns[:, 3].reshape((len(Roo_grid), len(sp_grid))), s=0)
# z_excite_both_no_der = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both)
# z_excite_both_dx1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dx=1)
# z_excite_both_dx2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dx=2)
# z_excite_both_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dy=1)
# z_excite_both_dy2 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dy=2)
# z_excite_both_dx1_dy1 = interpolate.bisplev(big_sp_grid, big_Roo_grid, interp_excite_both, dx=1, dy=1)
#
# excite_both_no_der = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_no_der)
# excite_both_dx1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dx1)
# excite_both_dx2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dx2)
# excite_both_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dy1)
# excite_both_dy2 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dy2)
# excite_both_dx1_dy1 = interpolate.interp2d(big_sp_grid, big_Roo_grid, z_excite_both_dx1_dy1)


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


def psi_t_alt2(coords, excite, shift):
    psi = np.zeros((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    dists[:, 0] = dists[:, 0] - shift[0]
    if excite == 'all':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_both_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & roo':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_both_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'roo & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_roo_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'roo':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_roo_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    return psi


def psi_t(coords, excite, shift, sp=None, roo=None):
    psi = np.zeros((len(coords), 2))
    dists = all_dists(coords)
    if sp is not None:
        dists[:, -1] = sp
    if roo is not None:
        dists[:, -2] = roo
    mw_h = m_OH * omega_asym
    dists[:, 0] = dists[:, 0] - shift[0]
    if excite == 'all':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_both_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & roo':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_both_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'roo & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_roo_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'roo':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_roo_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    return psi


def psi_t_alt(coords, excite, shift):
    psi = np.zeros((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    mw_H = m_OH * omega_sym
    dists[:, 0] = dists[:, 0] - shift[0]
    dists[:, -3] = dists[:, -3] - shift[1]
    if excite == 'both':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = (mw_H / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_H * dists[:, -3] ** 2)) * \
                    (2 * mw_H) ** (1 / 2) * dists[:, -3]
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = (mw_H / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_H * dists[:, -3] ** 2))
    elif excite == 's':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = (mw_H / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_H * dists[:, -3] ** 2)) * \
                    (2 * mw_H) ** (1 / 2) * dists[:, -3]
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = (mw_H / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_H * dists[:, -3] ** 2))
    return psi
    # return np.ones((len(coords), 2))


def dpsidx_alt2(coords, excite, shift):
    dists = all_dists(coords)
    dists[:, 0] = dists[:, 0] - shift[0]
    daroox = daroodx(coords, dists[:, [1, 2, -2]])
    dspx = dspdx(coords)
    dr = np.concatenate((daroox, dspx[..., None]), axis=-1)
    collect = dpsidasp(coords, excite, dists)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def dpsidx(coords, excite, shift):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])
    dspx = dspdx(coords)
    dr = np.concatenate((droox, dspx[..., None]), axis=-1)
    collect = dpsidasp(coords, excite, dists)
    test = np.matmul(dr, collect[:, None, :, None]).squeeze()
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def dpsidx_alt3(coords, excite, shift):
    chain = np.zeros((len(coords), 5, 3, 4))
    dx = 1e-3  #Bohr
    coeffs = np.array([1/12, -2/3, 2/3, -1/12])/dx
    atoms = [0, 1, 2, 3, 4]  # the only atoms that affect the derivative of sp
    for atom in atoms:
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 2] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 3] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)/np.broadcast_to(psi_t(coords, excite, shift)[..., None], (len(coords), 5, 3))


def d2psidx2_alt3(coords, excite, shift):
    chain = np.zeros((len(coords), 5, 3, 5))
    psi = psi_t(coords, excite, shift)
    dx = 1e-3  #Bohr
    chain[:, :, :, 2] = np.broadcast_to(psi[..., None], (len(coords), 5, 3))
    coeffs = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]) / (dx ** 2)
    atoms = [0, 1, 2, 3, 4]  # the only atoms that affect the derivative of sp
    for atom in atoms:
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 3] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 4] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)/np.broadcast_to(psi_t(coords, excite, shift)[..., None], (len(coords), 5, 3))


def dpsidx_alt(coords, excite, shift):
    dists = all_dists(coords)
    dr = dasdx(coords, dists[:, [1, 2]])
    collect = dpsidas(coords, excite, dists[:, [0, -3]]-shift)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def d2psidx2_alt2(coords, excite, shift):
    dists = all_dists(coords)
    dists[:, 0] = dists[:, 0] - shift[0]
    daroox = daroodx(coords, dists[:, [1, 2, -2]])
    dspx = dspdx(coords)
    dr1 = np.concatenate((daroox, dspx[..., None]), axis=-1)
    daroox2 = daroodx2(coords, dists[:, [1, 2, -2]])
    dspx2 = d2spdx2(coords, dists[:, -1])
    dr2 = np.concatenate((daroox2, dspx2[..., None]), axis=-1)
    first_dir = dpsidasp(coords, excite, dists)
    second_dir = d2psidasp(coords, excite, dists)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1**2, second_dir[:, None, 0:3, None]).squeeze()
    part3 = dr1[..., 1]*dr1[..., 2]*np.broadcast_to(second_dir[:, -1, None, None], (len(dr1), 5, 3)).squeeze()
    part4 = np.matmul(np.broadcast_to(dr1[..., 0, None], daroox.shape)*dr1[..., [1, 2]],
                      (np.broadcast_to(first_dir[:, 0, None],
                       first_dir[:, [1, 2]].shape)*first_dir[:, [1, 2]])[:, None, :, None]).squeeze()
    return part1 + part2 + 2*part3 + 2*part4


def d2psidx2(coords, excite, shift):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])
    dspx = dspdx(coords)
    dr1 = np.concatenate((droox, dspx[..., None]), axis=-1)
    droox2 = daroodx2(coords, dists[:, [1, 2, -2]])
    dspx2 = d2spdx2(coords, dists[:, -1])
    dr2 = np.concatenate((droox2, dspx2[..., None]), axis=-1)
    first_dir = dpsidasp(coords, excite, dists)
    second_dir = d2psidasp(coords, excite, dists)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1 ** 2, second_dir[:, None, 0:3, None]).squeeze()
    part3 = dr1[..., 0] * dr1[..., 1] * np.broadcast_to(second_dir[:, -1, None, None], (len(dr1), 5, 3)).squeeze()
    part4 = np.matmul(np.broadcast_to(dr1[..., 0, None], droox.shape)*dr1[..., [1, 2]],
                      (np.broadcast_to(first_dir[:, 0, None],
                       first_dir[:, [1, 2]].shape)*first_dir[:, [1, 2]])[:, None, :, None]).squeeze()
    return part1 + part2 + 2*part3 + 2*part4


def parts(coords, excite):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])[..., 1]
    dspx = dspdx(coords)
    dr1 = np.concatenate((droox[..., None], dspx[..., None]), axis=-1)
    droox2 = daroodx2(coords, dists[:, [1, 2, -2]])[..., 1]
    dspx2 = d2spdx2(coords, dists[:, -1])
    dr2 = np.concatenate((droox2[..., None], dspx2[..., None]), axis=-1)
    first_dir = dpsidasp(coords, excite, dists)[:, 1:]
    second_dir = d2psidasp(coords, excite, dists)[:, 1:]
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1 ** 2, second_dir[:, None, 0:2, None]).squeeze()
    part3 = dr1[..., 0] * dr1[..., 1] * np.broadcast_to(second_dir[:, -1, None, None], (len(dr1), 5, 3)).squeeze()
    return part1, part2, part3


def d2psidx2_alt(coords, excite, shift):
    dists = all_dists(coords)
    dr1 = dasdx(coords, dists[:, [1, 2]])
    dr2 = d2asdx2(coords, dists[:, [1, 2]])
    first_dir = dpsidas(coords, excite, dists[:, [0, -3]]-shift)
    second_dir = d2psidas(coords, excite, dists[:, [0, -3]]-shift)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1**2, second_dir[:, None, 0:3, None]).squeeze()
    part3 = np.matmul(dr1*dr1[..., [1, 0]], first_dir[:, None, :, None]*first_dir[:, None, [1, 0], None]).squeeze()
    return part1 + part2 + 2*part3


def dpsidasp(coords, excite, dists):
    collect = np.zeros((len(coords), 3))
    mw_h = m_OH * omega_asym
    if excite == 'all':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = excite_both_dx1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 1] = excite_both_dy1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & roo':
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 2] = excite_both_dx1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 1] = excite_both_dy1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = excite_xh_dx1(dists[:, -1], dists[:, -2])
        collect[:, 1] = excite_xh_dy1(dists[:, -1], dists[:, -2])
    elif excite == 'sp':
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 2] = excite_xh_dx1(dists[:, -1], dists[:, -2])
        collect[:, 1] = excite_xh_dy1(dists[:, -1], dists[:, -2])
    elif excite == 'roo & a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = excite_roo_dx1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 1] = excite_roo_dy1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'roo':
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 2] = excite_roo_dx1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 1] = excite_roo_dy1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = ground_dx1(dists[:, -1], dists[:, -2])
        collect[:, 1] = ground_dy1(dists[:, -1], dists[:, -2])
    else:
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 2] = ground_dx1(dists[:, -1], dists[:, -2])
        # collect[:, 2] = ground_dx1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]\
        #                 /ground_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = ground_dy1(dists[:, -1], dists[:, -2])
        # collect[:, 1] = ground_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]\
        #                 /ground_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    return collect


def dpsidas(coords, excite, dists):
    collect = np.zeros((len(coords), 2))
    mw_h = m_OH * omega_asym
    mw_H = m_OH * omega_sym
    if excite == 'both':
        collect[:, 0] = (1 - mw_h * dists[:, 0] ** 2) / dists[:, 0]
        collect[:, 1] = (1 - mw_H * dists[:, 1] ** 2) / dists[:, 1]
    elif excite == 'a':
        collect[:, 0] = (1 - mw_h * dists[:, 0] ** 2) / dists[:, 0]
        collect[:, 1] = -mw_H*dists[:, 1]
    elif excite == 's':
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 1] = (1 - mw_H * dists[:, 1] ** 2) / dists[:, 1]
    else:
        collect[:, 0] = -mw_h * dists[:, 0]
        collect[:, 1] = -mw_H * dists[:, 1]
    return collect
    # return np.zeros((len(coords), 2))


def d2psidasp(coords, excite, dists):
    collect = np.zeros((len(coords), 4))
    mw_h = m_OH * omega_asym
    if excite == 'all':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 2] = excite_both_dx2(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 1] = excite_both_dy2(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 3] = excite_both_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & roo':
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 2] = excite_both_dx2(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 1] = excite_both_dy2(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 3] = excite_both_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 2] = excite_xh_dx2(dists[:, -1], dists[:, -2])
        collect[:, 1] = excite_xh_dy2(dists[:, -1], dists[:, -2])
        collect[:, 3] = excite_xh_dx1_dy1(dists[:, -1], dists[:, -2])
    elif excite == 'sp':
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 2] = excite_xh_dx2(dists[:, -1], dists[:, -2])
        collect[:, 1] = excite_xh_dy2(dists[:, -1], dists[:, -2])
        collect[:, 3] = excite_xh_dx1_dy1(dists[:, -1], dists[:, -2])
    elif excite == 'roo & a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 2] = excite_roo_dx2(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 1] = excite_roo_dy2(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 3] = excite_roo_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'roo':
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 2] = excite_roo_dx2(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 1] = excite_roo_dy2(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
        collect[:, 3] = excite_roo_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 2] = ground_dx2(dists[:, -1], dists[:, -2])
        collect[:, 1] = ground_dy2(dists[:, -1], dists[:, -2])
        collect[:, 3] = ground_dx1_dy1(dists[:, -1], dists[:, -2])
    else:
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 2] = ground_dx2(dists[:, -1], dists[:, -2])
        # collect[:, 2] = ground_dx2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]\
        #                 / ground_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 1] = ground_dy2(dists[:, -1], dists[:, -2])
        # collect[:, 1] = ground_dy2(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]\
        #                 /ground_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
        collect[:, 3] = ground_dx1_dy1(dists[:, -1], dists[:, -2])
        # collect[:, 3] = ground_dx1_dy1(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]\
        #                 /ground_no_der(dists[:, -1], dists[:, -2])[np.argsort(dists[:, -1]), np.argsort(dists[:, -2])]
    return collect


def d2psidas(coords, excite, dists):
    collect = np.zeros((len(coords), 2))
    mw_h = m_OH * omega_asym
    mw_H = m_OH * omega_sym
    if excite == 'both':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 1] = mw_H*(mw_H*dists[:, 1]**2 - 3)
    elif excite == 'a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 1] = mw_H**2*dists[:, 1]**2 - mw_H
    elif excite == 's':
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 1] = mw_H*(mw_H*dists[:, 1]**2 - 3)
    else:
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 1] = mw_H**2*dists[:, 1]**2 - mw_H
    return collect
    # return np.zeros((len(coords), 2))


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
    chain[:, [2, 4]] = np.zeros((len(coords), 2, 3, 5))
    return np.dot(chain, coeffs)


def daroodx(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = (-1)**bond*1/np.sqrt(2)*(
                (coords[:, 2*bond + 1] - coords[:, 2*bond + 2]) / dists[:, bond, None])
        chain[:, 2*bond + 2, :, 0] = (-1)**bond*1/np.sqrt(2)*(
                (coords[:, 2*bond + 2] - coords[:, 2*bond + 1]) / dists[:, bond, None])
    chain[:, 1, :, 1] = ((coords[:, 1] - coords[:, 3]) / dists[:, -1, None])
    chain[:, 3, :, 1] = ((coords[:, 3] - coords[:, 1]) / dists[:, -1, None])
    return chain


def dasdx(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    # for bond in range(2):
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = (-1) ** bond * 1 / np.sqrt(2) * (
                    (coords[:, 2 * bond + 1] - coords[:, 2 * bond + 2]) / dists[:, bond, None])
        chain[:, 2*bond + 2, :, 0] = (-1) ** bond * 1 / np.sqrt(2) * (
                    (coords[:, 2 * bond + 2] - coords[:, 2 * bond + 1]) / dists[:, bond, None])
    for bond in range(2):
        chain[:, 2*bond + 1, :, 1] = 1 / np.sqrt(2) * (
                (coords[:, 2 * bond + 1] - coords[:, 2 * bond + 2]) / dists[:, bond, None])
        chain[:, 2*bond + 2, :, 1] = 1 / np.sqrt(2) * (
                (coords[:, 2 * bond + 2] - coords[:, 2 * bond + 1]) / dists[:, bond, None])
    return chain


def daroodx2(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = (-1)**bond*1/np.sqrt(2)*(1./dists[:, bond, None] - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2/dists[:, bond, None]**3)
        chain[:, 2*bond + 2, :, 0] = (-1)**bond*1/np.sqrt(2)*(1./dists[:, bond, None] - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2/dists[:, bond, None]**3)
    chain[:, 1, :, 1] = (1./dists[:, -1, None] - (coords[:, 1]-coords[:, 3])**2/dists[:, -1, None]**3)
    chain[:, 3, :, 1] = (1./dists[:, -1, None] - (coords[:, 1]-coords[:, 3])**2/dists[:, -1, None]**3)
    return chain


def d2asdx2(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2 * bond + 1, :, 0] = (-1) ** bond * 1 / np.sqrt(2) * (
                    1. / dists[:, bond, None] - (coords[:, 2 * bond + 1] - coords[:, 2 * bond + 2]) ** 2 / dists[:,
                                                                                                           bond,
                                                                                                           None] ** 3)
        chain[:, 2 * bond + 2, :, 0] = (-1) ** bond * 1 / np.sqrt(2) * (
                    1. / dists[:, bond, None] - (coords[:, 2 * bond + 1] - coords[:, 2 * bond + 2]) ** 2 / dists[:,
                                                                                                           bond,
                                                                                                           None] ** 3)
    for bond in range(2):
        chain[:, 2 * bond + 1, :, 1] = 1 / np.sqrt(2) * (
                    1. / dists[:, bond, None] - (coords[:, 2 * bond + 1] - coords[:, 2 * bond + 2]) ** 2 / dists[:,
                                                                                                           bond,
                                                                                                           None] ** 3)
        chain[:, 2 * bond + 2, :, 1] = 1 / np.sqrt(2) * (
                    1. / dists[:, bond, None] - (coords[:, 2 * bond + 1] - coords[:, 2 * bond + 2]) ** 2 / dists[:,
                                                                                                           bond,
                                                                                                           None] ** 3)
    return chain


def drift(coords, excite, shift):
    return 2*dpsidx(coords, excite, shift)


def metropolis(Fqx, Fqy, x, y, excite, shift):
    psi_1 = np.prod(parralel_psi(x, excite, shift), axis=1)
    psi_2 = np.prod(parralel_psi(y, excite, shift), axis=1)
    psi_ratio = (psi_2/psi_1)**2
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


def run(N_0, time_steps, propagation, equilibration, wait_time, excite, initial_struct, initial_shifts, shift_rate):
    DW = False
    psi = Walkers(N_0, initial_struct, excite, initial_shifts)
    Fqx = parrelel_drift(psi.coords, psi.excite, psi.shift)
    num_o_collections = int((time_steps - equilibration) / (propagation + wait_time)) + 1
    time = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    accept = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))

    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Eref_array = np.zeros(time_steps)

    shift = np.zeros((time_steps + 1, len(psi.shift)))
    shift[0] = psi.shift
    shift_rate = np.array(shift_rate)
    psi.shift = np.array(psi.shift)
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)

        if DW is False:
            prop = float(propagation)
            wait -= 1.
        else:
            prop -= 1.

        if i == 0:
            psi = pot(psi)
            psi = E_loc(psi)
            Eref = E_ref_calc(psi)

        if i == 5000:
            # lets_debug = True
            raise Exception(i)

        psi, Fqx, acceptance = Kinetic(psi, Fqx)
        shift[i + 1] = psi.shift
        psi = pot(psi)
        psi = E_loc(psi)

        psi = Weighting(Eref, psi, DW, Fqx)
        Eref = E_ref_calc(psi)

        Eref_array[i] = Eref
        time[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)
        accept[i] = acceptance

        if i >= 5000:
            psi.shift = psi.shift + shift_rate

        if i >= int(equilibration) - 1 and wait <= 0. < prop:
            DW = True
            wait = float(wait_time)
            Psi_tau = copy.deepcopy(psi)
            coords[num] = Psi_tau.coords
            weights[num] = Psi_tau.weights
        elif prop == 0:
            DW = False
            des[num] = descendants(psi)
            num += 1

    return coords, weights, time, Eref_array, sum_weights, accept, des


pool = mp.Pool(mp.cpu_count()-1)

# ref = np.array([
#   [0.000000000000000, 0.000000000000000, 0.000000000000000],
#   [-2.304566686034061, 0.000000000000000, 0.000000000000000],
#   [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
#   [2.304566686034061, 0.000000000000000, 0.000000000000000],
#   [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233],
# ])
#
# xh = np.linspace(-0.25, 0.25, 100)
# psi = Walkers(100, ref, None, [0, 2.5721982410729867])
# psi.coords[:, 0, 0] = xh
# psi = pot(psi)
# psi = E_loc(psi)
# plt.plot(xh, psi.El*har2wave)
# plt.plot(xh, psi.V*har2wave)
# plt.plot(xh, (psi.El-psi.V)*har2wave)
# plt.show()

test_structure = np.array([
        [ 2.75704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.07502706, -1.66894299, -0.69579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.84391277,  0.14767547,  1.4669121 ],
])

test_structure2 = np.array([
        [ 2.55704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.09502706, -1.86894299, -0.69579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.24391277,  0.14767547,  1.4669121 ],
])
trials = [4, 6, 8, 9]
for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        5000, 20000, 250, 500, 500, None, test_structure, [0, 2.5721982410729867], [0, 0]
    )
    np.savez(f'ground_state_full_h3o2_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)
#
for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        5000, 20000, 250, 500, 500, 'sp', test_structure, [0, 2.5721982410729867], [0, 0]
    )
    np.savez(f'XH_excite_state_full_h3o2_left_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)

for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        5000, 20000, 250, 500, 500, None, test_structure2, [0, 2.5721982410729867], [0, 0]
    )
    np.savez(f'ground_state_full_h3o2_{i+6}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)
#
for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        5000, 20000, 250, 500, 500, 'sp', test_structure2, [0, 2.5721982410729867], [0, 0]
    )
    np.savez(f'XH_excite_state_full_h3o2_right_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)

for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        5000, 20000, 250, 500, 500, 'a', test_structure, [0, 2.5721982410729867], [0, 0]
    )
    np.savez(f'Asym_excite_state_full_h3o2_left_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)

for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        5000, 20000, 250, 500, 500, 'a', test_structure2, [0, 2.5721982410729867], [0, 0]
    )
    np.savez(f'Asym_excite_state_full_h3o2_right_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)
