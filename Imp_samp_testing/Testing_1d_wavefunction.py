import copy
from scipy import interpolate
import numpy as np
import multiprocessing as mp
from ProtWaterPES import *
from itertools import repeat

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
sigma = np.sqrt(dtau/((m_O*m_H)/(m_O+m_H)))
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

omega_asym = 3815.044564/har2wave
omega_asym_D = 2235.4632530938925/har2wave

omega_sym = 2704.021674298211/har2wave
omega_sym_D = 1968.55510602268/har2wave

new_struct = np.array([
    [0.000000000000000, 0.000000000000000, 0.000000000000000],
    [-2.304566686034061, 0.000000000000000, 0.000000000000000],
    [-2.740400260927908, -1.766154718409233, 1.0814221449986587E-016],
    [2.304566686034061, 0.000000000000000, 0.000000000000000],
    [2.740400260927908, 1.766154718409233, 1.0814221449986587E-016]
])

re = np.linalg.norm(new_struct[2] - new_struct[1])
re_unit1 = (new_struct[2] - new_struct[1])/re
re_unit2 = (new_struct[4] - new_struct[3])/re

Roo_grid = np.linspace(3.9, 5.8, 100)
sp_grid = np.linspace(-1.5, 1.5, 100)
sp_grid, Roo_grid = np.meshgrid(sp_grid, Roo_grid)
big_Roo_grid = np.linspace(4, 5.4, 1000)
big_sp_grid = np.linspace(-1.2, 1.2, 1000)
X, Y = np.meshgrid(big_sp_grid, big_Roo_grid)

# z_ground_no_der = np.load('z_ground_no_der.npy')
#
# ground_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_no_der.flatten())
#
# z_excite_xh_no_der = np.load('z_excite_xh_no_der.npy')
#
# excite_xh_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                           z_excite_xh_no_der.flatten())


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
        self.psit = np.zeros((walkers, 3, 5, 3))


def get_da_psi(coords, excite, shift):
    coords = coords.squeeze()
    psi = np.ones((len(coords), 2))
    # dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    # dists[:, 0] = dists[:, 0] - shift[0]
    if excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * coords ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * coords
        # psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * coords ** 2))
        # psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    return psi


def all_da_psi(coords, excite, shift):
    dx = 1e-3
    psi = np.zeros((len(coords), 3, 1))
    # psi[:, 1] = np.broadcast_to(np.prod(get_da_psi(coords, excite, shift), axis=1)[:, None], (len(coords), 1))
    psi[:, 1] = np.prod(get_da_psi(coords, excite, shift), axis=1)[:, None]
    # for atom in range(5):
    #     for xyz in range(3):
    coords[:] -= dx
    psi[:, 0] = np.prod(get_da_psi(coords, excite, shift), axis=1)[:, None]
    coords[:] += 2*dx
    psi[:, 2] = np.prod(get_da_psi(coords, excite, shift), axis=1)[:, None]
    coords[:] -= dx
    return psi.squeeze()


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


def drift(coords, excite, shift):
    dx = 1e-3
    psi = psi_t(coords, excite, shift)
    der = (psi[:, 2] - psi[:, 0])/dx/psi[:, 1]
    return der, psi


def metropolis(Fqx, Fqy, x, y, psi1, psi2):
    psi_1 = psi1[:, 1]
    psi_2 = psi2[:, 1]
    psi_ratio = (psi_2/psi_1)**2
    a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x.squeeze())))
    # a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
    a = a * psi_ratio
    remove = np.argwhere(psi_2 * psi_1 < 0)
    a[remove] = 0.
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx):
    Drift = sigma**2/2.*Fqx
    randomwalk = np.random.normal(0.0, sigma, size=(len(Psi.coords)))
    y = randomwalk + Drift.squeeze() + np.array(Psi.coords).squeeze()
    Fqy, psi = drift(y, Psi.excite, Psi.shift)
    a = metropolis(Fqx, Fqy, Psi.coords, y, Psi.psit, psi)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept, None]
    Fqx[accept] = Fqy[accept]
    Psi.psit[accept] = psi[accept]
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
    inp = np.array([new_struct]*len(Psi.coords))
    inp[:, 2] = re_unit1*(re + np.sqrt(2)/2*Psi.coords) + new_struct[1]
    inp[:, 4] = re_unit2*(re - np.sqrt(2)/2*Psi.coords) + new_struct[3]
    coords = np.array_split(inp, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


pool = mp.Pool(mp.cpu_count()-1)


def psi_t(coords, excite, shift):
    coords = np.array_split(coords, mp.cpu_count() - 1)
    psi = pool.starmap(all_da_psi, zip(coords, repeat(excite), repeat(shift)))
    psi = np.concatenate(psi)
    return psi


def local_kinetic(Psi):
    dx = 1e-3
    d2psidx2 = ((Psi.psit[:, 0] - 2. * Psi.psit[:, 1] + Psi.psit[:, 2]) / dx ** 2) / Psi.psit[:, 1]
    kin = -1. / 2. * sigma ** 2 / dtau * d2psidx2
    return kin.squeeze()


def E_loc(Psi):
    Psi.El = local_kinetic(Psi) + Psi.V
    # Psi.El[np.abs(Psi.El) > 1] = 0
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
        Biggo_psit = np.array(Psi.psit[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Fqx[i[0]] = Biggo_force
        Psi.psit[i[0]] = Biggo_psit

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
        Biggo_psit = np.array(Psi.psit[i[0]])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[ind] = Biggo_pos
        Psi.V[ind] = Biggo_pot
        Psi.El[ind] = Biggo_el
        Fqx[ind] = Biggo_force
        Psi.psit[ind] = Biggo_psit
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


def run(N_0, time_steps, propagation, equilibration, wait_time, excite, initial_struct, initial_shifts, shift_rate):
    DW = False
    psi = Walkers(N_0, initial_struct, excite, initial_shifts)
    Fqx, psi.psit = drift(psi.coords, psi.excite, psi.shift)
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


# def twod_pot(coords, grid1, grid2):
#     print('started making our grid')
#     mesh = np.array(np.meshgrid(grid1, grid2))
#     gridz = np.reshape(mesh, (2, len(grid1)*len(grid2)))
#     roo_coords = oo_grid(coords, gridz[1])
#     full_coords = shared_prot_grid(roo_coords, gridz[0])
#     print('finished making the grid, now to start the potential')
#     mid = (full_coords[:, 3, 0] - full_coords[:, 1, 0])/2
#     full_coords[:, :, 0] -= mid[:, None]
#     pot = get_pot(full_coords)
#     np.save('coords_for_testing', full_coords)
#     print('finished evaluating the potential')
#     import scipy.sparse as sp
#     return sp.diags([pot], [0]), pot, full_coords
#
#
# def oo_grid(coords, Roo):
#     coords = np.array([coords] * len(Roo))
#     equil_roo_roh_x = coords[0, 3, 0] - coords[0, 4, 0]
#     coords[:, 3, 0] = Roo
#     coords[:, 4, 0] = Roo - equil_roo_roh_x
#     return coords
#
#
# def shared_prot_grid(coords, sp):
#     mid = (coords[:, 3, 0] - coords[:, 1, 0])/2
#     coords[:, 0, 0] = mid-sp
#     return coords
#
#
# new_struct = np.array([
#     [0.000000000000000, 0.000000000000000, 0.000000000000000],
#     [-2.304566686034061, 0.000000000000000, 0.000000000000000],
#     [-2.740400260927908, 1.0814221449986587E-016, -1.866154718409233],
#     [2.304566686034061, 0.000000000000000, 0.000000000000000],
#     [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233]
# ])
# new_struct[:, 0] = new_struct[:, 0] + 2.304566686034061
#
# big_sp_grid = np.linspace(-0.5, 0.5, 100)
# big_Roo_grid = np.linspace(4.6, 5.2, 100)
#
# blah, V, coords = twod_pot(new_struct, big_sp_grid, big_Roo_grid)
# #
# psi = Walkers(len(coords), 0, None, [0.33, 0])
# psi.coords = coords
# psi.V = V
# psi.psit = psi_t(psi.coords, psi.excite, psi.shift)
# psi = E_loc(psi)
# psi.El = psi.El*har2wave
# psit = psi.psit[:, 1, 0, 0]
# #
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# tcc = ax.contourf(big_sp_grid, big_Roo_grid, psi.El.reshape((len(big_Roo_grid), len(big_sp_grid))))
# fig.colorbar(tcc)
# plt.show()

test_structure = np.array([
        [ 2.75704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.07502706, -1.66894299, -0.69579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.84391277,  0.14767547,  1.4669121 ],
])

test_structure2 = np.array([
        [ 2.48704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.09502706, -1.86894299, -0.69579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.24391277,  0.14767547,  1.4669121 ],
])

# wow_im_tired = True
# trials = [4, 6, 8, 9]
# for i in range(10):
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         500, 20000, 250, 500, 500, None, [0.05], [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'ground_state_1d_a_h3o2_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
#
# coords = np.load('coords_for_testing.npy')
# print(coords.shape)
#
# psi = Walkers(10000, test_structure2, 'a', [0, 2.5721982410729867])
# psi.coords = coords
# psi.coords[:, -1, 0] += 0.02
# psi.psit = psi_t(psi.coords, psi.excite, psi.shift)
# psi = pot(psi)
# psi = E_loc(psi)
#
# import matplotlib.pyplot as plt
# roo = np.linspace(3.9, 5.8, 100)
# xh = np.linspace(-1.5, 1.5, 100)
#
# X, Y = np.meshgrid(xh, roo)
#
# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, psi.El.reshape((100, 100)))
# fig.colorbar(tcc)
# plt.xlabel('XH')
# plt.ylabel('Roo')
# plt.show()
#
# psi = Walkers(10000, test_structure2, 'sp', [0, 2.5721982410729867])
# psi.coords = coords
# psi.psit = psi_t(psi.coords, psi.excite, psi.shift)
# psi = pot(psi)
# psi = E_loc(psi)
#
# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, psi.El.reshape((100, 100)))
# fig.colorbar(tcc)
# plt.xlabel('XH')
# plt.ylabel('Roo')
# plt.show()
#
# psi = Walkers(10000, test_structure2, 'a', [0, 2.5721982410729867])
# psi.coords = coords
# psi.psit = psi_t(psi.coords, psi.excite, psi.shift)
# psi = pot(psi)
# psi = E_loc(psi)
#
# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, psi.El.reshape((100, 100)))
# fig.colorbar(tcc)
# plt.xlabel('XH')
# plt.ylabel('Roo')
# plt.show()
# xh_left = [0, 1, 2, 3, 4]
# for i in xh_left:
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, 'sp', test_structure, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'XH_excite_state_full_h3o2_left_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)

# for i in range(5):
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, None, test_structure2, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'ground_state_full_h3o2_{i+6}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
#
# xh_right = [0]
# for i in xh_right:
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, 'sp', test_structure2, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'XH_excite_state_full_h3o2_right_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
asym_left = [0, 1, 2, 3, 4]
# asym_left = [0]
for i in asym_left:
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        20000, 20000, 250, 500, 500, 'a', [-0.05], [0, 2.5721982410729867], [0, 0]
    )
    np.savez(f'Asym_excite_state_1d_a_h3o2_left_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)

asym_right = [0, 1, 2, 3, 4]
# asym_right = [0]
for i in asym_right:
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        20000, 20000, 250, 500, 500, 'a', [0.05], [0, 2.5721982410729867], [0, 0]
    )
    np.savez(f'Asym_excite_state_1d_a_h3o2_right_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)
#