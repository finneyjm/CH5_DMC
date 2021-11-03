import numpy as np
from scipy import interpolate
from Potential.Water_monomer_pot_fns import PatrickShinglePotential
# from Potential.Water_monomer_pot_fns import dipole_h2o
import multiprocessing as mp
from itertools import repeat
import copy
from Imp_samp_testing import Derivatives
import matplotlib.pyplot as plt

# np.random.seed(76)

ang2bohr = 1.e-10/5.291772106712e-11
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
har2wave = 219474.6
dtau = 1


# test_structure = np.array([
#         [0.1680942285,  0.2730106336, -0.2683675641],
#         [0.7515744938, 1.1894828276,  -1.3309202486],
#         [-0.4264345438, 0.9968892161, 0.9222274409],
# ])
#
# coords = np.array([test_structure]*4)
# randomwalk = np.random.normal(0.0, 0.01, size=(4, 3, 3))
# dip = dipole_h2o(coords + randomwalk)

wvfns = np.load('2d_anti_sym_stretch_water_wvfns.npz')
gridz = wvfns['grid']
# ground = np.abs(wvfns['ground'].reshape((len(gridz[0]), len(gridz[1]))))
# ground_ders = Derivatives(ground, grid1=gridz[0], grid2=gridz[1])
# z_ground_dx1 = ground_ders.compute_derivative(dx=1)/ground
# z_ground_dx2 = ground_ders.compute_derivative(dx=2)/ground
# z_ground_dx1_dy1 = ground_ders.compute_derivative(dx=1, dy=1)/ground
# z_ground_dy1 = ground_ders.compute_derivative(dy=1)/ground
# z_ground_dy2 = ground_ders.compute_derivative(dy=2)/ground
#
# np.savez('wvfn_derivs_ground_state', grid=gridz, no_der=ground, dx1=z_ground_dx1, dy1=z_ground_dy1, dx2=z_ground_dx2,
#         dy2=z_ground_dy2, dx1_dy1=z_ground_dx1_dy1)

wvfn_derivs = np.load('wvfn_derivs_ground_state.npz')
ground = wvfn_derivs['no_der']
gridz = wvfn_derivs['grid']
z_ground_dx1 = wvfn_derivs['dx1']
z_ground_dy1 = wvfn_derivs['dy1']
z_ground_dx2 = wvfn_derivs['dx2']
z_ground_dy2 = wvfn_derivs['dy2']
z_ground_dx1_dy1 = wvfn_derivs['dx1_dy1']

ground_no_der = interpolate.interp2d(gridz[0], gridz[1], ground.T, kind='cubic')
ground_dx1 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dx1.T, kind='cubic')
ground_dx2 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dx2.T, kind='cubic')
ground_dy1 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dy1.T, kind='cubic')
ground_dy2 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dy2.T, kind='cubic')
ground_dx1_dy1 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dx1_dy1.T, kind='cubic')


anti_excite = wvfns['excite_anti'].reshape((len(gridz[0]), len(gridz[1])))

ground_ders = Derivatives(anti_excite, grid1=gridz[0], grid2=gridz[1])
z_anti_dx1 = ground_ders.compute_derivative(dx=1)/anti_excite
z_anti_dx2 = ground_ders.compute_derivative(dx=2)/anti_excite
z_anti_dx1_dy1 = ground_ders.compute_derivative(dx=1, dy=1)/anti_excite
z_anti_dy1 = ground_ders.compute_derivative(dy=1)/anti_excite
z_anti_dy2 = ground_ders.compute_derivative(dy=2)/anti_excite

np.savez('wvfn_derivs_anti_excite_state', grid=gridz, no_der=anti_excite, dx1=z_anti_dx1, dy1=z_anti_dy1, dx2=z_anti_dx2,
        dy2=z_anti_dy2, dx1_dy1=z_anti_dx1_dy1)

wvfn_derivs = np.load('wvfn_derivs_anti_excite_state.npz')
anti_excite = wvfn_derivs['no_der']
gridz = wvfn_derivs['grid']
z_anti_dx1 = wvfn_derivs['dx1']
z_anti_dy1 = wvfn_derivs['dy1']
z_anti_dx2 = wvfn_derivs['dx2']
z_anti_dy2 = wvfn_derivs['dy2']
z_anti_dx1_dy1 = wvfn_derivs['dx1_dy1']

anti_no_der = interpolate.interp2d(gridz[0], gridz[1], anti_excite.T, kind='cubic')
anti_dx1 = interpolate.interp2d(gridz[0], gridz[1], z_anti_dx1.T, kind='cubic')
anti_dx2 = interpolate.interp2d(gridz[0], gridz[1], z_anti_dx2.T, kind='cubic')
anti_dy1 = interpolate.interp2d(gridz[0], gridz[1], z_anti_dy1.T, kind='cubic')
anti_dy2 = interpolate.interp2d(gridz[0], gridz[1], z_anti_dy2.T, kind='cubic')
anti_dx1_dy1 = interpolate.interp2d(gridz[0], gridz[1], z_anti_dx1_dy1.T, kind='cubic')


def interp(x, y, poiuy):
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = poiuy(x[i], y[i])
    return out


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct, excite, inital_weights=None):
        self.walkers = np.arange(0, walkers)
        if len(initial_struct) == walkers:
            self.coords = initial_struct
        else:
            self.coords = np.array([initial_struct]*walkers)
        if inital_weights is not None:
            self.weights = inital_weights
        else:
            self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.excite = excite
        # self.shift = initial_shifts
        # self.atoms = atoms
        # self.interp = interpa
        self.psit = np.zeros((walkers, 3))


def psi_t(coords, excite):
    psi = np.ones((len(coords), 1))
    dists = oh_dists(coords)
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    if excite == 'anti':
        psi[:, 0] = interp(anti, sym, anti_no_der)
    else:
        psi[:, 0] = interp(anti, sym, ground_no_der)
    return psi


def dpsidx(coords, excite):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)
    collect = dpsidrtheta(coords, dists, excite)
    return np.matmul(drx, collect[:, None, :, None]).squeeze()


def duvdx(drx):
    return 1/np.sqrt(2)*np.concatenate(((drx[..., 1, None] - drx[..., 0, None]),
                                        (drx[..., 0, None] + drx[..., 1, None])), axis=-1)


def d2psidx2(coords, excite):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)
    drx2 = drdx2(coords, dists)
    drx2 = duvdx(drx2)
    first_dir = dpsidrtheta(coords, dists, excite)
    second_dir = d2psidrtheta(coords, dists, excite)
    part1 = np.matmul(drx2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul((drx**2), second_dir[:, None, :-1, None]).squeeze()
    part3 = np.matmul((drx[..., 0]*drx[..., 1])[..., None], second_dir[:, None, -1, None, None]).squeeze()
    return part1 + part2 + 2*part3


def dpsidrtheta(coords, dists, excite):
    collect = np.zeros((len(coords), 2))
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    if excite == 'anti':
        collect[:, 0] = interp(anti, sym, anti_dx1)
        collect[:, 1] = interp(anti, sym, anti_dy1)
    else:
        collect[:, 0] = interp(anti, sym, ground_dx1)
        collect[:, 1] = interp(anti, sym, ground_dy1)
    return collect


def d2psidrtheta(coords, dists, excite):
    collect = np.zeros((len(coords), 3))
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    if excite == 'anti':
        collect[:, 0] = interp(anti, sym, anti_dx2)
        collect[:, 1] = interp(anti, sym, anti_dy2)
        collect[:, 2] = interp(anti, sym, anti_dx1_dy1)
    else:
        collect[:, 0] = interp(anti, sym, ground_dx2)
        collect[:, 1] = interp(anti, sym, ground_dy2)
        collect[:, 2] = interp(anti, sym, ground_dx1_dy1)
    return collect


def oh_dists(coords):
    bonds = [[1, 2], [1, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def drdx(coords, dists):
    chain = np.zeros((len(coords), 3, 3, 2))
    for bond in range(2):
        chain[:, 0, :, bond] += ((coords[:, 0]-coords[:, bond+1])/dists[:, bond, None])
        chain[:, bond+1, :, bond] += ((coords[:, bond+1]-coords[:, 0])/dists[:, bond, None])
    return chain


def drdx2(coords, dists):
    chain = np.zeros((len(coords), 3, 3, 2))
    for bond in range(2):
        chain[:, 0, :, bond] = (1./dists[:, bond, None] - (coords[:, 0]-coords[:, bond+1])**2/dists[:, bond, None]**3)
        chain[:, bond + 1, :, bond] = (1./dists[:, bond, None] - (coords[:, bond + 1] - coords[:, 0])**2 / dists[:, bond, None]**3)
    return chain


def drift(coords, excite):
    coordz = np.array_split(coords, mp.cpu_count()-1)
    psi = pool.starmap(psi_t, zip(coordz, repeat(excite)))
    psi = np.concatenate(psi)
    der = 2*np.concatenate(pool.starmap(dpsidx, zip(coordz, repeat(excite))))
    return der, psi


def local_kinetic(Psi, sigma):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    d2psi = pool.starmap(d2psidx2, zip(coords, repeat(Psi.excite)))
    d2psi = np.concatenate(d2psi)
    kin = -1/2 * np.sum(np.sum(sigma**2/dtau*d2psi, axis=1), axis=1)
    return kin


# def all_da_psi(coords):
#     dx = 1e-3
#     psi = np.zeros((len(coords), 3, 3, 3))
#     psi[:, 1] = np.broadcast_to(np.prod(psi_t(coords), axis=1)[:, None, None],
#                                 (len(coords), 3, 3))
#     for atom in range(3):
#         for xyz in range(3):
#             coords[:, atom, xyz] -= dx
#             psi[:, 0, atom, xyz] = np.prod(psi_t(coords), axis=1)
#             coords[:, atom, xyz] += 2*dx
#             psi[:, 2, atom, xyz] = np.prod(psi_t(coords), axis=1)
#             coords[:, atom, xyz] -= dx
#     return psi
#
#
# def local_kinetic_finite(Psi, sigma):
#     dx = 1e-3
#     d2psidx2 = ((Psi[:, 0] - 2. * Psi[:, 1] + Psi[:, 2]) / dx ** 2) / Psi[:, 1]
#     # d2psidx2 = ((Psi[:, 0] - 2. * Psi[:, 1] + Psi[:, 2]) / dx ** 2)
#     kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2, axis=1), axis=1)
#     return kin
#
#
# def drift_fd(coords):
#     dx = 1e-3
#     psi = all_da_psi(coords)
#     der = (psi[:, 2] - psi[:, 0])/dx/psi[:, 1]
#     # der = (psi[:, 2] - psi[:, 0])/dx
#     return der


def metropolis(Fqx, Fqy, x, y, psi_1, psi_2, sigma):
    psi_ratio = np.prod((psi_2/psi_1)**2, axis=-1)
    a_full = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
    a = np.prod(np.prod(a_full, axis=1), axis=1) * psi_ratio
    remove = np.argwhere(psi_2 * psi_1 < 0)
    a[remove] = 0.
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, sigma):
    randomwalk = np.random.normal(0.0, sigma, size=(len(Psi.coords), sigma.shape[0], sigma.shape[1]))
    Drift = sigma**2/2.*Fqx
    x = np.array(Psi.coords) + randomwalk + Drift
    y = x
    Fqy, psi = drift(y, Psi.excite)
    a = metropolis(Fqx, Fqy, Psi.coords, y, Psi.psit, psi, sigma)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    # print(f'acceptance rate = {acceptance}%')
    return Psi, Fqx, acceptance


def get_pot(coords):
    V = PatrickShinglePotential(coords)
    return V


def pot(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


def E_loc(Psi, sigma):
    Psi.El = local_kinetic(Psi, sigma) + Psi.V
    return Psi


def E_ref_calc(Psi):
    alpha = 1. / (2. * dtau)
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
        Biggo_psi = np.array(Psi.psit[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Psi.psit[i[0]] = Biggo_psi
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
        Biggo_psi = np.array(Psi.psit[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[ind] = Biggo_pos
        Psi.V[ind] = Biggo_pot
        Psi.El[ind] = Biggo_el
        Psi.psit[i[0]] = Biggo_psi
        Fqx[ind] = Biggo_force
    return Psi, Fqx


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


def run(N_0, time_steps, propagation, equilibration, wait_time, excite, initial_struct,
        atoms, inital_weights=None):
    DW = False
    psi = Walkers(N_0, initial_struct, excite, inital_weights)
    sigma = np.zeros((3, 3))
    sigma[0] = np.array([[np.sqrt(dtau/m_O)] * 3])
    if atoms[1].upper() == 'H':
        sigma[1] = np.array([[np.sqrt(dtau/m_H)]*3])
    else:
        sigma[1] = np.array([[np.sqrt(dtau/m_D)]*3])
    if atoms[2].upper() == 'H':
        sigma[2] = np.array([[np.sqrt(dtau/m_H)]*3])
    else:
        sigma[2] = np.array([[np.sqrt(dtau/m_D)]*3])

    Fqx, psi.psit = drift(psi.coords, psi.excite)
    num_o_collections = int((time_steps - equilibration) / (propagation + wait_time)) + 1
    timez = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    accept = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))

    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Eref_array = np.zeros(time_steps)

    # jet = plt.cm.jet
    # colors = jet(np.linspace(0, 1, 50))
    # c = 0
    # shift = np.zeros((time_steps + 1, len(psi.shift)))
    # shift[0] = psi.shift
    # shift_rate = np.array(shift_rate)
    # psi.shift = np.array(psi.shift)
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i, flush=True)

        # dis1 = oh_dists(psi.coords)
        # anti1 = 1/np.sqrt(2)*(dis1[:, 1] + dis1[:, 0])
        # amp, xx = np.histogram(anti1, weights=psi.weights, bins=70, density=True)
        # x = (xx[1:] + xx[:-1]) / 2
        # plt.plot(x, amp, color=colors[c])
        # c += 1
        #
        # if (i+1)%50 == 0:
        #     print(f'time step {i+1}')
        #     c=0
        #     plt.ylim(-0.5, 20)
        #     plt.xlabel('s (Bohr)')
        #     plt.show()

        if DW is False:
            prop = float(propagation)
            wait -= 1.
        else:
            prop -= 1.

        if i == 0:
            psi = pot(psi)
            psi = E_loc(psi, sigma)
            Eref = E_ref_calc(psi)

        psi, Fqx, acceptance = Kinetic(psi, Fqx, sigma)
        # shift[i + 1] = psi.shift
        psi = pot(psi)
        psi = E_loc(psi, sigma)

        psi, Fqx = Weighting(Eref, psi, DW, Fqx)
        Eref = E_ref_calc(psi)

        Eref_array[i] = Eref
        timez[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)
        accept[i] = acceptance

        # if i >= 5000:
        #     psi.shift = psi.shift + shift_rate

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

    return coords, weights, timez, Eref_array, sum_weights, accept, des


pool = mp.Pool(mp.cpu_count()-1)


# test_structure = np.array([
#         [0.1680942285,  0.2730106336, -0.2683675641],
#         [0.7515744938, 1.1894828276,  -1.3309202486],
#         [-0.4264345438, 0.9968892161, 0.9222274409],
# ])

# blah1 = np.load('test_ground_state_h2o_chain_rule_2d_1.npz')
#
# test_structure = blah1['coords'][-1]
#
# test_structure = np.array([test_structure]).reshape((1000, 3, 3))
#
# test_weights = np.array([blah1['weights'][-1]]).flatten()

# coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         1000, 2000, 250, 500, 500, test_structure,
#         ['O', 'H', 'H'], test_weights)
# np.savez(f"test_ground_state_h2o_chain_rule_2d_1", coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
#
# blah = np.load('test_ground_state_h2o_chain_rule_2d_1.npz')
# print(np.mean(blah['Eref'][500:]*har2wave))



#
#
# from Coordinerds.CoordinateSystems import *
#
#
# def linear_combo_stretch_grid(r1, r2, coords):
#     re = np.linalg.norm(coords[0]-coords[1])
#     re2 = np.linalg.norm(coords[0]-coords[2])
#     coords = np.array([coords] * 1)
#     zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
#                                                                         ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
#                                                                                    [2, 0, 1, 0]])).coords
#     N = len(r1)
#     zmat = np.array([zmat]*N).squeeze()
#     zmat[:, 0, 1] = r1
#     zmat[:, 1, 1] = r2
#     new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     return new_coords
#
#
# def grid_angle(a, b, num, coords):
#     spacing = np.linspace(a, b, num)
#     zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
#                                                                         ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
#                                                                                    [2, 0, 1, 0]])).coords
#     g = np.array([zmat]*num)
#     g[:, 1, 3] = spacing
#     new_coords = CoordinateSet(g, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     return new_coords
#
#
# def grid_dis(a, b, num, coords, r):
#     spacing = np.linspace(a, b, num)
#     zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
#                                                                         ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
#                                                                                    [2, 0, 1, 0]])).coords
#     g = np.array([zmat]*num)
#     g[:, r-1, 1] = spacing
#     new_coords = CoordinateSet(g, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     return new_coords
#
#
# def angle(coords):
#     dists = oh_dists(coords)
#     v1 = (coords[:, 1] - coords[:, 0]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
#     v2 = (coords[:, 2] - coords[:, 0]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))
#
#     ang1 = np.arccos(np.matmul(v1[:, None, :], v2[..., None]).squeeze())
#
#     return ang1.T
#
#
# # anti = np.load('antisymmetric_stretch_water_wvfns.npz')
# # sym = np.load('symmetric_stretch_water_wvfns.npz')
# #
# # anti2 = np.load('antisymmetric_stretch_water_wvfns_sym_105.npz')
#
# molecule = np.load('monomer_coords.npy')
#
# import pyvibdmc as pv
#
# num_points = 200
# mode = '2d'
# excite = None
#
# # if excite == 'ang':
# #     interp_anti = interpolate.splrep(anti['grid'], anti['ground'], s=0)
# #     interp_sym = interpolate.splrep(sym['grid'], sym['ground'], s=0)
# #     interp = [interp_anti, interp_sym]
# # elif excite == 'asym':
# #     interp_anti = interpolate.splrep(anti['grid'], anti['excite'], s=0)
# #     interp_sym = interpolate.splrep(sym['grid'], sym['ground'], s=0)
# #     interp = [interp_anti, interp_sym]
# # elif excite == 'sym':
# #     interp_anti = interpolate.splrep(anti['grid'], anti['ground'], s=0)
# #     interp_sym = interpolate.splrep(sym['grid'], sym['excite'], s=0)
# #     interp = [interp_anti, interp_sym]
# # else:
# #     interp_anti = interpolate.splrep(anti['grid'], anti['ground'], s=0)
# #     interp_sym = interpolate.splrep(sym['grid'], sym['ground'], s=0)
# #     interp = [interp_anti, interp_sym]
# #     interp_anti2 = interpolate.splrep(anti2['grid'], anti2['ground'], s=0)
# #     interp2 = [interp_anti2, interp_sym]
# interpa = [0, 0, 0]
#
# if mode == 'sym':
#     re = 0.95784 * ang2bohr
#     re2 = 0.95783997 * ang2bohr
#     sym = np.linspace(-0.35, 0.75, num_points) + 1/np.sqrt(2)*(re + re2)
#     anti = np.zeros(num_points) + 0.15
#     A = 1/np.sqrt(2)*np.array([[-1, 1], [1, 1]])
#     eh = np.matmul(np.linalg.inv(A), np.vstack((anti, sym)))
#     r1 = eh[0]
#     r2 = eh[1]
#     grid = linear_combo_stretch_grid(r1, r2, molecule)
#     aang = angle(grid)/2
#     rot_mat = pv.MolRotator.gen_rot_mats(-aang, 2)
#     grid = pv.MolRotator.rotate_geoms(rot_mat, grid)
#     g = sym
#     label = 's Bohr'
# elif mode == 'asym':
#     anti = np.linspace(-0.75, 0.75, num_points)
#     sym = np.zeros(num_points) + 0.15
#     A = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
#     eh = np.matmul(np.linalg.inv(A), np.vstack((anti, sym)))
#     r1 = eh[0]
#     r2 = eh[1]
#     grid = linear_combo_stretch_grid(r1, r2, molecule)
#     aang = angle(grid) / 2
#     rot_mat = pv.MolRotator.gen_rot_mats(-aang, 2)
#     grid = pv.MolRotator.rotate_geoms(rot_mat, grid)
#     g = anti
#     label = 'a Bohr'
# elif mode == '2d':
#     re = 0.95784 * ang2bohr
#     re2 = 0.95783997 * ang2bohr
#     a = np.linspace(-0.70, 0.70, num_points)
#     s = np.linspace(-0.35, 0.75, num_points) + 1/np.sqrt(2)*(re + re2)
#     anti, sym = np.meshgrid(a, s, indexing='ij')
#     A = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
#     eh = np.matmul(np.linalg.inv(A), np.vstack((anti.flatten(), sym.flatten())))
#     r1 = eh[0]
#     r2 = eh[1]
#     grid = linear_combo_stretch_grid(r1, r2, molecule)
#     aang = angle(grid) / 2
#     rot_mat = pv.MolRotator.gen_rot_mats(-aang, 2)
#     grid = pv.MolRotator.rotate_geoms(rot_mat, grid)
# elif mode == 'ang':
#     theta = np.deg2rad(104.50800290215986)
#     grid = grid_angle(theta-1, theta+1, num_points, molecule)
#     aang = angle(grid) / 2
#     rot_mat = pv.MolRotator.gen_rot_mats(-aang, 2)
#     grid = pv.MolRotator.rotate_geoms(rot_mat, grid)
#     g = np.rad2deg(np.linspace(theta-1, theta+1, num_points))
# elif mode == 'r1':
#     grid = grid_dis(1, 3, num_points, molecule, 1)
#     aang = angle(grid) / 2
#     rot_mat = pv.MolRotator.gen_rot_mats(-aang, 2)
#     grid = pv.MolRotator.rotate_geoms(rot_mat, grid)
#     g = np.linspace(1, 3, num_points)
#     label = 'r1 Bohr'
# elif mode == 'r2':
#     grid = grid_dis(1, 3, num_points, molecule, 2)
#     aang = angle(grid) / 2
#     rot_mat = pv.MolRotator.gen_rot_mats(-aang, 2)
#     grid = pv.MolRotator.rotate_geoms(rot_mat, grid)
#     g = np.linspace(1, 3, num_points)
#     label = 'r2 Bohr'
#
#
# psi = Walkers(num_points, molecule, excite, [0, 0, 0], ['O', 'H', 'H'], interpa)
# psi.coords = grid
#
# d, _ = drift(psi.coords, excite, [0, 0, 0], ['O', 'H', 'H'], interpa)
# psi = pot(psi)
# sigma = np.zeros((3, 3))
# sigma[0] = np.array([[np.sqrt(dtau / m_O)] * 3])
# sigma[1] = np.array([[np.sqrt(dtau / m_H)] * 3])
# sigma[2] = np.array([[np.sqrt(dtau / m_H)] * 3])
#
# d = d*sigma**2/2
#
# psi = E_loc(psi, sigma)
# psi_fd = all_da_psi(psi.coords, psi.excite, psi.shift, interpa, psi.atoms)
# eloc_fd = local_kinetic_finite(psi_fd, sigma) + psi.V
# d_fd = drift_fd(psi_fd)*sigma**2/2
# import matplotlib.pyplot as plt
# atom = ['O', 'H1', 'H2']
# xyz = ['x', 'y', 'z']
# if mode == '2d':
#     for i in range(3):
#         for j in range(3):
#             fig, axes = plt.subplots(3)
#             cb0 = axes[0].contourf(anti, sym, d[:, i, j].reshape((num_points, num_points))/ang2bohr)
#             cb1 = axes[1].contourf(anti, sym, d_fd[:, i, j].reshape((num_points, num_points))/ang2bohr)
#             cb2 = axes[2].contourf(anti, sym, (d[:, i, j]-d_fd[:, i, j]).reshape((num_points,
#                                                                                   num_points))/ang2bohr)
#             plt.colorbar(cb0, ax=axes[0])
#             plt.colorbar(cb1, ax=axes[1])
#             plt.colorbar(cb2, ax=axes[2])
#             axes[0].set_xlabel('a (Bohr)')
#             axes[1].set_xlabel('a (Bohr)')
#             axes[2].set_xlabel('a (Bohr)')
#             axes[0].set_ylabel('s (Bohr)')
#             axes[2].set_ylabel('s (Bohr)')
#             axes[1].set_ylabel('s (Bohr)')
#             fig.suptitle(f'Drift of {atom[i]}{xyz[j]}')
#             plt.savefig(f'Drift_of_{mode}_with_{excite}_excite_2d')
#             # plt.show()
#             plt.close()
# else:
#     for i in range(3):
#         for j in range(3):
#             plt.plot(g, d[:, i, j]/ang2bohr, label=f'{atom[i]}{xyz[j]}')
#             # plt.plot(g, d_fd[:, i, j] / ang2bohr, label=f'{atom[i]}{xyz[j]} fd')
#     if mode == 'ang':
#         plt.xlabel(r'$\rm{\theta}^{\circ}$')
#     else:
#         plt.xlabel(label)
#     plt.ylabel(r'D [/$\rm{\AA}$]')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'Drift_of_{mode}_with_{excite}_excite_lin_combo')
#     # plt.show()
#     plt.close()
#
# if mode == '2d':
#     fig, axes = plt.subplots(3, figsize=(10, 6))
#     cb0 = axes[0].contourf(anti, sym, psi.El.reshape((num_points, num_points))*har2wave)
#     cb1 = axes[1].contourf(anti, sym, eloc_fd.reshape((num_points, num_points))*har2wave)
#     cb2 = axes[2].contourf(anti, sym, (psi.El-eloc_fd).reshape((num_points, num_points))*har2wave)
#     plt.colorbar(cb0, ax=axes[0])
#     plt.colorbar(cb1, ax=axes[1])
#     plt.colorbar(cb2, ax=axes[2])
#     axes[0].set_xlabel('a (Bohr)')
#     axes[1].set_xlabel('a (Bohr)')
#     axes[2].set_xlabel('a (Bohr)')
#     axes[0].set_ylabel('s (Bohr)')
#     axes[2].set_ylabel('s (Bohr)')
#     axes[1].set_ylabel('s (Bohr)')
#     plt.tight_layout()
#     plt.savefig(f'Local_energy_of_{mode}_with_{excite}_excite_2d')
#     plt.show()
# else:
#     plt.plot(g, psi.V*har2wave, label='potential')
#     plt.plot(g, psi.El*har2wave, label='local energy')
#     plt.plot(g, eloc_fd*har2wave, label=r'local energy fd')
#     if mode == 'ang':
#         plt.xlabel(r'$\rm{\theta}^{\circ}$')
#     else:
#         plt.xlabel(label)
#     plt.ylabel(r'Energy [/cm$^{-1}$]')
#     # plt.ylim(-1000, 10000)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'Local_energy_of_{mode}_with_{excite}_excite_lin_combo')
#     plt.show()
#
#
#
#
#
#
