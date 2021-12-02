import numpy as np
from scipy import interpolate
from Potential.Water_monomer_pot_fns import PatrickShinglePotential
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


sym_excite = wvfns['excite_sym'].reshape((len(gridz[0]), len(gridz[1])))

sym_ders = Derivatives(sym_excite, grid1=gridz[0], grid2=gridz[1], fd="do it")
z_sym_dx1 = sym_ders.compute_derivative(dx=1)/sym_excite
z_sym_dx2 = sym_ders.compute_derivative(dx=2)/sym_excite
z_sym_dx1_dy1 = sym_ders.compute_derivative(dx=1, dy=1)/sym_excite
z_sym_dy1 = sym_ders.compute_derivative(dy=1)/sym_excite
z_sym_dy2 = sym_ders.compute_derivative(dy=2)/sym_excite

# np.savez('wvfn_derivs_anti_excite_state', grid=gridz, no_der=anti_excite, dx1=z_anti_dx1, dy1=z_anti_dy1, dx2=z_anti_dx2,
#         dy2=z_anti_dy2, dx1_dy1=z_anti_dx1_dy1)

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

wvfn_derivs = np.load('wvfn_derivs_sym_excite_state.npz')
sym_excite2 = wvfn_derivs['no_der']
gridz = wvfn_derivs['grid']
z_sym_dx12 = wvfn_derivs['dx1']
z_sym_dy12 = wvfn_derivs['dy1']
z_sym_dx22 = wvfn_derivs['dx2']
z_sym_dy22 = wvfn_derivs['dy2']
z_sym_dx1_dy12 = wvfn_derivs['dx1_dy1']

sym_no_der = interpolate.interp2d(gridz[0], gridz[1], sym_excite.T, kind='cubic')
sym_dx1 = interpolate.interp2d(gridz[0], gridz[1], z_sym_dx1.T, kind='cubic')
sym_dx2 = interpolate.interp2d(gridz[0], gridz[1], z_sym_dx2.T, kind='cubic')
sym_dy1 = interpolate.interp2d(gridz[0], gridz[1], z_sym_dy1.T, kind='cubic')
sym_dy2 = interpolate.interp2d(gridz[0], gridz[1], z_sym_dy2.T, kind='cubic')
sym_dx1_dy1 = interpolate.interp2d(gridz[0], gridz[1], z_sym_dx1_dy1.T, kind='cubic')


def interp(x, y, poiuy):
    out = np.zeros(len(x))
    # print(len(out), flush=True)
    # print(x[0], flush=True)
    # print(y[0], flush=True)
    for i in range(len(x)):
        out[i] = poiuy(x[i], y[i])
    return out


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct, excite, initial_shifts, inital_weights=None):
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
        self.shift = np.array(initial_shifts)
        # self.atoms = atoms
        # self.interp = interpa
        self.psit = np.zeros((walkers, 3))
        self.marked_for_death = np.zeros(walkers)


def psi_t(coords, excite, shifts):
    psi = np.ones((len(coords), 1))
    dists = oh_dists(coords)
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    anti = anti - shifts[0]
    sym = sym - shifts[1]
    if excite == 'anti':
        psi[:, 0] = interp(anti, sym, anti_no_der)
    elif excite == 'sym':
        psi[:, 0] = interp(anti, sym, sym_no_der)
    else:
        psi[:, 0] = interp(anti, sym, ground_no_der)
    return psi


def dpsidx(coords, excite, shifts):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)
    collect = dpsidrtheta(coords, dists, excite, shifts)
    return np.matmul(drx, collect[:, None, :, None]).squeeze()


def duvdx(drx):
    return 1/np.sqrt(2)*np.concatenate(((drx[..., 1, None] - drx[..., 0, None]),
                                        (drx[..., 0, None] + drx[..., 1, None])), axis=-1)


def d2psidx2(coords, excite, shifts):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)
    drx2 = drdx2(coords, dists)
    drx2 = duvdx(drx2)
    first_dir = dpsidrtheta(coords, dists, excite, shifts)
    second_dir = d2psidrtheta(coords, dists, excite, shifts)
    part1 = np.matmul(drx2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul((drx**2), second_dir[:, None, :-1, None]).squeeze()
    part3 = np.matmul((drx[..., 0]*drx[..., 1])[..., None], second_dir[:, None, -1, None, None]).squeeze()
    return part1 + part2 + 2*part3


def dpsidrtheta(coords, dists, excite, shifts):
    collect = np.zeros((len(coords), 2))
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    anti = anti - shifts[0]
    sym = sym - shifts[1]
    if excite == 'anti':
        collect[:, 0] = interp(anti, sym, anti_dx1)
        collect[:, 1] = interp(anti, sym, anti_dy1)
    elif excite == 'sym':
        collect[:, 0] = interp(anti, sym, sym_dx1)
        collect[:, 1] = interp(anti, sym, sym_dy1)
    else:
        collect[:, 0] = interp(anti, sym, ground_dx1)
        collect[:, 1] = interp(anti, sym, ground_dy1)
    return collect


def d2psidrtheta(coords, dists, excite, shifts):
    collect = np.zeros((len(coords), 3))
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    anti = anti - shifts[0]
    sym = sym - shifts[1]
    if excite == 'anti':
        collect[:, 0] = interp(anti, sym, anti_dx2)
        collect[:, 1] = interp(anti, sym, anti_dy2)
        collect[:, 2] = interp(anti, sym, anti_dx1_dy1)
    elif excite == 'sym':
        collect[:, 0] = interp(anti, sym, sym_dx2)
        collect[:, 1] = interp(anti, sym, sym_dy2)
        collect[:, 2] = interp(anti, sym, sym_dx1_dy1)
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


def drift(coords, excite, shifts):
    coordz = np.array_split(coords, mp.cpu_count()-1)
    # print('calculating psi', flush=True)
    psi = pool.starmap(psi_t, zip(coordz, repeat(excite), repeat(shifts)))
    psi = np.concatenate(psi)
    # print('calculating dpsidx', flush=True)
    der = 2*np.concatenate(pool.starmap(dpsidx, zip(coordz, repeat(excite), repeat(shifts))))
    return der, psi


def local_kinetic(Psi, sigma):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    # print('calculating d2psidx', flush=True)
    d2psi = pool.starmap(d2psidx2, zip(coords, repeat(Psi.excite), repeat(Psi.shift)))
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
    remove = np.argwhere(psi_2 * psi_1 < 0)[:, 0]
    a[remove] = 0.
    # if len(remove) >= 1:
    #     print("oh no, our table, it's broken")
    # a = np.ones(len(a))
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, sigma):
    randomwalk = np.random.normal(0.0, sigma, size=(len(Psi.coords), sigma.shape[0], sigma.shape[1]))
    Drift = sigma**2/2.*Fqx
    x = np.array(Psi.coords) + randomwalk + Drift
    y = x
    Fqy, psi = drift(y, Psi.excite, Psi.shift)
    Psi.marked_for_death = np.zeros(len(Psi.coords))
    a = metropolis(Fqx, Fqy, Psi.coords, y, Psi.psit, psi, sigma)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)[:, 0]
    Psi.marked_for_death[accept] = np.ones(len(accept))
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
#    print(f'acceptance rate = {acceptance}%')
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


def E_ref_calc(Psi, weighting):
    alpha = 1. / (2. * dtau)
    P0 = sum(Psi.weights_i)
    if weighting == 'discrete':
        P = len(Psi.coords)
        E_ref = np.mean(Psi.El) - alpha*(P-P0)/P0
    else:
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


def Discrete_weighting(Vref, Psi, DW, Fqx):
    probs = np.nan_to_num(np.exp(-(Psi.El - Vref)*dtau))
    kill = np.argwhere(Psi.marked_for_death == 0)[:, 0]
    # probs[kill] = 0
    check = np.random.random(len(Psi.coords))
    death = np.argwhere((1-probs) < (1-check))[:, 0]
    Psi.coords = Psi.coords[death]
    Psi.El = Psi.El[death]
    Psi.psit = Psi.psit[death]
    Fqx = Fqx[death]
    check = check[death]
    probs = probs[death]
    if DW:
        Psi.walkers = Psi.walkers[death]
    int_probs = probs.astype(int)
    zeros = np.argwhere(int_probs == 0)
    int_probs[zeros] = 1
    birth = np.argwhere((probs-int_probs) > check)
    extra_birth = np.argwhere((probs.astype(int)-1) >= 1)[:, 0]
    for i in range(len(extra_birth)):
        extra_birth = np.append(extra_birth, np.array([extra_birth[i]]*(int_probs[extra_birth[i]]-2))).astype(int)
    birth = np.append(birth, extra_birth).astype(int)
    Psi.coords = np.concatenate((Psi.coords, Psi.coords[birth]))
    Psi.El = np.concatenate((Psi.El, Psi.El[birth]))
    Psi.psit = np.concatenate((Psi.psit, Psi.psit[birth]))
    Fqx = np.concatenate((Fqx, Fqx[birth]))
    if DW:
        Psi.walkers = np.concatenate((Psi.walkers, Psi.walkers[birth]))
    else:
        Psi.walkers = np.arange(0, len(Psi.coords))
    return Psi, Fqx


def descendants(Psi, weighting, N_0):
    if weighting == 'discrete':
        d = np.bincount(Psi.walkers)
    else:
        d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < N_0:
        d = np.append(d, 0.)
    return d


def run(N_0, time_steps, propagation, equilibration, wait_time, excite, initial_struct,
        atoms, initial_shifts=[0, 0, 0], shift_rate=[0, 0, 0], inital_weights=None, 
        weight_type='continuous'):
    DW = False
    psi = Walkers(N_0, initial_struct, excite, initial_shifts, inital_weights)
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

    Fqx, psi.psit = drift(psi.coords, psi.excite, psi.shift)
    num_o_collections = int((time_steps - equilibration) / (propagation + wait_time)) + 1
    timez = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    accept = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))
    if weight_type == 'discrete':
        buffer = int(len(psi.coords)/2)
        coords = np.hstack((coords, np.zeros((num_o_collections, buffer, psi.coords.shape[1], psi.coords.shape[2]))))
        des = np.hstack((des, np.zeros((num_o_collections, buffer))))


    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Eref_array = np.zeros(time_steps)

    # jet = plt.cm.jet
    # colors = jet(np.linspace(0, 1, 50))
    # c = 0
    shift = np.zeros((time_steps + 1, len(psi.shift)))
    shift[0] = psi.shift
    shift_rate = np.array(shift_rate)
    psi.shift = np.array(psi.shift)
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
            Eref = E_ref_calc(psi, weight_type)

        psi, Fqx, acceptance = Kinetic(psi, Fqx, sigma)
        shift[i + 1] = psi.shift
        psi = pot(psi)
        psi = E_loc(psi, sigma)
        if weight_type == 'discrete':
            psi, Fqx = Discrete_weighting(Eref, psi, DW, Fqx)
        else:
            psi, Fqx = Weighting(Eref, psi, DW, Fqx)
        Eref = E_ref_calc(psi, weight_type)

        Eref_array[i] = Eref
        timez[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)
        accept[i] = acceptance

        if i >= 5000:
            psi.shift = psi.shift + shift_rate

        if i >= int(equilibration) - 1 and wait <= 0. < prop:
            DW = True
            wait = float(wait_time)
            Psi_tau = copy.deepcopy(psi)
            if weight_type == 'discrete':
                coords[num, :len(Psi_tau.coords)] = Psi_tau.coords
            else:
                coords[num] = Psi_tau.coords
                weights[num] = Psi_tau.weights
        elif prop == 0:
            DW = False
            if weight_type == 'discrete':
                des[num, :len(Psi_tau.coords)] = descendants(psi, weight_type, len(Psi_tau.coords))
            else:
                des[num] = descendants(psi, weight_type, len(psi.coords))
            num += 1

    return coords, weights, timez, Eref_array, sum_weights, accept, des, shift


pool = mp.Pool(mp.cpu_count()-1)


test = 4