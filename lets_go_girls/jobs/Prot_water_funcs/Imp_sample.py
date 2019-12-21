import copy
import numpy as np
from scipy import interpolate
from Prot_water_funcs.Non_imp_sampled import descendants


# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
m_OD = (m_O*m_D)/(m_D+m_O)
m_OH = (m_O*m_H)/(m_H+m_O)
har2wave = 219474.6

dx = 1.e-3


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, atoms, coords_initial, bond_order):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial]*walkers)*1.01
        self.weights = np.zeros(walkers) + 1.
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.atoms = atoms
        self.El = np.zeros(walkers)
        self.interp = []
        self.psit = np.zeros((walkers, 3, len(atoms), 3))
        self.bond_order = bond_order


def get_da_psi(coords, bond_order):
    much_psi = np.zeros((coords.shape[0], 3, coords.shape[1], coords.shape[2]))
    psi = psi_t(coords, bond_order)
    much_psi[:, 1] = np.broadcast_to(np.prod(psi, axis=1)[:, None, None], coords.shape)
    for atoms in range(coords.shape[1]):
        for xyz in range(coords.shape[2]):
            coords[:, atoms, xyz] -= dx
            much_psi[:, 0, atoms, xyz] = np.prod(psi_t(coords, bond_order), axis=1)
            coords[:, atoms, xyz] += 2.*dx
            much_psi[:, 2, atoms, xyz] = np.prod(psi_t(coords, bond_order), axis=1)
            coords[:, atoms, xyz] -= dx
    return much_psi


def psi_t(coords, num_waters):
    reg_oh = dists(coords, num_waters, 'OH')
    hbond_oh = dists(coords, num_waters, 'hbond_OH')
    hbond_oo = dists(coords, num_waters, 'hbond_OO')

    return np.zeros((len(coords), 2))


def dists(coords, num_waters, dist_type):
    if num_waters == 3:
        if dist_type == 'OH':
            bonds = [[4, 1], [7, 6], [7, 5], [10, 9], [10, 8]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 3], [4, 2]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7], [4, 10]]
        elif dist_type == 'OO':
            bonds = [[7, 10]]
    elif num_waters == 4:
        if dist_type == 'OH':
            bonds = [[7, 6], [7, 5], [10, 9], [10, 8], [13, 12], [13, 11]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 3], [4, 2], [4, 1]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7], [4, 13], [4, 10]]
        elif dist_type == 'OO':
            bonds = [[7, 10], [7, 13], [10, 13]]

    cd1 = coords[:, tuple(x[0] for x in bonds)]
    cd2 = coords[:, tuple(x[1] for x in bonds)]
    dis = np.linalg.norm(cd2-cd1, axis=2).T
    return dis


def drift(coords, bond_order):
    psi = get_da_psi(coords, bond_order)
    der = (psi[:, 2] - psi[:, 0]) / dx / psi[:, 1]
    return der, psi


# The metropolis step based on those crazy Green's functions
def metropolis(Fqx, Fqy, x, y, sigma, psi1, psi2):
    psi_1 = psi1[:, 1, 0, 0]
    psi_2 = psi2[:, 1, 0, 0]
    psi_ratio = (psi_2 / psi_1) ** 2
    a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
    a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, sigma):
    Drift = sigma**2/2.*Fqx   # evaluate the drift term from the F that was calculated in the previous step
    N = len(Psi.coords)
    randomwalk = np.random.normal(0.0, sigma, size=(N, sigma.shape[0], sigma.shape[1]))
    y = randomwalk + Drift + np.array(Psi.coords)  # the proposed move for the walkers
    Fqy, psi = drift(y, Psi.bond_order)
    a = metropolis(Fqx, Fqy, Psi.coords, y, sigma, Psi.psit, psi)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    # Update everything that is good
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept) / len(Psi.coords)) * 100.
    return Psi, Fqx, acceptance


def local_kinetic(Psi, sigma, dtau):
    d2psidx2 = ((Psi.psit[:, 0] - 2. * Psi.psit[:, 1] + Psi.psit[:, 2]) / dx ** 2) / Psi.psit[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin


# Bring together the kinetic and potential energy
def E_loc(Psi, sigma, dtau):
    Psi.El = local_kinetic(Psi, sigma, dtau) + Psi.V
    return Psi


# Calculate the Eref for use in the weighting
def E_ref_calc(Psi, alpha):
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/len(Psi.coords))
    return E_ref


# Calculate the weights of the walkers and figure out the birth/death if needed
def Weighting(Eref, Psi, Fqx, dtau, DW):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    threshold = 1./float(len(Psi.coords))
    death = np.argwhere(Psi.weights < threshold)  # should I kill a walker?
    for i in death:
        ind = np.argmax(Psi.weights)
        # copy things over
        if DW is True:
            Biggo_num = int(Psi.walkers[ind])
            Psi.walkers[i[0]] = Biggo_num
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_el = float(Psi.El[ind])
        Biggo_zmat = np.array(Psi.zmat[ind])
        Biggo_force = np.array(Fqx[ind])
        Biggo_psit = np.array(Psi.psit[ind])
        Psi.psit[i[0]] = Biggo_psit
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Psi.zmat[i[0]] = Biggo_zmat
        Fqx[i[0]] = Biggo_force
    return Psi


def simulation_time(psi, alpha, sigma, Fqx, imp_samp_type, time_steps, dtau,
                    equilibration, wait_time, propagation, multicore=True, interp_exp=None):
    DW = False
    num_o_collections = int((time_steps - equilibration) / (propagation + wait_time)) + 1
    time = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    accept = np.zeros(time_steps)
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))
    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Eref_array = np.zeros(time_steps)

    for i in range(int(time_steps)):
        if DW is False:
            prop = float(propagation)
            wait -= 1.
        else:
            prop -= 1.

        if i == 0:
            if multicore is True:
                psi = Parr_Potential(psi)
            else:
                psi.V = get_pot(psi.coords)
            psi = E_loc(psi, sigma, dtau)
            Eref = E_ref_calc(psi, alpha)

        psi, Fqx, acceptance = Kinetic(psi, Fqx, sigma)

        if multicore is True:
            psi = Parr_Potential(psi)
        else:
            psi.V = get_pot(psi.coords)

        psi = E_loc(psi, sigma, dtau)

        psi = Weighting(Eref, psi, Fqx, dtau, DW)
        Eref = E_ref_calc(psi, alpha)

        Eref_array[i] = Eref
        time[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)
        accept[i] = acceptance

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





