import copy
import numpy as np
from scipy import interpolate
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from Prot_water_funcs.Non_imp_sampled import descendants
from ProtWaterPES import Potential
import multiprocessing as mp
from itertools import repeat


# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
m_OD = (m_O*m_D)/(m_D+m_O)
m_OH = (m_O*m_H)/(m_H+m_O)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

dx = 1.e-3


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, atoms, coords_initial):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial]*walkers)*1.05
        self.weights = np.zeros(walkers) + 1.
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.atoms = atoms
        self.El = np.zeros(walkers)
        self.interp_reg_oh = []
        self.interp_hbond = []
        self.interp_OO_shift = []
        self.interp_OO_scale = []
        self.interp_ang = []
        self.psit = np.zeros((walkers, 3, len(atoms), 3))


# Evaluate PsiT for each bond CH bond length in the walker set
def psi_t(coords, atoms, num_waters, interp_reg_oh, interp_hbond, interp_OO_shift, interp_OO_scale, interp_ang, multicore=True):
    if multicore is True:
        psi = all_da_psi(coords, atoms, num_waters, interp_reg_oh, interp_hbond, interp_OO_shift, interp_OO_scale, interp_ang)
    else:
        psi = get_da_psi(coords, atoms, num_waters, interp_reg_oh, interp_hbond, interp_OO_shift, interp_OO_scale, interp_ang)
    return psi


def all_da_psi(coords, atoms, num_waters, interp_reg_oh, interp_hbond, interp_OO_shift, interp_OO_scale, interp_ang):
    coords = np.array_split(coords, mp.cpu_count() - 1)
    psi = pool.starmap(get_da_psi, zip(coords, repeat(atoms), repeat(num_waters), repeat(interp_reg_oh),
                                       repeat(interp_hbond), repeat(interp_OO_shift), repeat(interp_OO_scale),
                                       repeat(interp_ang)))
    psi = np.concatenate(psi)
    return psi


def get_da_psi(coords, atoms, num_waters, interp_reg_oh, interp_hbond, interp_OO_shift, interp_OO_scale, interp_ang):
    reg_oh = dists(coords, num_waters, 'OH')
    if num_waters > 1:
        hbond_oh = dists(coords, num_waters, 'hbond_OH')
        hbond_oo = dists(coords, num_waters, 'hbond_OO')
        angs = angles(coords, reg_oh, num_waters)
    else:
        hbond_oh = None
        hbond_oo = None
        angs = None

    much_psi = np.zeros((len(coords), 3, len(atoms), 3))
    psi = psi_t_extra(coords, atoms, num_waters, interp_reg_oh, interp_hbond, interp_OO_shift, interp_OO_scale,
                      interp_ang, reg_oh, hbond_oh, hbond_oo, angs)
    much_psi[:, 1] += np.broadcast_to(np.prod(psi, axis=1)[:, None, None], (len(coords), len(atoms), 3))
    for atom_label in range(len(atoms)):
        for xyz in range(3):
            coords[:, atom_label, xyz] -= dx
            much_psi[:, 0, atom_label, xyz] = np.prod(psi_t_extra(coords, atoms, num_waters, interp_reg_oh, interp_hbond,
                                                             interp_OO_shift, interp_OO_scale, interp_ang), axis=1)
            coords[:, atom_label, xyz] += 2.*dx
            much_psi[:, 2, atom_label, xyz] = np.prod(psi_t_extra(coords, atoms, num_waters, interp_reg_oh, interp_hbond,
                                                             interp_OO_shift, interp_OO_scale, interp_ang), axis=1)
            coords[:, atom_label, xyz] -= dx
    return much_psi


def psi_t_extra(coords, atoms, num_waters, interp_reg_oh, interp_hbond=None, interp_OO_shift=None,
                interp_OO_scale=None, interp_ang=None, reg_oh=None, hbond_oh=None, hbond_oo=None, angs=None):
    if reg_oh is None:
        reg_oh = dists(coords, num_waters, 'OH')
        if num_waters > 1:
            hbond_oh = dists(coords, num_waters, 'hbond_OH')
            hbond_oo = dists(coords, num_waters, 'hbond_OO')
            angs = angles(coords, reg_oh, num_waters)

    if num_waters > 1:
        shift = np.zeros((len(coords), int(num_waters-1)))
        scale = np.zeros((len(coords), int(num_waters-1)))
    if num_waters == 2:
        psi = np.zeros((len(coords), 7))
    else:
        psi = np.zeros((len(coords), int(num_waters*3)))

    if num_waters == 1:
        for i in range(3):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
    elif num_waters == 2: 
        for i in range(4):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
        if interp_hbond is None:
            psi[:, 4] = np.ones((len(coords)))
        else:
            shift = shift_calc(hbond_oo, interp_OO_shift)
            scale = scale_calc(hbond_oo, interp_OO_scale)
            psi[:, 4] = interpolate.splev(scale*(hbond_oh-shift), interp_hbond, der=0)
        for k in range(2):
            a = k*2 + k + 1
            psi[:, k+5] = angle_function(angs[:, k], interp_ang, atoms[a:a+2])
    elif num_waters == 3:
        for i in range(5):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
        for j in range(2):
            if interp_hbond is None:
                psi[:, j+5] = np.ones((len(coords)))
            else:
                shift[:, j] = shift_calc(hbond_oo[:, j], interp_OO_shift)
                scale[:, j] = scale_calc(hbond_oo[:, j], interp_OO_scale)
                psi[:, j+5] = interpolate.splev(scale[:, j]*(hbond_oh[:, j]-shift[:, j]), interp_hbond, der=0)
        for k in range(2):
            p = k+1
            a = p*2 + p + 1
            psi[:, k+5+2] = angle_function(angs[:, k], interp_ang, atoms[a:a+2])
    elif num_waters == 4:
        for i in range(6):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
        for j in range(3):
            if interp_hbond is None:
                psi[:, j+6] = np.ones((len(coords)))
            else:
                shift[:, j] = shift_calc(hbond_oo[:, j], interp_OO_shift)
                scale[:, j] = scale_calc(hbond_oo[:, j], interp_OO_scale)
                psi[:, j+6] = interpolate.splev(scale[:, j]*(hbond_oh[:, j]-shift[:, j]), interp_hbond, der=0)
        for k in range(3):
            p = k + 1
            a = p * 2 + p + 1
            psi[:, k+6+3] = angle_function(angs[:, k], interp_ang, atoms[a:a+2])
    return psi


def shift_calc(oo_dists, interp):
    if interp is None:
        return np.zeros(oo_dists.shape)
    else:
        f = np.poly1d(interp)
        oh_max = f(oo_dists)
        return oh_max


def scale_calc(oo_dists, interp):
    if interp is None:
        return np.ones(oo_dists.shape)
    else:
        f = np.poly1d(interp)
        oh_std = f(oo_dists)
        return oh_std


def angle_function(angs, interp, atoms):
    if interp is None:
        r1 = 0.95784*ang2bohr
        r2 = 0.95783997*ang2bohr
        theta = np.deg2rad(104.5080029)
    else:
        r1 = interp['r1']
        r2 = interp['r2']
        theta = interp['theta']
    muH = 1/m_H
    muD = 1/m_D
    muO = 1/m_O
    if atoms[0].upper() == 'H':
        if atoms[1].upper() == 'H':
            G = gmat(muH, muH, muO, r1, r2, theta)
            freq = 1668.4590610594878
        else:
            G = gmat(muH, muD, muO, r1, r2, theta)
            freq = 1462.5810039828614
    else:
        if atoms[1].upper() == 'H':
            G = gmat(muD, muH, muO, r1, r2, theta)
            freq = 1462.5810039828614
        else:
            G = gmat(muD, muD, muO, r1, r2, theta)
            freq = 1222.5100195873742

    freq /= har2wave
    alpha = freq/G
    return (alpha/np.pi)**(1/4)*np.exp(-alpha*(angs-theta)**2/2)


def gmat(mu1, mu2, mu3, r1, r2, ang):
    return mu1/r1**2 + mu2/r2**2 + mu3*(1/r1**2 + 1/r2**2 - 2*np.cos(ang)/(r1*r2))


def angles(coords, dists, num_waters):
    if num_waters == 2:
        v1 = (coords[:, 1] - coords[:, 3]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
        v2 = (coords[:, 2] - coords[:, 3]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))
        v3 = (coords[:, 4] - coords[:, 6]) / np.broadcast_to(dists[:, 2, None], (len(dists), 3))
        v4 = (coords[:, 5] - coords[:, 6]) / np.broadcast_to(dists[:, 3, None], (len(dists), 3))

        v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
        v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
        v3_new = np.reshape(v3, (v3.shape[0], 1, v3.shape[1]))
        v4_new = np.reshape(v4, (v4.shape[0], v4.shape[1], 1))

        ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())
        ang2 = np.arccos(np.matmul(v3_new, v4_new).squeeze())

        return np.vstack((ang1, ang2)).T

    elif num_waters == 3:
        v1 = (coords[:, 4] - coords[:, 6])/np.broadcast_to(dists[:, 1, None], (len(dists), 3))
        v2 = (coords[:, 5] - coords[:, 6])/np.broadcast_to(dists[:, 2, None], (len(dists), 3))
        v3 = (coords[:, 7] - coords[:, 9])/np.broadcast_to(dists[:, 3, None], (len(dists), 3))
        v4 = (coords[:, 8] - coords[:, 9])/np.broadcast_to(dists[:, 4, None], (len(dists), 3))

        v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
        v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
        v3_new = np.reshape(v3, (v3.shape[0], 1, v3.shape[1]))
        v4_new = np.reshape(v4, (v4.shape[0], v4.shape[1], 1))

        ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())
        ang2 = np.arccos(np.matmul(v3_new, v4_new).squeeze())

        return np.vstack((ang1, ang2)).T

    elif num_waters == 4:
        v1 = (coords[:, 4] - coords[:, 6]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
        v2 = (coords[:, 5] - coords[:, 6]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))
        v3 = (coords[:, 7] - coords[:, 9]) / np.broadcast_to(dists[:, 2, None], (len(dists), 3))
        v4 = (coords[:, 8] - coords[:, 9]) / np.broadcast_to(dists[:, 3, None], (len(dists), 3))
        v5 = (coords[:, 10] - coords[:, 12]) / np.broadcast_to(dists[:, 4, None], (len(dists), 3))
        v6 = (coords[:, 11] - coords[:, 12]) / np.broadcast_to(dists[:, 5, None], (len(dists), 3))

        v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
        v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
        v3_new = np.reshape(v3, (v3.shape[0], 1, v3.shape[1]))
        v4_new = np.reshape(v4, (v4.shape[0], v4.shape[1], 1))
        v5_new = np.reshape(v5, (v5.shape[0], 1, v5.shape[1]))
        v6_new = np.reshape(v6, (v6.shape[0], v6.shape[1], 1))

        ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())
        ang2 = np.arccos(np.matmul(v3_new, v4_new).squeeze())
        ang3 = np.arccos(np.matmul(v5_new, v6_new).squeeze())

        return np.vstack((ang1, ang2, ang3)).T


def dists(coords, num_waters, dist_type):
    if num_waters == 1:
        bonds = [[4, 1], [4, 2], [4, 3]]
    elif num_waters == 2:
        if dist_type == 'OH':
            bonds = [[4, 2], [4, 3], [7, 6], [7, 5]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 1]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7]]
    elif num_waters == 3:
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

    cd1 = coords[:, tuple(x[0] for x in np.array(bonds)-1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds)-1)]
    dis = np.linalg.norm(cd2-cd1, axis=2)
    return dis


def drift(coords, atoms, num_waters, interp_reg_oh, interp_hbond, interp_OO_shift, interp_OO_scale, interp_ang, multicore=True):
    psi = psi_t(coords, atoms, num_waters, interp_reg_oh, interp_hbond, interp_OO_shift, interp_OO_scale, interp_ang, multicore)
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
def Kinetic(Psi, Fqx, sigma, multicore=True):
    Drift = sigma**2/2.*Fqx   # evaluate the drift term from the F that was calculated in the previous step
    N = len(Psi.coords)
    num_waters = int((len(Psi.atoms)-1)/3)
    randomwalk = np.random.normal(0.0, sigma, size=(N, sigma.shape[0], sigma.shape[1]))
    y = randomwalk + Drift + np.array(Psi.coords)  # the proposed move for the walkers
    Fqy, psi = drift(y, Psi.atoms, num_waters, Psi.interp_reg_oh, Psi.interp_hbond, Psi.interp_OO_shift,
                     Psi.interp_OO_scale, Psi.interp_ang, multicore)
    a = metropolis(Fqx, Fqy, Psi.coords, y, sigma, Psi.psit, psi)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    # Update everything that is good
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept) / len(Psi.coords)) * 100.
    return Psi, Fqx, acceptance


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def Parr_Potential(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


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
    Psi.weights = Psi.weights * np.nan_to_num(np.exp(-(Psi.El - Eref) * dtau))
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
        Biggo_force = np.array(Fqx[ind])
        Biggo_psit = np.array(Psi.psit[ind])
        Psi.psit[i[0]] = Biggo_psit
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Fqx[i[0]] = Biggo_force
    return Psi


def simulation_time(psi, alpha, sigma, Fqx, time_steps, dtau,
                    equilibration, wait_time, propagation, multicore=True):
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

        psi, Fqx, acceptance = Kinetic(psi, Fqx, sigma, multicore)

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


pool = mp.Pool(mp.cpu_count()-1)

