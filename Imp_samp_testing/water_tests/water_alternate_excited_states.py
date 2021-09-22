import numpy as np
from scipy import interpolate
from Potential.Water_monomer_pot_fns import PatrickShinglePotential
import multiprocessing as mp
from itertools import repeat
import copy
ang2bohr = 1.e-10/5.291772106712e-11
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_OD = (m_D*m_O)/(m_D+m_O)
m_OH = (m_H*m_O)/(m_H+m_O)
har2wave = 219474.6
omega_OD = 2832.531899782715
omega_OH = 3890.7865072878913
mw_d = m_OD * omega_OD/har2wave
mw_h = m_OH * omega_OH/har2wave
dtau = 1


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct, excite, initial_shifts, atoms, interp):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([initial_struct]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.excite = excite
        self.shift = initial_shifts
        self.atoms = atoms
        self.interp = interp


def psi_t(coords, excite, shift, atoms):
    dists = oh_dists(coords)
    r1 = 1.8100552720044216
    r2 = 1.8100552155510128
    req = [r1, r2]
    dists = dists - req
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            mw1 = mw_h
            mw2 = mw_h
        else:
            mw1 = mw_h
            mw2 = mw_d
    else:
        if atoms[2].upper() == 'H':
            mw1 = mw_d
            mw2 = mw_h
        else:
            mw1 = mw_d
            mw2 = mw_d

    if excite == 'asym' or excite == 'sym':
        psi = np.zeros((len(coords), 2))
        term1 = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 1] ** 2)) * \
                (2 * mw1) ** (1 / 2) * dists[:, 1]
        term1 *= (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
        term2 = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 1] ** 2))
        term2 *= (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2)) * \
                 (2 * mw2) ** (1 / 2) * dists[:, 1]
        if excite == 'asm':
            psi[:, 1] = 1/np.sqrt(2)*(term1 - term2)
        else:
            psi[:, 1] = 1/np.sqrt(2)*(term1 + term2)
    else:
        psi = np.zeros((len(coords), 3))
        psi[:, 1] = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2))
        psi[:, 2] = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
    psi[:, 0] = angle_function(coords, excite, shift, atoms)
    return np.prod(psi, axis=-1)


def angle_function(coords, excite, shift, atoms):
    angs = angle(coords)
    angs = angs - shift[2]
    r1 = 1.8100552720044216
    r2 = 1.8100552155510128
    theta = np.deg2rad(104.50800290215986)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            G = gmat(muH, muH, muO, r1, r2, theta)
            freq = 1668.4590610594878
        else:
            G = gmat(muH, muD, muO, r1, r2, theta)
            freq = 1462.5810039828614
    else:
        if atoms[2].upper() == 'H':
            G = gmat(muD, muH, muO, r1, r2, theta)
            freq = 1462.5810039828614
        else:
            G = gmat(muD, muD, muO, r1, r2, theta)
            freq = 1222.5100195873742
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2) * (2*alpha) ** (1/2) * (angs-theta)
    else:
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2)


def dangle(coords, excite, shift, atoms):
    angs = angle(coords)
    angs = angs - shift[-1]
    r1 = 1.8100552720044216
    r2 = 1.8100552155510128
    theta = np.deg2rad(104.50800290215986)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            G = gmat(muH, muH, muO, r1, r2, theta)
            freq = 1668.4590610594878
        else:
            G = gmat(muH, muD, muO, r1, r2, theta)
            freq = 1462.5810039828614
    else:
        if atoms[2].upper() == 'H':
            G = gmat(muD, muH, muO, r1, r2, theta)
            freq = 1462.5810039828614
        else:
            G = gmat(muD, muD, muO, r1, r2, theta)
            freq = 1222.5100195873742
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        return (1 - alpha * (angs-theta) ** 2) / (angs-theta)
    else:
        return -alpha*(angs-theta)


def d2angle(coords, excite, shift, atoms):
    angs = angle(coords)
    angs = angs - shift[-1]
    r1 = 1.8100552720044216
    r2 = 1.8100552155510128
    theta = np.deg2rad(104.50800290215986)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            G = gmat(muH, muH, muO, r1, r2, theta)
            freq = 1668.4590610594878
        else:
            G = gmat(muH, muD, muO, r1, r2, theta)
            freq = 1462.5810039828614
    else:
        if atoms[2].upper() == 'H':
            G = gmat(muD, muH, muO, r1, r2, theta)
            freq = 1462.5810039828614
        else:
            G = gmat(muD, muD, muO, r1, r2, theta)
            freq = 1222.5100195873742
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        return alpha * (alpha * (angs-theta) ** 2 - 3)
    else:
        return alpha**2*(angs-theta)**2 - alpha