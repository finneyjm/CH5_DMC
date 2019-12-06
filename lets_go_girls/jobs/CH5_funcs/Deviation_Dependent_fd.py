from scipy import interpolate
import numpy as np
from itertools import repeat
from .Dev_indep_analytic_imp_samp import ch_dist, E_ref_calc
import multiprocessing as mp

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_CH = (m_C*m_H)/(m_H+m_C)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# Starting orientation of walkers
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
bonds = 5
dx = 1.e-3


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, rand_samp=False):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial]*walkers)
        if rand_samp is True:
            rand_idx = np.random.rand(walkers, 5).argsort(axis=1) + 1
            b = self.coords[np.arange(walkers)[:, None], rand_idx]
            self.coords[:, 1:6, :] = b
        else:
            self.coords *= 1.01
        self.zmat = ch_dist(self.coords)
        self.weights = np.ones(walkers)
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.drdx = np.zeros((walkers, 6, 6, 3))
        self.psit = np.zeros((walkers, 3, 6, 3))


# Evaluate PsiT for each bond CH bond length in the walker set
def psi_t(rch, interp, imp_samp_type, coords, interp_exp=None):
    if imp_samp_type == 'dev_dep':
        psi = all_da_psi(coords, rch, interp, type, interp_exp)
    elif imp_samp_type == 'fd':
        psi = all_da_psi(coords, rch, interp, type)
    return psi


def all_da_psi(coords, rch, interp, imp_samp_type, interp_exp=None):
    coords = np.array_split(coords, mp.cpu_count() - 1)
    rch = np.array_split(rch, mp.cpu_count() - 1)
    psi = pool.starmap(get_da_psi, zip(coords, rch, repeat(interp), repeat(imp_samp_type), repeat(interp_exp)))
    psi = np.concatenate(psi)
    return psi


def get_da_psi(coords, rch, interp, imp_samp_type, interp_exp=None):
    much_psi = np.zeros((len(coords), 3, 6, 3))
    psi = psi_t_extra(coords, interp, imp_samp_type, interp_exp, rch=rch)
    asdf = np.broadcast_to(np.prod(psi, axis=1)[:, None, None], (len(coords), 6, 3))
    much_psi[:, 1] += asdf
    for atoms in range(6):
        for xyz in range(3):
            coords[:, atoms, xyz] -= dx
            much_psi[:, 0, atoms, xyz] = np.prod(psi_t_extra(coords, interp, imp_samp_type, interp_exp), axis=1)
            coords[:, atoms, xyz] += 2.*dx
            much_psi[:, 2, atoms, xyz] = np.prod(psi_t_extra(coords, interp, imp_samp_type, interp_exp), axis=1)
            coords[:, atoms, xyz] -= dx
    return much_psi


def psi_t_extra(coords, interp, imp_samp_type, interp_exp=None, rch=None):
    if rch is None:
        rch = ch_dist(coords)
    if imp_samp_type == 'dev_dep':
        hh = hh_dist(coords, rch)
    shift = np.zeros((len(coords), bonds))
    psi = np.zeros((len(coords), bonds))
    for i in range(bonds):
        if type == 'dev_dep':
            shift[:, i] = interpolate.splev(hh[:, i], interp_exp, der=0)
        psi[:, i] = interpolate.splev(rch[:, i] - shift[:, i], interp[i], der=0)
    return psi


def hh_dist(carts, rch):
    N = len(carts)
    coords = np.array(carts)
    # shift the carbon to the origin and everything else along with it
    coords -= np.broadcast_to(coords[:, None, 0], (N, bonds + 1, 3))
    # Normalize each of the bond lengths to 1
    coords[:, 1:] /= np.broadcast_to(rch[:, :, None], (N, bonds, 3))
    hh = np.zeros((N, 5, 5))
    # create a mask because I don't want the diagonals of this guy
    little_mask = np.full((5, 5), True)
    np.fill_diagonal(little_mask, False)
    mask = np.broadcast_to(little_mask, (N, 5, 5))
    # filling in the upper right triangle of hh distances for each walker
    for i in range(4):
        for j in np.arange(i + 1, 5):
            hh[:, i, j] = np.sqrt((coords[:, j + 1, 0] - coords[:, i + 1, 0]) ** 2 +
                                  (coords[:, j + 1, 1] - coords[:, i + 1, 1]) ** 2 +
                                  (coords[:, j + 1, 2] - coords[:, i + 1, 2]) ** 2)
    hh += np.transpose(hh, (0, 2, 1))
    # getting the actual standard deviations that I care about
    hh_std = np.std(hh[mask].reshape(N, 5, 4), axis=2)
    return hh_std


def drift(rch, coords, interp, imp_samp_type, interp_exp=None):
    psi = psi_t(rch, interp, imp_samp_type, coords, interp_exp=interp_exp)
    blah = (psi[:, 2] - psi[:, 0]) / dx / psi[:, 1]
    return blah, psi


# The metropolis step based on those crazy Green's functions
def metropolis(Fqx, Fqy, x, y, sigmaCH, psi1, psi2):
    psi_1 = psi1[:, 1, 0, 0]
    psi_2 = psi2[:, 1, 0, 0]
    psi_ratio = (psi_2 / psi_1) ** 2
    a = np.exp(1. / 2. * (Fqx + Fqy) * (sigmaCH ** 2 / 4. * (Fqx - Fqy) - (y - x)))
    a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, sigmaCH, imp_samp_type, interp_exp=None):
    Drift = sigmaCH**2/2.*Fqx   # evaluate the drift term from the F that was calculated in the previous step
    randomwalk = np.zeros((len(Psi.coords), 6, 3))  # normal randomwalk from DMC
    randomwalk[:, 1:6, :] = np.random.normal(0.0, sigmaCH[1, 0], size=(len(Psi.coords), 5, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaCH[0, 0], size=(len(Psi.coords), 3))
    y = randomwalk + Drift + np.array(Psi.coords)  # the proposed move for the walkers
    rchy = ch_dist(y)
    Fqy, psi = drift(rchy, y, Psi.interp, imp_samp_type, interp_exp)
    a = metropolis(Fqx, Fqy, Psi.coords, y, sigmaCH, imp_samp_type, Psi.psit, psi)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    # Update everything that is good
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.zmat[accept] = rchy[accept]
    Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept) / len(Psi.coords)) * 100.
    return Psi, Fqx, acceptance


def local_kinetic(Psi, sigmaCH, dtau):
    d2psidx2 = ((Psi.psit[:, 0] - 2. * Psi.psit[:, 1] + Psi.psit[:, 2]) / dx ** 2) / Psi.psit[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigmaCH ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin


# Bring together the kinetic and potential energy
def E_loc(Psi, sigmaCH, dtau):
    Psi.El = local_kinetic(Psi, sigmaCH, dtau) + Psi.V
    return Psi


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


pool = mp.Pool(mp.cpu_count()-1)














