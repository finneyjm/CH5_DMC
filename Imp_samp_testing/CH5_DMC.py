import copy
import CH5pot
from scipy import interpolate
import numpy as np
# from Coordinerds.CoordinateSystems import *
import multiprocessing as mp
from itertools import repeat
# import Timing_p3 as tm

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
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
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
        self.interp = []
        self.psit = np.zeros((walkers, 3, 6, 3))


# Evaluate PsiT for each bond CH bond length in the walker set
def psi_t(rch, interp, type, coords=None, interp_exp=None):
    if type == 'dev_indep':
        psi = np.zeros((len(rch), bonds))
        for i in range(bonds):
            psi[:, i] += interpolate.splev(rch[:, i], interp[i], der=0)
    elif type == 'dev_dep':
        psi = all_da_psi(coords, rch, interp, type, interp_exp)
    elif type == 'fd':
        psi = all_da_psi(coords, rch, interp, type)
    return psi


def all_da_psi(coords, rch, interp, type, interp_exp=None):
    coords = np.array_split(coords, mp.cpu_count() - 1)
    rch = np.array_split(rch, mp.cpu_count() - 1)
    psi = pool.starmap(get_da_psi, zip(coords, rch, repeat(interp), repeat(type), repeat(interp_exp)))
    psi = np.concatenate(psi)
    return psi


def get_da_psi(coords, rch, interp, type, interp_exp=None):
    much_psi = np.zeros((len(coords), 3, 6, 3))
    psi = psi_t_extra(coords, interp, type, interp_exp, rch=rch)
    asdf = np.broadcast_to(np.prod(psi, axis=1)[:, None, None], (len(coords), 6, 3))
    much_psi[:, 1] += asdf
    for atoms in range(6):
        for xyz in range(3):
            coords[:, atoms, xyz] -= dx
            much_psi[:, 0, atoms, xyz] = np.prod(psi_t_extra(coords, interp, type, interp_exp), axis=1)
            coords[:, atoms, xyz] += 2.*dx
            much_psi[:, 2, atoms, xyz] = np.prod(psi_t_extra(coords, interp, type, interp_exp), axis=1)
            coords[:, atoms, xyz] -= dx
    return much_psi


def psi_t_extra(coords, interp, type, interp_exp=None, rch=None):
    if rch is None:
        rch = ch_dist(coords)
    if type == 'dev_dep':
        hh = hh_dist(coords, rch)
    shift = np.zeros((len(coords), bonds))
    psi = np.zeros((len(coords), bonds))
    for i in range(bonds):
        if type == 'dev_dep':
            shift[:, i] = interpolate.splev(hh[:, i], interp_exp, der=0)
        psi[:, i] = interpolate.splev(rch[:, i] - shift[:, i], interp[i], der=0)
    return psi


def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, bonds))
    for i in range(bonds):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch


def hh_dist(carts, rch):
    N = len(carts)
    coords = np.array(carts)
    coords -= np.broadcast_to(coords[:, None, 0], (N, bonds + 1, 3))
    coords[:, 1:] /= np.broadcast_to(rch[:, :, None], (N, bonds, 3))
    hh = np.zeros((N, 5, 5))
    a = np.full((5, 5), True)
    np.fill_diagonal(a, False)
    mask = np.broadcast_to(a, (N, 5, 5))
    for i in range(4):
        for j in np.arange(i + 1, 5):
            hh[:, i, j] = np.sqrt((coords[:, j + 1, 0] - coords[:, i + 1, 0]) ** 2 +
                                  (coords[:, j + 1, 1] - coords[:, i + 1, 1]) ** 2 +
                                  (coords[:, j + 1, 2] - coords[:, i + 1, 2]) ** 2)
    hh += np.transpose(hh, (0, 2, 1))
    blah = hh[mask].reshape(N, 5, 4)
    hh_std = np.std(blah, axis=2)
    return hh_std


# Build the dr/dx matrix that is used for calculating dPsi/dx
def drdx(rch, coords):
    chain = np.zeros((len(coords), 5, 6, 3))
    for xyz in range(3):
        for CH in range(bonds):
            chain[:, CH, 0, xyz] += ((coords[:, 0, xyz]-coords[:, CH+1, xyz])/rch[:, CH])  # dr/dx for the carbon for each bond length
            chain[:, CH, CH+1, xyz] += ((coords[:, CH+1, xyz]-coords[:, 0, xyz])/rch[:, CH])  # dr/dx for the hydrogens for each bond length
    return chain


# Calculate the drift term using dPsi/dx and some nice matrix manipulation
def drift(rch, coords, interp, type, interp_exp=None):
    psi = psi_t(rch, interp, type, coords=coords, interp_exp=interp_exp)
    if type == 'dev_indep':
        dr1 = drdx(rch, coords)  # dr/dx values
        der = np.zeros((len(coords), bonds))  # dPsi/dr evaluation using that nice spline interpolation
        for i in range(bonds):
            der[:, i] += (interpolate.splev(rch[:, i], interp[i], der=1)/psi[:, i])
        a = dr1.reshape((len(coords), 5, 18))
        b = der.reshape((len(coords), 1, 5))
        drift = 2.*np.matmul(b, a).reshape((len(coords), 6, 3))
        return drift, dr1
    else:
        blah = (psi[:, 2] - psi[:, 0])/dx/psi[:, 1]
        return blah, psi


# The metropolis step based on those crazy Green's functions
def metropolis(r1, r2, Fqx, Fqy, x, y, interp, sigmaCH, type, psi1=None, psi2=None):
    if type == 'dev_indep':
        psi_1 = psi_t(r1, interp, type)  # evaluate psi for before the move
        psi_2 = psi_t(r2, interp, type)  # evaluate psi for after the move
        psi_ratio = np.prod(psi_2/psi_1, axis=1)**2
        a = np.exp(1. / 2. * (Fqx + Fqy) * (sigmaCH ** 2 / 4. * (Fqx - Fqy) - (y - x)))
        a = psi_ratio * np.prod(np.prod(a, axis=1), axis=1)
    else:
        psi_1 = psi1[:, 1, 0, 0]
        psi_2 = psi2[:, 1, 0, 0]
        psi_ratio = (psi_2 / psi_1) ** 2
        a = np.exp(1. / 2. * (Fqx + Fqy) * (sigmaCH ** 2 / 4. * (Fqx - Fqy) - (y - x)))
        a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, sigmaCH, type, interp_exp):
    Drift = sigmaCH**2/2.*Fqx   # evaluate the drift term from the F that was calculated in the previous step
    randomwalk = np.zeros((len(Psi.coords), 6, 3))  # normal randomwalk from DMC
    randomwalk[:, 1:6, :] = np.random.normal(0.0, sigmaCH[1, 0], size=(len(Psi.coords), 5, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaCH[0, 0], size=(len(Psi.coords), 3))
    y = randomwalk + Drift + np.array(Psi.coords)  # the proposed move for the walkers
    rchy = ch_dist(y)
    if type == 'dev_indep':
        Fqy, dr1 = drift(rchy, y, Psi.interp, type)  # evaluate new F
        a = metropolis(Psi.zmat, rchy, Fqx, Fqy, Psi.coords, y, Psi.interp, sigmaCH, type)  # Is it a good move?
    elif type == 'dev_dep':
        Fqy, psi = drift(rchy, y, Psi.interp, type, interp_exp)
        a = metropolis(Psi.zmat, rchy, Fqx, Fqy, Psi.coords, y, Psi.interp, sigmaCH, type, psi1=Psi.psit, psi2=psi)
    else:
        Fqy, psi = drift(rchy, y, Psi.interp, type)
        a = metropolis(Psi.zmat, rchy, Fqx, Fqy, Psi.coords, y, Psi.interp, sigmaCH, type, psi1=Psi.psit, psi2=psi)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    # Update everything that is good
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.zmat[accept] = rchy[accept]
    if type == 'dev_indep':
        Psi.drdx[accept] = dr1[accept]
    else:
        Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqx, acceptance


# Function for the potential for the mp to use
def get_pot(coords):
    V = CH5pot.mycalcpot(coords, len(coords))
    return V


# Split up those coords to speed up dat potential
def Potential(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    # Psi.V = np.array(CH5pot.mycalcpot(Psi.coords, len(Psi.coords)))
    return Psi


def local_kinetic(Psi, type, sigmaCH=None, dtau=None):
    if type == 'dev_indep':
        psi = psi_t(Psi.zmat, Psi.interp, type)
        der1 = np.zeros((len(Psi.coords), bonds))
        der2 = np.zeros((len(Psi.coords), bonds))
        dpsidx = np.zeros((len(Psi.coords), bonds))
        for i in range(bonds):
            der1[:, i] = (interpolate.splev(Psi.zmat[:, i], Psi.interp[i], der=1)/psi[:, i])
            dpsidx[:, i] = der1[:, i]*(2./Psi.zmat[:, i])
            der2[:, i] = (interpolate.splev(Psi.zmat[:, i], Psi.interp[i], der=2)/psi[:, i])
        kin = -1./(2.*m_CH)*np.sum(der2+dpsidx, axis=1)
        a = Psi.drdx[:, :, 0]*np.broadcast_to(der1[:, :, None], (len(Psi.coords), 5, 3))
        carb_correct = np.sum(np.sum(a, axis=1)**2-np.sum(a**2, axis=1), axis=1)
        kin += -1./(2.*m_C)*carb_correct
    else:
        d2psidx2 = ((Psi.psit[:, 0] - 2. * Psi.psit[:, 1] + Psi.psit[:, 2]) / dx ** 2) / Psi.psit[:, 1]
        kin = -1. / 2. * np.sum(np.sum(sigmaCH ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin


# Bring together the kinetic and potential energy
def E_loc(Psi, type, sigmaCH=None, dtau=None):
    if type == 'dev_indep':
        Psi.El = local_kinetic(Psi, type) + Psi.V
    else:
        Psi.El = local_kinetic(Psi, type, sigmaCH, dtau) + Psi.V
    return Psi


# Calculate the Eref for use in the weighting
def E_ref_calc(Psi, alpha):
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/len(Psi.coords))
    return E_ref


# Calculate the weights of the walkers and figure out the birth/death if needed
def Weighting(Eref, Psi, Fqx, dtau, DW, type):
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
        if type == 'dev_indep':
            Biggo_drdx = np.array(Psi.drdx[ind])
            Psi.drdx[i[0]] = Biggo_drdx
        else:
            Biggo_psit = np.array(Psi.psit[ind])
            Psi.psit[i[0]] = Biggo_psit
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Psi.zmat[i[0]] = Biggo_zmat
        Fqx[i[0]] = Biggo_force
    return Psi


# Adding up all the descendant weights
def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


class JacobHasNoFile(FileNotFoundError):
    pass


class JacobIsDumb(ValueError):
    pass


# Function to go through the DMC algorithm
def run(N_0, time_steps, dtau, equilibration, wait_time, output,
        imp_samp=False, imp_samp_type='dev_indep', hh_relate=None, trial_wvfn=None, DW=False, dw_num=None, dwfunc=None, rand_samp=True):
    interp_exp = 0
    psi = Walkers(N_0, rand_samp)
    if imp_samp is True:
        if trial_wvfn is None:
            raise JacobHasNoFile('Please supply a trial wavefunction if you wanna do importance sampling')
        if imp_samp_type == 'dev_indep':
            if len(trial_wvfn) == 2:
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(trial_wvfn[0, :], trial_wvfn[1, :], s=0))
            elif len(trial_wvfn) == 5000:
                x = np.linspace(0.4, 6., 5000)
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(x, trial_wvfn, s=0))
            else:
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(trial_wvfn[CH, 0], trial_wvfn[CH, 1], s=0))
        elif imp_samp_type == 'dev_dep':
            if hh_relate is None:
                raise JacobIsDumb('Give me dat hh-rch function')
            interp_exp = interpolate.splrep(hh_relate[0, :], hh_relate[1, :], s=0)
            if len(trial_wvfn) == 2:
                if np.max(trial_wvfn[1, :]) < 0.02:
                    shift = trial_wvfn[0, np.argmin(trial_wvfn[1, :])]
                else:
                    shift = trial_wvfn[0, np.argmax(trial_wvfn[1, :])]
                trial_wvfn[0, :] -= shift
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(trial_wvfn[0, :], trial_wvfn[1, :], s=0))
            elif len(trial_wvfn) == 5000:
                x = np.linspace(0.4, 6., 5000)
                if np.max(trial_wvfn) < 0.02:
                    shift = x[np.argmin(trial_wvfn)]
                else:
                    shift = x[np.argmax(trial_wvfn)]
                x -= shift
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(x, trial_wvfn, s=0))
        elif imp_samp_type == 'fd':
            if len(trial_wvfn) == 2:
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(trial_wvfn[0, :], trial_wvfn[1, :], s=0))
            elif len(trial_wvfn) == 5000:
                x = np.linspace(0.4, 6., 5000)
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(x, trial_wvfn, s=0))
        else:
            raise JacobIsDumb('Not a valid type of importance sampling yet')
    else:
        x = np.linspace(0, 10, num=50000)
        y = np.ones(50000)
        for CH in range(bonds):
            psi.interp.append(interpolate.splrep(x, y, s=0))

    alpha = 1. / (2. * dtau)
    sigmaH = np.sqrt(dtau / m_H)
    sigmaC = np.sqrt(dtau / m_C)
    sigmaCH = np.array([[sigmaC] * 3, [sigmaH] * 3, [sigmaH] * 3, [sigmaH] * 3, [sigmaH] * 3, [sigmaH] * 3])

    if DW is True:
        if dw_num is None:
            raise JacobIsDumb('Indicate the walkers that you want to use with an integer value')
        if dwfunc is None:
            raise JacobHasNoFile('Indicate the walkers to use for des weighting')
        wvfn = np.load(dwfunc)
        psi.coords = wvfn['coords'][dw_num-1]
        psi.weights = wvfn['weights'][dw_num-1]

    if imp_samp_type == 'dev_indep':
        Fqx, psi.drdx = drift(psi.zmat, psi.coords, psi.interp, imp_samp_type)
    elif imp_samp_type == 'dev_dep':
        Fqx, psi.psit = drift(psi.zmat, psi.coords, psi.interp, imp_samp_type, interp_exp=interp_exp)
    else:
        Fqx, psi.psit = drift(psi.zmat, psi.coords, psi.interp, imp_samp_type)

    num_o_collections = int((time_steps-equilibration)/wait_time) + 1
    time = np.zeros(time_steps)
    accept = np.zeros(time_steps)
    Eref_array = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = 0
    num = 0
    wait = float(wait_time)
    for i in range(int(time_steps)):
        wait -= 1.

        psi, Fqx, acceptance = Kinetic(psi, Fqx, sigmaCH, imp_samp_type, interp_exp)
        psi = Potential(psi)
        if imp_samp_type == 'dev_indep':
            psi = E_loc(psi, imp_samp_type)

        else:
            psi = E_loc(psi, imp_samp_type, sigmaCH, dtau)
        if i == 0:
            Eref = E_ref_calc(psi, alpha)

        psi = Weighting(Eref, psi, Fqx, dtau, DW, imp_samp_type)

        Eref = E_ref_calc(psi, alpha)
        Eref_array[i] += Eref
        time[i] += i + 1
        sum_weights[i] += np.sum(psi.weights)
        accept[i] += acceptance

        if i >= int(equilibration)-1 and wait <= 0.:
            wait = float(wait_time)
            Psi_tau = copy.deepcopy(psi)
            coords[num] += Psi_tau.coords
            weights[num] += Psi_tau.weights
            num += 1
    if DW is True:
        des = descendants(psi)
        coords = wvfn['coords'][dw_num-1]
        weights = wvfn['weights'][dw_num-1]
    np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, des=des)
    return time


pool = mp.Pool(mp.cpu_count()-1)


