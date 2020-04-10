from scipy import interpolate
from .Potential import *
from .Non_imp_sampled import descendants
import copy

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.000000000 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_CH = (m_C*m_H)/(m_H+m_C)
m_CD = (m_C*m_D)/(m_D+m_C)
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


class JacobIsDumb(ValueError):
    pass


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, atoms=None, rand_samp=False):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial]*walkers)*1.01
        if rand_samp is True:
            rand_idx = np.random.rand(walkers, 5).argsort(axis=1) + 1
            b = self.coords[np.arange(walkers)[:, None], rand_idx]
            self.coords[:, 1:6, :] = b
        # else:
        #     self.coords *= 1.01
        self.zmat = ch_dist(self.coords)
        self.weights = np.ones(walkers)
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.drdx = np.zeros((walkers, 6, 6, 3))
        self.interp = []
        if atoms is None:
            atoms = ['C', 'H', 'H', 'H', 'H', 'H']
        self.atoms = atoms
        self.m_red = np.zeros(5)
        for i in range(5):
            if self.atoms[i + 1] == 'H' or self.atoms[i + 1] == 'h':
                self.m_red[i] = m_CH
            elif self.atoms[i + 1] == 'D' or self.atoms[i + 1] == 'd':
                self.m_red[i] = m_CD
            else:
                raise JacobIsDumb('That atom is not currently supported you dingus')


ln


# Evaluate PsiT for each bond CH bond length in the walker set
def psi_t(rch, interp):
    psi = np.zeros((len(rch), bonds))
    for i in range(bonds):
        psi[:, i] += interpolate.splev(rch[:, i], interp[i], der=0)
    return psi


# Build the dr/dx matrix that is used for calculating dPsi/dx
def drdx(rch, coords):
    chain = np.zeros((len(coords), 5, 6, 3))
    for xyz in range(3):
        for CH in range(bonds):
            chain[:, CH, 0, xyz] += ((coords[:, 0, xyz]-coords[:, CH+1, xyz])/rch[:, CH])  # dr/dx for the carbon for each bond length
            chain[:, CH, CH+1, xyz] += ((coords[:, CH+1, xyz]-coords[:, 0, xyz])/rch[:, CH])  # dr/dx for the hydrogens for each bond length
    return chain


# Calculate the drift term using dPsi/dx and some nice matrix manipulation
def drift(rch, coords, interp):
    psi = psi_t(rch, interp)
    dr1 = drdx(rch, coords)  # dr/dx values
    der = np.zeros((len(coords), bonds))  # dPsi/dr evaluation using that nice spline interpolation
    for i in range(bonds):
        der[:, i] += (interpolate.splev(rch[:, i], interp[i], der=1)/psi[:, i])
    a = dr1.reshape((len(coords), 5, 18))
    b = der.reshape((len(coords), 1, 5))
    drift = 2.*np.matmul(b, a).reshape((len(coords), 6, 3))
    return drift, dr1


# The metropolis step based on those crazy Green's functions
def metropolis(r1, r2, Fqx, Fqy, x, y, interp, sigmaCH):
    psi_1 = psi_t(r1, interp)  # evaluate psi for before the move
    psi_2 = psi_t(r2, interp)  # evaluate psi for after the move
    psi_ratio = np.prod(psi_2/psi_1, axis=1)**2
    a = np.exp(1. / 2. * (Fqx + Fqy) * (sigmaCH ** 2 / 4. * (Fqx - Fqy) - (y - x)))
    a = psi_ratio * np.prod(np.prod(a, axis=1), axis=1)
    return a


def Kinetic(Psi, Fqx, sigmaCH):
    Drift = sigmaCH ** 2 / 2. * Fqx  # evaluate the drift term from the F that was calculated in the previous step
    randomwalk = np.zeros((len(Psi.coords), 6, 3))  # normal randomwalk from DMC
    randomwalk[:, 1:6, :] = np.random.normal(0.0, sigmaCH[1:6], size=(len(Psi.coords), 5, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaCH[0], size=(len(Psi.coords), 3))
    y = randomwalk + Drift + np.array(Psi.coords)  # the proposed move for the walkers
    rchy = ch_dist(y)
    Fqy, dr1 = drift(rchy, y, Psi.interp)  # evaluate new F
    a = metropolis(Psi.zmat, rchy, Fqx, Fqy, Psi.coords, y, Psi.interp, sigmaCH)  # Is it a good move?
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    # Update everything that is good
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.zmat[accept] = rchy[accept]
    Psi.drdx[accept] = dr1[accept]
    acceptance = float(len(accept) / len(Psi.coords)) * 100.
    return Psi, Fqx, acceptance


def local_kinetic(Psi):
    psi = psi_t(Psi.zmat, Psi.interp)
    der1 = np.zeros((len(Psi.coords), bonds))
    der2 = np.zeros((len(Psi.coords), bonds))
    dpsidx = np.zeros((len(Psi.coords), bonds))
    for i in range(bonds):
        der1[:, i] = (interpolate.splev(Psi.zmat[:, i], Psi.interp[i], der=1) / psi[:, i])
        dpsidx[:, i] = der1[:, i] * (2. / Psi.zmat[:, i])
        der2[:, i] = (interpolate.splev(Psi.zmat[:, i], Psi.interp[i], der=2) / psi[:, i])
    kin = -1. / 2. * np.sum((der2 + dpsidx)/Psi.m_red, axis=1)
    a = Psi.drdx[:, :, 0] * np.broadcast_to(der1[:, :, None], (len(Psi.coords), 5, 3))
    carb_correct = np.sum(np.sum(a, axis=1) ** 2 - np.sum(a ** 2, axis=1), axis=1)
    kin += -1. / (2. * m_C) * carb_correct
    return kin


def E_loc(Psi):
    Psi.El = local_kinetic(Psi) + Psi.V
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
        Biggo_drdx = np.array(Psi.drdx[ind])
        Psi.drdx[i[0]] = Biggo_drdx
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Psi.zmat[i[0]] = Biggo_zmat
        Fqx[i[0]] = Biggo_force
    return Psi


def simulation_time(psi, alpha, sigmaCH, Fqx, time_steps, dtau, equilibration, wait_time, propagation, multicore=True):
    DW = True
    num_o_collections = int((time_steps - equilibration) / wait_time) + 1
    time = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    accept = np.zeros(time_steps)
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))
    prop = float(propagation)
    num = 0
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
            psi = E_loc(psi)
            Eref = E_ref_calc(psi, alpha)

        psi, Fqx, acceptance = Kinetic(psi, Fqx, sigmaCH)

        if multicore is True:
            psi = Parr_Potential(psi)
        else:
            psi.V = get_pot(psi.coords)

        psi = E_loc(psi)

        psi = Weighting(Eref, psi, Fqx, dtau, DW)
        Eref = E_ref_calc(psi, alpha)

        Eref_array[i] = Eref
        time[i] = i+1
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
