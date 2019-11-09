import copy
import CH5pot
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp
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
        self.zmat = CoordinateSet(self.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
        self.weights = np.ones(walkers)
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.drdx = np.zeros((walkers, 6, 6, 3))
        self.interp = []


# Evaluate PsiT for each bond CH bond length in the walker set
def psi_t(zmatrix, interp):
    psi = np.zeros((len(zmatrix), bonds))
    for i in range(bonds):
        psi[:, i] += interpolate.splev(zmatrix[:, i, 1], interp[i], der=0)
    return psi


# Build the dr/dx matrix that is used for calculating dPsi/dx
def drdx(zmatrix, coords):
    chain = np.zeros((len(coords), 5, 6, 3))
    for xyz in range(3):
        for CH in range(bonds):
            chain[:, CH, 0, xyz] += ((coords[:, 0, xyz]-coords[:, CH+1, xyz])/zmatrix[:, CH, 1])  # dr/dx for the carbon for each bond length
            chain[:, CH, CH+1, xyz] += ((coords[:, CH+1, xyz]-coords[:, 0, xyz])/zmatrix[:, CH, 1])  # dr/dx for the hydrogens for each bond length
    return chain


# Calculate the drift term using dPsi/dx and some nice matrix manipulation
def drift(zmatrix, coords, interp):
    psi = psi_t(zmatrix, interp)
    dr1 = drdx(zmatrix, coords)  # dr/dx values
    der = np.zeros((len(coords), bonds))  # dPsi/dr evaluation using that nice spline interpolation
    for i in range(bonds):
        der[:, i] += (interpolate.splev(zmatrix[:, i, 1], interp[i], der=1)/psi[:, i])
    a = dr1.reshape((len(coords), 5, 18))
    b = der.reshape((len(coords), 1, 5))
    drift = np.matmul(b, a)
    return 2.*drift.reshape((len(coords), 6, 3)), dr1


# The metropolis step based on those crazy Green's functions
def metropolis(r1, r2, Fqx, Fqy, x, y, interp, sigmaC, sigmaH):
    psi_1 = psi_t(r1, interp)  # evaluate psi for before the move
    psi_2 = psi_t(r2, interp)  # evaluate psi for after the move
    psi_ratio = 1.
    for i in range(bonds):
        psi_ratio *= (psi_2[:, i]/psi_1[:, i])**2  # evaluate the ratio of psi before and after the move
    a = psi_ratio
    for atom in range(6):
        for xyz in range(3):
            if atom == 0:
                sigma = sigmaC
            else:
                sigma = sigmaH
            # Use dat Green's function
            a *= np.exp(1./2.*(Fqx[:, atom, xyz] + Fqy[:, atom, xyz])*(sigma**2/4.*(Fqx[:, atom, xyz]-Fqy[:, atom, xyz])
                                                                       - (y[:, atom, xyz]-x[:, atom, xyz])))
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, interp, sigmaCH, sigmaH, sigmaC):
    Drift = sigmaCH**2/2.*Fqx   # evaluate the drift term from the F that was calculated in the previous step
    randomwalk = np.zeros((len(Psi.coords), 6, 3))  # normal randomwalk from DMC
    randomwalk[:, 1:6, :] = np.random.normal(0.0, sigmaH, size=(len(Psi.coords), 5, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaC, size=(len(Psi.coords), 3))
    y = randomwalk + Drift + np.array(Psi.coords)  # the proposed move for the walkers
    zmatriy = CoordinateSet(y, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    Fqy, dr1 = drift(zmatriy, y, interp)  # evaluate new F
    a = metropolis(Psi.zmat, zmatriy, Fqx, Fqy, Psi.coords, y, interp, sigmaC, sigmaH)  # Is it a good move?
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    # Update everything that is good
    Psi.coords[accept] = y[accept]
    nah = np.argwhere(a <= check)
    Fqy[nah] = Fqx[nah]
    Psi.zmat[accept] = zmatriy[accept]
    Psi.drdx[accept] = dr1[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqy, acceptance


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


def local_kinetic(Psi, interp):
    psi = psi_t(Psi.zmat, interp)
    der1 = np.zeros((len(Psi.coords), bonds))
    der2 = np.zeros((len(Psi.coords), bonds))
    dpsidx = np.zeros((len(Psi.coords), bonds))
    for i in range(bonds):
        der1[:, i] = (interpolate.splev(Psi.zmat[:, i, 1], interp[i], der=1)/psi[:, i])
        dpsidx[:, i] = der1[:, i]*(2./Psi.zmat[:, i, 1])
        der2[:, i] = (interpolate.splev(Psi.zmat[:, i, 1], interp[i], der=2)/psi[:, i])
    kin = -1./(2.*m_CH)*np.sum(der2+dpsidx, axis=1)
    a = Psi.drdx[:, :, 0]*np.broadcast_to(der1[:, :, None], (len(Psi.coords), 5, 3))
    carb_correct = np.sum(np.sum(a, axis=1)**2-np.sum(a**2, axis=1), axis=1)
    kin += -1./(2.*m_C)*carb_correct
    return kin


# Bring together the kinetic and potential energy
def E_loc(Psi, interp):
    Psi.El = local_kinetic(Psi, interp) + Psi.V
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
        imp_samp=False, trial_wvfn=None, DW=False, dw_num=None, dwfunc=None, rand_samp=True):

    psi = Walkers(N_0, rand_samp)

    if imp_samp is True:
        if trial_wvfn is None:
            raise JacobHasNoFile('Please supply a trial wavefunction if you wanna do importance sampling')
        if len(trial_wvfn) is 2:
            Psi_t = trial_wvfn
            for CH in range(bonds):
                psi.interp.append(interpolate.splrep(Psi_t[CH, 0, :], Psi_t[CH, 1, :], s=0))
        elif len(trial_wvfn) is 1:
            x = np.linspace(0.4, 6., 5000)
            for CH in range(bonds):
                psi.interp.append(interpolate.splrep(x, trial_wvfn, s=0))
        else:
            for CH in range(bonds):
                psi.interp.append(interpolate.splrep(trial_wvfn[CH, 0], trial_wvfn[CH, 1], s=0))

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

    Fqx, psi.drdx = drift(psi.zmat, psi.coords, psi.interp)

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

        psi, Fqx, acceptance = Kinetic(psi, Fqx, psi.interp, sigmaCH, sigmaH, sigmaC)
        psi = Potential(psi)
        psi = E_loc(psi, psi.interp)

        if i == 0:
            Eref = E_ref_calc(psi, alpha)

        psi = Weighting(Eref, psi, Fqx, dtau, DW)

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


pool = mp.Pool(mp.cpu_count()-1)


