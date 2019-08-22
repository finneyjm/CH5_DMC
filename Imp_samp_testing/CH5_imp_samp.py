import copy
import CH5pot
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
# import Timing_p3 as tm
import matplotlib.pyplot as plt

# DMC parameters
dtau = 1.
# N_0 = 1000
time_steps = 20000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_CH = (m_C*m_H)/(m_H+m_C)
m_CH5 = ((m_C + m_H*4)*m_H)/(m_H*5 + m_C)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# Values for Simulation
sigmaH = np.sqrt(dtau/m_H)
sigmaC = np.sqrt(dtau/m_C)
sigmaCH = np.array([[sigmaC]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3])
bonds = 5
# Starting orientation of walkers
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]

# ch_stretch = 4
Psi_t = np.load('Switch_min_wvfn_speed_1.0.npy')
interp = interpolate.splrep(Psi_t[0, :], Psi_t[1, :], s=0)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.arange(0, N_0)
        self.coords = np.array([coords_initial]*walkers)
        self.zmat = CoordinateSet(self.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


def psi_t(zmatrix):
    psi = np.zeros((N_0, bonds))
    for i in range(bonds):
        psi[:, i] += interpolate.splev(zmatrix[:, i, 1], interp, der=0)
    return psi


def drdx(zmatrix, coords):
    chain = np.zeros((N_0, 5, 6, 3))
    for xyz in range(3):
        for CH in range(bonds):
            chain[:, CH, 0, xyz] += ((coords[:, 0, xyz]-coords[:, CH+1, xyz])/zmatrix[:, CH, 1])
            chain[:, CH, CH+1, xyz] += ((coords[:, CH+1, xyz]-coords[:, 0, xyz])/zmatrix[:, CH, 1])
    return chain


def drift(zmatrix, coords):
    psi = psi_t(zmatrix)
    dr1 = drdx(zmatrix, coords)
    der = np.zeros((N_0, bonds))
    for i in range(bonds):
        der[:, i] += (interpolate.splev(zmatrix[:, i, 1], interp, der=1)/psi[:, i])
    a = dr1.reshape((N_0, 5, 18))
    b = der.reshape((N_0, 1, 5))
    drift = np.matmul(b, a)
    return 2.*drift.reshape((N_0, 6, 3))


def metropolis(r1, r2, Fqx, Fqy, x, y):
    psi_1 = psi_t(r1)
    psi_2 = psi_t(r2)
    psi_ratio = 1.
    for i in range(bonds):
        psi_ratio *= (psi_2[:, i]/psi_1[:, i])**2
    a = psi_ratio
    for atom in range(6):
        for xyz in range(3):
            if atom == 0:
                sigma = sigmaC
            else:
                sigma = sigmaH
            a *= np.exp(1./2.*(Fqx[:, atom, xyz] + Fqy[:, atom, xyz])*(sigma**2/4.*(Fqx[:, atom, xyz]-Fqy[:, atom, xyz])
                                                                       - (y[:, atom, xyz]-x[:, atom, xyz])))
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx):
    Drift = sigmaCH**2/2.*Fqx
    randomwalk = np.zeros((N_0, 6, 3))
    randomwalk[:, 1:6, :] = np.random.normal(0.0, sigmaH, size=(N_0, 5, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaC, size=(N_0, 3))
    y = randomwalk + Drift + np.array(Psi.coords)
    zmatriy = CoordinateSet(y, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    Fqy = drift(zmatriy, y)
    a = metropolis(Psi.zmat, zmatriy, Fqx, Fqy, Psi.coords, y)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    nah = np.argwhere(a <= check)
    Fqy[nah] = Fqx[nah]
    Psi.zmat[accept] = zmatriy[accept]
    return Psi, Fqy


def Potential(Psi):
    V = CH5pot.mycalcpot(Psi.coords, N_0)
    Psi.V = np.array(V)
    return Psi


def local_kinetic(Psi):
    psi = psi_t(Psi.zmat)
    der1 = np.zeros((N_0, bonds))
    der2 = np.zeros((N_0, bonds))
    for i in range(bonds):
        der1[:, i] += (interpolate.splev(Psi.zmat[:, i, 1], interp, der=1)/psi[:, i]*(2./Psi.zmat[:, i, 1]))
        der2[:, i] += (interpolate.splev(Psi.zmat[:, i, 1], interp, der=2)/psi[:, i])
    kin = -1./(2.*m_CH)*np.sum(der2+der1, axis=1)
    return kin


def E_loc(Psi):
    Psi.El = local_kinetic(Psi) + Psi.V
    return Psi


def E_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/P0)
    return E_ref


def Weighting(Eref, Psi, DW):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    threshold = 1./float(N_0)
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
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    return d


def run(propagation, test_number):
    DW = False
    psi = Walkers(N_0)
    Fqx = drift(psi.zmat, psi.coords)
    Psi, Fqx = Kinetic(psi, Fqx)
    Psi = Potential(Psi)
    Psi = E_loc(Psi)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, DW)

    Psi_tau = 0
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)
        Psi, Fqx = Kinetic(new_psi, Fqx)
        Psi = Potential(Psi)
        Psi = E_loc(Psi)

        if DW is False:
            prop = float(propagation)
        elif DW is True:
            prop -= 1.
            if Psi_tau == 0:
                Psi_tau = copy.deepcopy(Psi)
        new_psi = Weighting(Eref, Psi, DW)
        Eref = E_ref_calc(new_psi)
        Eref_array = np.append(Eref_array, Eref)

        if i >= (time_steps - 1. - float(propagation)) and prop > 0.:
            DW = True
        elif i >= (time_steps - 1. - float(propagation)) and prop == 0.:
            d_values = descendants(new_psi)
    np.save(f'DMC_imp_samp_CH5_energy_{N_0}_walkers_{test_number}', Eref_array)
    return Eref_array


tests = [100, 200, 500, 1000, 2000, 5000, 10000]
for j in range(5):
    for i in range(7):
        N_0 = tests[i]
        run(50, j+1)


# Psi = Walkers(N_0)
# fqx, fq_list = tm.time_me(drift, Psi.zmat, Psi.coords)
# tm.print_time_list(drift, fq_list)
# Psi, Fqx, kin_list = tm.time_me(Kinetic, Psi, fqx)
# tm.print_time_list(Kinetic, kin_list)
# Psi, psi_list = tm.time_me(Potential, Psi)
# tm.print_time_list(Potential, psi_list)
# Psi, psi_list = tm.time_me(E_loc, Psi)
# tm.print_time_list(E_loc, psi_list)
# Eref, eref_list = tm.time_me(E_ref_calc, Psi)
# tm.print_time_list(E_ref_calc, eref_list)
# Psi, weight_list = tm.time_me(Weighting, Eref, Psi, True)
# tm.print_time_list(Weighting, weight_list)
# d, d_list = tm.time_me(descendants, Psi)
# tm.print_time_list(descendants, d_list)
#
# Eref, time = tm.time_me(run, 0)
# tm.print_time_list(run, time)
# plt.plot(Eref*har2wave)
# plt.xlabel('Time')
# plt.ylabel('Energy (cm^-1)')
# plt.ylim(0, 12000)
# plt.savefig('Importance_sampling_Eref_full.png')
# print(np.mean(Eref[1000:])*har2wave)