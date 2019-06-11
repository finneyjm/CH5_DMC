import numpy as np
import copy
from scipy import interpolate
import matplotlib.pyplot as plt

# DMC parameters
dtau = 1.
N_0 = 500
time_total = 1000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6


De = 0.02
sigmaOH = np.sqrt(dtau/m_red)
omega = 3600./har2wave
A = np.sqrt(omega**2 * m_red/(2*De))


Psi_t = np.load('Ground_state_wavefunction_HO.npy')


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.zeros(walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.


def psi_t(coords):
    # wvfn = Psi_t
    # x = wvfn[0, :]
    # y = wvfn[1, :]
    # int = interpolate.splrep(x, y, s=0)
    # return interpolate.splev(coords, int, der=0)
    mw = m_red * omega
    return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))


def drift(coords):
    psi = psi_t(coords)
    # wvfn = Psi_t
    # x = wvfn[0, :]
    # y = wvfn[1, :]
    # int = interpolate.splrep(x, y, s=0)
    # return interpolate.splev(coords, int, der=1)/psi
    mw = m_red * omega
    return ((mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))*-mw*coords)/psi


def sec_dir(coords):
    # wvfn = Psi_t
    # x = wvfn[0, :]
    # y = wvfn[1, :]
    # int = interpolate.splrep(x, y, s=0)
    # return interpolate.splev(coords, int, der=2)
    mw = m_red * omega
    return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))*(mw**2*coords**2-mw)


def metropolis(x, y, Fqx, Fqy):
    psi_x = psi_t(x)
    psi_y = psi_t(y)
    pre_factor = (psi_y/psi_x)**2
    M = pre_factor*np.exp(1./2.*(Fqx + Fqy)*(sigmaOH**2/2.*(Fqx-Fqy) - (y-x)))
    return M


def Kinetic(Psi, Fqx):
    randomwalk = np.random.normal(0.0, sigmaOH, N_0)
    Drift = sigmaOH**2*Fqx
    y = Psi.coords + randomwalk + Drift
    Fqy = drift(y)
    a = metropolis(Psi.coords, y, Fqx, Fqy)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a >= check)
    Psi.coords[accept] = y[accept]
    return Psi, Fqy


def potential(Psi):
    return 1./2.*m_red*omega**2*Psi.coords**2
    # return De*(1. - np.exp(-A*Psi.coords))**2


def E_loc(V, Psi):
    psi = psi_t(Psi.coords)
    kin = -1./(2*m_red)*sec_dir(Psi.coords)
    pot = V*psi
    return (kin + pot)/psi

# number = 1000000
# points = np.linspace(-2, 2, num=number)
# wvfn = Walkers(number)
# wvfn.coords += points
# V = potential(wvfn)
# E = E_loc(V, wvfn)*har2wave
# s = np.std(E)
# m = np.mean(E)
# plt.figure()
# plt.plot(wvfn.coords, E)
# # plt.ylim(1799, 1801)
# plt.savefig('testing_the_local_energy.png')


def E_ref_calc(V, Psi):
    El = E_loc(V, Psi)
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*El)/P - alpha*(np.log(P/P0))
    return E_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vi, Vref, Psi):
    El = E_loc(Vi, Psi)
    Psi.weights = Psi.weights * np.exp(-(El - Vref) * dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
    return Psi


# Descendant weighting where the descendants of the walkers that replace those that die are kept track of
def desWeight(Vi, Vref, Psi):
    El = E_loc(Vi, Psi)
    Psi.weights = Psi.weights*np.exp(-(El-Vref)*dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_num = float(Psi.walkers[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.walkers[i[0]] = Biggo_num
        Psi.coords[i[0]] = Biggo_pos
    return Psi


# Calculates the descendant weight for the walkers after descendant weighting
def descendants(Psi):
    for i in range(N_0):
        Psi.d[i] = np.sum(Psi.weights[Psi.walkers == i])
    return Psi.d


def run(propagation):
    psi = Walkers(N_0)
    # Vstart = potential(psi)
    # El_start = E_loc(Vstart, psi)
    # print(El_start*har2wave)
    Fqx = drift(psi.coords)
    Psi, Fqx = Kinetic(psi, Fqx)
    Vi = potential(Psi)
    Eref_array = np.array([])
    Eref = E_ref_calc(Vi, Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Vi, Eref, Psi)

    # initial parameters before running the calculation
    DW = False  # a parameter that will implement descendant weighting when True
    Psi_dtau = 0
    for i in range(int(time_total)):
        if DW is False:
            prop = float(propagation)

        Psi, Fqx = Kinetic(new_psi, Fqx)
        Eref = E_ref_calc(Vi, Psi)
        Vi = potential(Psi)

        Eref_array = np.append(Eref_array, Eref)

        if DW is False:
            new_psi = Weighting(Vi, Eref, Psi)
        elif DW is True:
            if Psi_dtau == 0:
                Psi_tau = copy.deepcopy(Psi)
                Psi_dtau = copy.deepcopy(Psi_tau)
                new_psi = desWeight(Vi, Eref, Psi_dtau)
            else:
                new_psi = desWeight(Vi, Eref, Psi)
            prop -= 1.

        if i >= (time_total - 1. - float(propagation)) and prop > 0:  # start of descendant weighting
            DW = True
        elif i >= (time_total - 1. - float(propagation)) and prop == 0.:  # end of descendant weighting
            d_values = descendants(new_psi)
            Psi_tau.d += d_values
    wvfn = np.zeros((3, N_0))
    wvfn[0, :] += Psi_tau.coords
    wvfn[1, :] += Psi_tau.weights
    wvfn[2, :] += Psi_tau.d
    np.save('Imp_samp_HO_energy', Eref_array)
    np.save('Imp_samp_HO_Psi', wvfn)
    return


run(100)

