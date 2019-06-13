import numpy as np
import copy
from scipy import interpolate
import matplotlib.pyplot as plt

# DMC parameters
dtau = 1.
N_0 = 500
time_total = 10000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6

# parameters for the potential and for the analytic wavefuntion
De = 0.02
sigmaOH = np.sqrt(dtau/m_red)
omega = 3600./har2wave
mw = m_red * omega
A = np.sqrt(omega**2 * m_red/(2*De))


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.zeros(walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.


def Kinetic(psi):
    randomwalk = np.random.normal(0.0, sigmaOH, size=N_0)
    psi.coords += randomwalk
    return psi


def potential(Psi):
    # return 1./2.*m_red*omega**2*Psi.coords**2  # Harmonic potential
    return De*(1. - np.exp(-A*Psi.coords))**2  # Morse potential


def E_ref_calc(V, Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*V)/P - alpha*(sum((Psi.weights-Psi.weights_i))/P0)
    return E_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(V, Vref, Psi):
    Psi.weights = Psi.weights * np.exp(-(V - Vref) * dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:  # iterate over the list of dead walkers
        ind = np.argmax(Psi.weights)  # find the walker with with most weight
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
    return Psi


# Descendant weighting where the descendants of the walkers that replace those that die are kept track of
def desWeight(V, Vref, Psi):
    Psi.weights = Psi.weights * np.exp(-(V-Vref)*dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_num = float(Psi.walkers[ind])  # make sure to keep track of the walker that is donating its weight
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
    Psi = Kinetic(psi)
    Vi = potential(Psi)
    Eref = np.array([])
    Vref = E_ref_calc(Vi, Psi)
    Eref = np.append(Eref, Vref)
    new_psi = Weighting(Vi, Vref, Psi)

    # initial parameters before running the calculation
    DW = False  # a parameter that will implement descendant weighting when True
    Psi_dtau = 0  #
    for i in range(int(time_total)):
        if DW is False:
            prop = float(propagation)

        Psi = Kinetic(new_psi)
        Vi = potential(Psi)

        if DW is False:
            new_psi = Weighting(Vi, Vref, Psi)
        elif DW is True:
            if Psi_dtau == 0:
                Psi_tau = copy.deepcopy(Psi)
                Psi_dtau = copy.deepcopy(Psi_tau)
                new_psi = desWeight(Vi, Vref, Psi_dtau)
            else:
                new_psi = desWeight(Vi, Vref, Psi)
            prop -= 1.

        Vi = potential(new_psi)

        Vref = E_ref_calc(Vi, new_psi)

        Eref = np.append(Eref, Vref)

        if i >= (time_total - 1. - float(propagation)) and prop > 0:  # start of descendant weighting
            DW = True
        elif i >= (time_total - 1. - float(propagation)) and prop == 0:  # end of descendant weighting
            d_values = descendants(new_psi)
            Psi_tau.d += d_values
    return Eref


Eref = run(100)
print(np.mean(Eref[400:])*har2wave)
plt.figure()
plt.plot(Eref[400:]*har2wave)
plt.savefig('Non_imp_samp_morse.png')

