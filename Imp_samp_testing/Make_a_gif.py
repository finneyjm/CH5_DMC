import numpy as np
import copy
from scipy import interpolate

# DMC parameters
dtau = 1
N_0 = 2000
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6

# parameters for the potential and for the analytic wavefuntion
De = 0.1896
# De = 0.0147
sigmaOH = np.sqrt(dtau/m_red)
# omega = 3600./har2wave
omega = 3600/har2wave
# mw = m_red * omega
mw = m_red * 3600/har2wave
# mw = 0
A = np.sqrt(omega**2 * m_red/(2*De))

trial_wvfn = np.load('Harmonic_oscillator/Anharmonic_trial_wvfn_150_wvnum.npy')
interp = interpolate.splrep(trial_wvfn[0], trial_wvfn[1], s=0)

# Loads the wavefunction from the DVR for interpolation
# Psi_t = np.load('Harmonic_oscillator/Ground_state_wavefunction_HO.npy')

# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_loc):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = initial_loc
        self.weights = np.zeros(walkers) + 1.
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)



def Kinetic(Psi):
    randomwalk = np.random.normal(0.0, sigmaOH, N_0)
    Psi.coords = Psi.coords + randomwalk
    return Psi

def potential(Psi):
    Psi.V = De*(1. - np.exp(-A*Psi.coords))**2  # Morse potential
    # Psi.V = 0.5*m_red*omega**2*Psi.coords**2  # Harmonic oscillator
    return Psi


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vref, Psi, teff):
    # teff = 10
    Psi.weights = Psi.weights * np.exp(-(Psi.V - Vref) * teff)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1/2000.
    death = np.argwhere(Psi.weights < threshold)
    for i in death:  # iterate over the list of dead walkers
        ind = np.argmax(Psi.weights)  # find the walker with with most weight
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
    return Psi


def run(time_total, weights, initial_loc):
    psi = Walkers(N_0, initial_loc)
    psi.weights = weights
    Psi = Kinetic(psi)
    Psi = potential(Psi)
    Vref = 1650./har2wave
    psi = Weighting(Vref, Psi, dtau)

    sum_weights = np.zeros(time_total+1)
    sum_weights[0] = np.sum(psi.weights)
    for i in range(int(time_total)):
        if i == 500:
            Vref = 1850./har2wave
        elif i == 1900:
            Vref = 1780./har2wave

        psi = Kinetic(psi)
        psi = potential(psi)
        psi = Weighting(Vref, psi, dtau)
        sum_weights[i+1] = np.sum(psi.weights)


    return sum_weights

# coords = np.linspace(-0.5, 0.5, 2000)
#
# a = (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (coords-0.039) ** 2))
#
# weights = run(5000, a, coords)
#
# np.save('weights', weights)

import matplotlib.pyplot as plt
weights = np.load('weights.npy')

for i in np.arange(0, 5000, 10):
    if i < 500:
        plt.plot(weights[0:i+1], color='purple', label=r'E$_{\rmref}$ < E$_0$')
    elif i < 1900:
        plt.plot(weights[0:i + 1], color='purple', label=r'E$_{\rmref}$ > E$_0$')
    else:
        plt.plot(weights[0:i + 1], color='purple', label=r'E$_{\rmref}$ = E$_0$')
    plt.xlabel(r'$\tau$', fontsize=15)
    plt.ylabel(r'Population', fontsize=15)
    plt.legend(loc='upper left', fontsize=18)
    plt.savefig(f'Harmonic_oscillator/figure_{i}.JPG')
    plt.close()


plt.plot(weights, color='purple', label=r'E$_{\rmref}$ = E$_0$')
plt.xlabel(r'$\tau$', fontsize=15)
plt.ylabel(r'Population', fontsize=15)
plt.legend(loc='upper left', fontsize=18)
plt.savefig(f'Harmonic_oscillator/figure_{5000}.JPG')
plt.show()






