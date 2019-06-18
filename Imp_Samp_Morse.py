import numpy as np
import copy
from scipy import interpolate
import matplotlib.pyplot as plt

# DMC parameters
dtau = 1.
N_0 = 500
time_total = 20000.
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
sigmaOH = np.sqrt(dtau/m_red)
omega = 3600./har2wave
mw = m_red * omega
A = np.sqrt(omega**2 * m_red/(2*De))

# Loads the wavefunction from the DVR for interpolation
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
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


# function that plugs in the coordinates of the walkers and gets back the values of the trial wavefunction
def psi_t(coords):
    # wvfn = Psi_t
    # x = wvfn[0, :]
    # y = wvfn[1, :]
    # int = interpolate.splrep(x, y, s=0)
    # return interpolate.splev(coords, int, der=0)
    return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))


# Calculation of the drift term
def drift(coords):
    # psi = psi_t(coords)
    # wvfn = Psi_t
    # x = wvfn[0, :]
    # y = wvfn[1, :]
    # int = interpolate.splrep(x, y, s=0)
    # return interpolate.splev(coords, int, der=1)/psi
    # return ((mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))*-mw*coords)/psi
    return -2.*mw*coords


# Calculates the second derivative of the trial wavefunction for the kinetic energy of the local energy
def sec_dir(coords):
    # wvfn = Psi_t
    # x = wvfn[0, :]
    # y = wvfn[1, :]
    # int = interpolate.splrep(x, y, s=0)
    # return interpolate.splev(coords, int, der=2)
    return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))*(mw**2*coords**2-mw)


# Metropolis step to determine the ratio of Green's functions
def metropolis(x, y, Fqx, Fqy):
    psi_x = psi_t(x)
    psi_y = psi_t(y)
    pre_factor = (psi_y/psi_x)**2
    M = pre_factor*np.exp(1./2.*(Fqx + Fqy)*(sigmaOH**2/4.*(Fqx-Fqy) - (y-x)))
    return M


def Kinetic(Psi, Fqx):
    randomwalk = np.random.normal(0.0, sigmaOH, N_0)
    Drift = sigmaOH**2/2.*Fqx
    y = Psi.coords + randomwalk + Drift
    Fqy = drift(y)
    a = metropolis(Psi.coords, y, Fqx, Fqy)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    return Psi, Fqy


def potential(Psi):
    Psi.V = De*(1. - np.exp(-A*Psi.coords))**2  # Morse potential
    # return 1./2.*m_red*omega**2*Psi.coords**2  # Harmonic potential
    return Psi


# Calculates the local energy of the trial wavefunction
def E_loc(Psi):
    psi = psi_t(Psi.coords)
    kin = -1./(2*m_red)*sec_dir(Psi.coords)/psi
    pot = Psi.V
    Psi.El = kin + pot
    return Psi


# Calculate Eref from the local energy and the weights of the walkers
def E_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/P0)
    return E_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vref, Psi):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Vref) * dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:  # iterate over the list of dead walkers
        ind = np.argmax(Psi.weights)  # find the walker with with most weight
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_El = float(Psi.El[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.pot[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_El
    return Psi


# Descendant weighting where the descendants of the walkers that replace those that die are kept track of
def desWeight(El, Vref, Psi):
    Psi.weights = Psi.weights*np.exp(-(El-Vref)*dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_num = float(Psi.walkers[ind])  # make sure to keep track of the walker that is donating its weight
        Biggo_pot = float(Psi.V[ind])
        Biggo_El = float(Psi.El[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.walkers[i[0]] = Biggo_num
        Psi.coords[i[0]] = Biggo_pos
        Psi.pot[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_El
    return Psi


# Calculates the descendant weight for the walkers after descendant weighting
def descendants(Psi):
    for i in range(N_0):
        Psi.d[i] = np.sum(Psi.weights[Psi.walkers == i])
    return Psi.d


def run(propagation):
    psi = Walkers(N_0)
    Fqx = drift(psi.coords)
    Psi, Fqx = Kinetic(psi, Fqx)
    Psi = potential(Psi)
    Psi = E_loc(Psi)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi)

    # initial parameters before running the calculation
    DW = False  # a parameter that will implement descendant weighting when True
    Psi_dtau = 0
    for i in range(int(time_total)):
        if DW is False:
            prop = float(propagation)

        Psi, Fqx = Kinetic(new_psi, Fqx)
        Psi = potential(Psi)
        Psi = E_loc(Psi)

        if DW is False:
            new_psi = Weighting(Eref, Psi)
        elif DW is True:
            if Psi_dtau == 0:
                Psi_tau = copy.deepcopy(Psi)
                Psi_dtau = copy.deepcopy(Psi_tau)
                new_psi = desWeight(Eref, Psi_dtau)
            else:
                new_psi = desWeight(Eref, Psi)
            prop -= 1.

        Eref = E_ref_calc(new_psi)

        Eref_array = np.append(Eref_array, Eref)

        if i >= (time_total - 1. - float(propagation)) and prop > 0:  # start of descendant weighting
            DW = True
        elif i >= (time_total - 1. - float(propagation)) and prop == 0.:  # end of descendant weighting
            d_values = descendants(new_psi)
            Psi_tau.d += d_values

    wvfn = np.zeros((3, N_0))
    wvfn[0, :] += Psi_tau.coords
    wvfn[1, :] += Psi_tau.weights
    wvfn[2, :] += Psi_tau.d
    np.save('Imp_samp_morse_energy', Eref_array)
    np.save('Imp_samp_morse_Psi', wvfn)
    return


run(100)

