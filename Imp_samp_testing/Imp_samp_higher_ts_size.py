import numpy as np
import copy

# DMC parameters
dtau = 10.
N_0 = 1000
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
De = 0.1896
# De = 0.02
sigmaOH = np.sqrt(dtau/m_red)
omega = 3600./har2wave
mw = m_red * omega
A = np.sqrt(omega**2 * m_red/(2*De))

# Loads the wavefunction from the DVR for interpolation
# Psi_t = np.load('Harmonic_oscillator/Ground_state_wavefunction_HO.npy')

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
    return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))


# Calculation of the drift term
def drift(coords):
    return -mw*coords*2


# Calculates the second derivative of the trial wavefunction for the kinetic energy of the local energy
def sec_dir(coords):
    # return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))*(mw**2*coords**2-mw)
    return mw**2*coords**2 - mw


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
    # nah = np.argwhere(a <= check)
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    acceptance = float(len(accept) / len(Psi.coords))
    if acceptance <= 0.95:
        print(f'acceptance = {acceptance}')
    return Psi, Fqy, acceptance


def potential(Psi):
    Psi.V = De*(1. - np.exp(-A*Psi.coords))**2  # Morse potential
    return Psi


# Calculates the local energy of the trial wavefunction
def E_loc(Psi):
    # psi = psi_t(Psi.coords)
    # kin = -1./(2.*m_red)*sec_dir(Psi.coords)/psi
    kin = -1. / (2. * m_red) * sec_dir(Psi.coords)
    pot = Psi.V
    Psi.El = kin + pot
    return Psi


# Calculate Eref from the local energy and the weights of the walkers
def E_ref_calc(Psi, alpha):
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/len(Psi.coords))
    return E_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vref, Psi, teff, DW):
    teff = 9
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Vref) * teff)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 0.01
    death = np.argwhere(Psi.weights < threshold)
    # if len(death) >= 1:
    #         print('%s walkers dead' %len(death))
    for i in death:  # iterate over the list of dead walkers
        ind = np.argmax(Psi.weights)  # find the walker with with most weight
        if DW is True:
            Biggo_num = int(Psi.walkers[ind])
            Psi.walkers[i[0]] = Biggo_num
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_El = float(Psi.El[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
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
    Psi, Fqx, accept = Kinetic(psi, Fqx)
    teff = accept*dtau
    # teff = dtau
    Psi = potential(Psi)
    Psi = E_loc(Psi)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi, alpha)
    Eref_array = np.append(Eref_array, Eref)
    DW = False  # a parameter that will implement descendant weighting when True
    new_psi = Weighting(Eref, Psi, teff, DW)

    # initial parameters before running the calculation
    Psi_dtau = 0
    for i in range(int(time_total)):
        if (i+1) % 500 == 0:
            print(i)
        if DW is False:
            prop = float(propagation)

        Psi, Fqx, accept = Kinetic(psi, Fqx)
        teff = accept * dtau
        if accept <= 0.95:
            print(np.max(Psi.V)*har2wave)
            ind = np.argmax(Psi.V)
            print(Psi.coords[ind])
        # teff = dtau
        Psi = potential(Psi)
        Psi = E_loc(Psi)

        if DW is False:
            new_psi = Weighting(Eref, Psi, teff, DW)
        elif DW is True:
            if Psi_dtau == 0:
                Psi_tau = copy.deepcopy(Psi)
                Psi_dtau = copy.deepcopy(Psi_tau)
                new_psi = Weighting(Eref, Psi_dtau, teff, DW)
            else:
                new_psi = Weighting(Eref, Psi, teff, DW)
            prop -= 1.

        Eref = E_ref_calc(new_psi, alpha)

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
    print(np.mean(Eref_array[5000:])*har2wave)
    # np.save('Imp_samp_morse_energy', Eref_array)
    # np.save('Imp_samp_morse_Psi', wvfn)
    return Eref_array

b = np.zeros(5)
for i in range(5):
    e = run(100)
    b[i] = np.mean(e[5000:])*har2wave
print(f'Energy = {np.mean(b)}  {np.std(b)}')