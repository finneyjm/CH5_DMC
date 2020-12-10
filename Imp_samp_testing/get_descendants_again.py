import numpy as np

# DMC parameters
dtau = 1
N_0 = 10000
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
omega = 3600/har2wave
mw = m_red * omega
A = np.sqrt(omega**2 * m_red/(2*De))


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_loc, initial_shift):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = initial_loc
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.shift = initial_shift

# function that plugs in the coordinates of the walkers and gets back the values of the trial wavefunction
def psi_t(coords, shift, DW):
    coords = coords - shift
    if DW:
        return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))
    else:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * coords ** 2)) * (2 * mw) ** (1 / 2) * coords

# Calculation of the drift term
def drift(coords, shift, DW):
    coords = coords - shift
    if DW:
        return -mw*coords*2
    else:
        return 2*(1-mw*coords**2)/coords



# Calculates the second derivative of the trial wavefunction for the kinetic energy of the local energy
def sec_dir(coords, shift, DW):
    coords = coords - shift
    if DW:
        return mw**2*coords**2 - mw
    else:
        return mw*(mw*coords**2 - 3)

# Metropolis step to determine the ratio of Green's functions
def metropolis(x, y, Fqx, Fqy, shift, DW):
    psi_x = psi_t(x, shift, DW)
    psi_y = psi_t(y, shift, DW)
    pre_factor = (psi_y/psi_x)**2
    a = pre_factor*np.exp(1./2.*(Fqx + Fqy)*(sigmaOH**2/4.*(Fqx-Fqy) - (y-x)))
    remove = np.argwhere(psi_y*psi_x < 0)
    a[remove] = 0.
    return a


def Kinetic(Psi, Fqx, DW):
    randomwalk = np.random.normal(0.0, sigmaOH, N_0)
    Drift = sigmaOH**2/2.*Fqx
    y = Psi.coords + randomwalk + Drift
    Fqy = drift(y, Psi.shift, DW)
    a = metropolis(Psi.coords, y, Fqx, Fqy, Psi.shift, DW)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    acceptance = float(len(accept) / len(Psi.coords))
    if acceptance <= 0.95:
        print(f'acceptance = {acceptance}')
    return Psi, Fqy, acceptance


def potential(Psi):
    Psi.V = De*(1. - np.exp(-A*Psi.coords))**2  # Morse potential
    # Psi.V = 0.5*m_red*omega**2*Psi.coords**2  # Harmonic oscillator
    return Psi


# Calculates the local energy of the trial wavefunction
def E_loc(Psi, DW):
    kin = -1. / (2. * m_red) * sec_dir(Psi.coords, Psi.shift, DW)
    pot = Psi.V
    Psi.El = kin + pot
    return Psi


# Calculate Eref from the local energy and the weights of the walkers
def E_ref_calc(Psi, alpha):
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/len(Psi.coords))
    return E_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vref, Psi, teff):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Vref) * teff)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 0.01
    death = np.argwhere(Psi.weights < threshold)
    for i in death:  # iterate over the list of dead walkers
        ind = np.argmax(Psi.weights)  # find the walker with with most weight
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


def run(propagation, initial_loc, initial_shift, weights, DW):
    psi = Walkers(N_0, initial_loc, initial_shift)
    psi.weights = weights
    Fqx = drift(psi.coords, psi.shift, DW)
    tau = np.zeros(propagation+1)
    Psi = potential(psi)
    Psi = E_loc(Psi, DW)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi, alpha)
    Eref_array = np.append(Eref_array, Eref)
    psi = Weighting(Eref, Psi, dtau)

    for i in range(int(propagation)):
        if (i+1) % 50 == 0:
            print(i)

        Psi, Fqx, accept = Kinetic(psi, Fqx, DW)
        teff = accept * dtau
        tau[i+1] = teff + tau[i]
        if accept <= 0.95:
            print(np.max(Psi.V)*har2wave)
            ind = np.argmax(Psi.V)
            print(Psi.coords[ind])
        Psi = potential(Psi)
        Psi = E_loc(Psi, DW)

        Psi = Weighting(Eref, Psi, teff)

        Eref = E_ref_calc(Psi, alpha)

        Eref_array = np.append(Eref_array, Eref)

    d_values = descendants(Psi)
    return d_values

wvfn = np.load('getting_coords.npy')