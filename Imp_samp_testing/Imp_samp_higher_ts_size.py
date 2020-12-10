import numpy as np
import copy
from scipy import interpolate

# DMC parameters
dtau = 1
N_0 = 10000
time_total = 20000
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6

# parameters for the potential and for the analytic wavefuntion
omega = 3600./har2wave

wexe = 500./har2wave
De = omega**2/4/wexe
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

    def __init__(self, walkers, initial_loc, initial_shift):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        half = int(walkers/2)
        self.coords = np.zeros(walkers)+initial_loc
        # self.coords[:half] -= 0.8
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        # self.shift = -0.004
        # self.shift = 0.0400374
        self.shift = initial_shift

# function that plugs in the coordinates of the walkers and gets back the values of the trial wavefunction
def psi_t(coords, shift, DW):
    coords = coords - shift
    # return interpolate.splev(coords, interp, der=0)
    if DW is False:
        return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))
    else:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * coords ** 2)) * (2 * mw) ** (1 / 2) * coords
    # return np.ones(len(coords))

# Calculation of the drift term
def drift(coords, shift, DW):
    # psi = psi_t(coords, shift)
    coords = coords - shift

    # first = interpolate.splev(coords, interp, der=1)
    # dur = 2*first/psi
    if DW is False:
        return -mw*coords*2
    else:
        return 2*(1-mw*coords**2)/coords
    # replace = np.argwhere(np.abs(dur) > (1/sigmaOH**2))
    # dur[replace] = 1/sigmaOH**2*np.sign(dur[replace])
    # return dur
    # return np.zeros(len(coords))



# Calculates the second derivative of the trial wavefunction for the kinetic energy of the local energy
def sec_dir(coords, shift, DW):
    # psi = psi_t(coords, shift)
    coords = coords - shift
    # sec = interpolate.splev(coords, interp, der=2)
    # return sec/psi
    #

    if DW is False:
        return mw**2*coords**2 - mw
    else:
        return mw*(mw*coords**2 - 3)
    # return np.zeros(len(coords))

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
def Weighting(Vref, Psi, teff, DW):
    # teff = 10
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Vref) * teff)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 0.01
    death = np.argwhere(Psi.weights < threshold)
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


def run(propagation, initial_loc, initial_shift, shift_rate, addendum):
    psi = Walkers(N_0, initial_loc, initial_shift)
    DW = False  # a parameter that will implement descendant weighting when True
    Fqx = drift(psi.coords, psi.shift, DW)
    Psi, Fqx, accept = Kinetic(psi, Fqx, DW)
    tau = np.zeros(time_total+1)
    teff = accept*dtau
    tau[0] = teff
    Psi = potential(Psi)
    Psi = E_loc(Psi, DW)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi, alpha)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, teff, DW)

    # initial parameters before running the calculation
    Psi_dtau = 0
    shift = np.zeros(time_total+1)
    shift[0] = Psi.shift

    for i in range(int(time_total)):

        if (i+1) % 500 == 0:
            print(i)
        if DW is False:
            prop = float(propagation)

        Psi, Fqx, accept = Kinetic(psi, Fqx, DW)
        teff = accept * dtau
        tau[i+1] = teff + tau[i]
        if accept <= 0.95:
            print(np.max(Psi.V)*har2wave)
            ind = np.argmax(Psi.V)
            print(Psi.coords[ind])
        Psi = potential(Psi)
        Psi = E_loc(Psi, DW)
        shift[i+1] = Psi.shift

        if i >= 5000:
            Psi.shift = Psi.shift + shift_rate

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
            # Psi_tau.d += d_values

    wvfn = np.zeros((2, N_0))
    wvfn[0, :] += new_psi.coords
    wvfn[1, :] += new_psi.weights
    # wvfn[2, :] += Psi_tau.d
    print(np.mean(Eref_array[5000:])*har2wave)
    import matplotlib.pyplot as plt
    # amp, bin = np.histogram(Psi_tau.coords, weights=Psi_tau.weights, range=(-.4, 1.), bins=50)
    # bin = (bin[1:] + bin[:-1])/2
    # y = psi_t(bin, Psi.shift, DW)**2
    # ma = np.max(amp)
    # amp /= (ma/np.max(y))
    # plt.plot(bin, amp)
    # plt.plot(bin, y)
    # plt.show()
    # np.save(f'Imp_samp_morse_energy_{addendum}', np.vstack((Eref_array, shift)))
    np.save(f'{addendum}', wvfn)
    return Eref_array

# import matplotlib.pyplot as plt
# wvfn = np.load('testing_dw_wvfn.npy')
# print(np.sum(wvfn[2]))
# print(np.sum(wvfn[1]))
# print(np.sum(wvfn[2]**2))
# print(np.sum(wvfn[2]**2/wvfn[1]))
# print(np.sum(wvfn[2]*-wvfn[0])/(np.sqrt(np.sum(wvfn[1])*np.sum(wvfn[2]**2/wvfn[1]))))
# a = np.zeros(5)
# for i in range(5):
#     wvfn = np.load(f'testing_dw_{i+1}.npy')
#     a[i] = np.sum(wvfn[2]*(-wvfn[0]))/(np.sqrt(np.sum(wvfn[1])*np.sum(wvfn[2]**2/wvfn[1])))
#
# print(np.mean(a))
# print(np.std(a))



# for i in np.arange(0, 5):
e = run(0, 0, 0.039, 0.0000, f'getting_coords')
# e = run(0, 0.8, 0.15, -0.00001, 'right')

# np.save('testing_dw_wvfn', wvfn)
# #
# a = np.linspace(-1, 1, num=500)
# psi = Walkers(500)
# psi.coords = a
# # psi.shift = 0.04021
# psi = potential(psi)
# psi = E_loc(psi)
# x = psi_t(psi.coords, 0.04021)
# plt.plot(a, (x*5000))
# plt.plot(a, psi.El*har2wave)
# plt.xlim(-0.7, 0.7)
# plt.ylim(-10000, 20000)
# plt.xlabel('x')
# plt.ylabel(r'Energy cm$^{-1}$')
# plt.show()


def cub_fit(x, *params):
    a, b, c, d = params
    return a*x**3 + b*x**2 + c*x + d

import scipy.optimize


import matplotlib.pyplot as plt

e1 = np.load('Imp_samp_morse_energy_left.npy')
e2 = np.load('Imp_samp_morse_energy_right.npy')

node_pos = 0.039
# node_pos = 0.10055

params = [-0.28662516,  0.31322083, -0.13109515,  0.02856335]
fitted_params1, _ = scipy.optimize.curve_fit(cub_fit, e1[1, 5000:], e1[0, 5000:], p0=params)
params = [-0.3359762,   0.15442415,  0.06611351,  0.02092535]
fitted_params2, _ = scipy.optimize.curve_fit(cub_fit, e2[1, 5000:], e2[0, 5000:], p0=params)


plt.plot(e1[1, 5000:], e1[0, 5000:]*har2wave)
plt.plot(e1[1, 5000:], cub_fit(e1[1, 5000:], *fitted_params1)*har2wave, label='Left Side of Node')

plt.plot(e2[1, 5000:], e2[0, 5000:]*har2wave)
plt.plot(e2[1, 5000:], cub_fit(e2[1, 5000:], *fitted_params2)*har2wave, label='Right Side of Node')

plt.scatter(node_pos, 5225, s=100, color='black', label='Correct Node Position', zorder=3)

plt.ylim(3000, 7000)
# plt.xlim(0.18, 0.22)
plt.ylabel(r'Energy cm$^{-1}$', fontsize=14)
plt.xlabel(r'Node position (Bohr)', fontsize=14)
plt.tight_layout()

plt.legend()
plt.show()



# from scipy import interpolate
# # interp = interpolate.splrep(e[1], e[0], s=0)
# # x = np.linspace(100, (time_total-300)*dtau, int(time_total-400))
# # plt.plot(e[1], e[0]*har2wave, label='raw')
# # plt.plot(x, interpolate.splev(x, interp, der=0)*har2wave, label='interp')
# # plt.xlim(100*dtau, 8000*dtau)
# # plt.legend()
# # # plt.show()
# # y = interpolate.splev(x, interp, der=0)
# # print(np.mean(y[5000:9000]*har2wave))
# #
# e = np.load('Imp_samp_morse_energy.npy')
# # from scipy import interpolate
# interp = interpolate.splrep(e[1], e[0], s=0)
# x = np.linspace(100, (time_total-300)*dtau, int(time_total-400))
# y = interpolate.splev(x, interp, der=0)
# print(np.mean(y[5000:9000]*har2wave))
