import numpy as np
import copy
from scipy import interpolate

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
omega = 3600./har2wave

# wexe = 150./har2wave
wexe = 75/har2wave
De = omega**2/4/wexe
sigmaOH = np.sqrt(dtau/m_red)
omega = 3600/har2wave
mw = m_red * 3600/har2wave
A = np.sqrt(omega**2 * m_red/(2*De))


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_loc, initial_shift, initial_weights):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = initial_loc
        self.weights = initial_weights
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.shift = initial_shift


# function that plugs in the coordinates of the walkers and gets back the values of the trial wavefunction
def psi_t(coords, shift, DW):
    coords = coords - shift
    if DW is None:
        return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))
    else:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * coords ** 2)) * (2 * mw) ** (1 / 2) * coords


# Calculation of the drift term
def drift(coords, shift, DW):
    coords = coords - shift
    if DW is None:
        return -mw*coords*2
    else:
        return 2*(1-mw*coords**2)/coords


# Calculates the second derivative of the trial wavefunction for the kinetic energy of the local energy
def sec_dir(coords, shift, DW):
    coords = coords - shift
    if DW is None:
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


def run(N_0, time_steps, propagation, equilibration, wait_time, excite, initial_struct, initial_shifts, shift_rate, initial_weights):
    DW = False
    psi = Walkers(N_0, initial_struct, initial_shifts, initial_weights)
    Fqx = drift(psi.coords, psi.shift, excite)
    num_o_collections = int((time_steps - equilibration) / (propagation + wait_time)) + 1
    time = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    accept = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))

    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Eref_array = np.zeros(time_steps)

    shift = np.zeros((time_steps + 1))
    shift[0] = psi.shift
    shift_rate = np.array(shift_rate)
    psi.shift = np.array(psi.shift)
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)

        if DW is False:
            prop = float(propagation)
            wait -= 1.
        else:
            prop -= 1.

        if i == 0:
            psi = potential(psi)
            psi = E_loc(psi, excite)
            Eref = E_ref_calc(psi, alpha)

        psi, Fqx, acceptance = Kinetic(psi, Fqx, excite)
        shift[i + 1] = psi.shift
        psi = potential(psi)
        psi = E_loc(psi, excite)

        psi = Weighting(Eref, psi, dtau)
        Eref = E_ref_calc(psi, alpha)

        Eref_array[i] = Eref
        time[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)
        accept[i] = acceptance

        if i >= 5000:
            psi.shift = psi.shift + shift_rate

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


wvfn = np.load('getting_coords.npy')
wvfn_coords_right = np.abs(wvfn[0] + 0.039)
wvfn_coords_left = np.abs(wvfn[0]*-1)

for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        10000, 20000, 250, 500, 500, None, wvfn[0], 0.039, 0, wvfn[1]
    )
    np.savez(f'ground_state_morse_{i+5}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)

for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        10000, 20000, 250, 500, 500, 'excite', wvfn_coords_left, 0.039, 0, wvfn[1]
    )
    np.savez(f'excite_state_morse_left_{i}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)

for i in range(5):
    coords, weights, time, Eref_array, sum_weights, accept, des = run(
        10000, 20000, 250, 500, 500, 'excite', wvfn_coords_right, 0.039, 0, wvfn[1]
    )
    np.savez(f'excite_state_morse_right_{i}', coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, d=des)

# x = np.linspace(-1, 1, 5000)
# psi = Walkers(1000, x, 0.039)
# psi = potential(psi)
# psi = E_loc(psi, False)
# f = drift(x, 0.039, False)
#
# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots()
# ax1.plot(x, psi.El*har2wave, color='blue')
# ax1.set_ylabel(r'Energy cm$^{-1}$', color='blue', fontsize=16)
# ax1.set_xlabel(r'x Bohr', fontsize=16)
# ax1.tick_params(axis='x', labelsize=12)
# ax1.tick_params(axis='y', labelcolor='blue', labelsize=12)
#
# ax2 = ax1.twinx()
#
# ax2.plot(x, f, color='red')
# ax2.set_ylabel(r'Drift Hartree*Bohr$^2$', color='red', fontsize=16)
# ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
# ax2.set_ylim(-2000, 2000)
#
# # plt.tight_layout()
# plt.show()

# ground_coords = np.zeros((1, 5, 10000))
# ground_weights = np.zeros((1, 5, 10000))
# for i in range(5):
#     coords, weights = run(20000, wvfn[0], 0.039, wvfn[1], True)
#     ground_coords[:, i] = coords
#     ground_weights[:, i] = weights

# excite_coords = np.zeros((1, 5, 10000))
# excite_weights = np.zeros((1, 5, 10000))
# for i in range(5):
#     coords, weights = run(20000, wvfn[0], 0.039, wvfn[1], False)
#     excite_coords[:, i] = coords
#     excite_weights[:, i] = weights
#
# # np.save('ground_state_wvfn', np.vstack((ground_weights, ground_coords)))
# np.save('excited_state_wvfn', np.vstack((excite_weights, excite_coords)))
