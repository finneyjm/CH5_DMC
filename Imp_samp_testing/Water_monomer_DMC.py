import copy
from scipy import interpolate
import numpy as np
import Water_monomer_pot_fns as wm

# DMC parameters
dtau = 1.
time_steps = 20000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_OH = (m_H*m_O)/(m_H+m_O)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# Values for Simulation
sigmaH = np.sqrt(dtau/m_H)
sigmaO = np.sqrt(dtau/m_O)

coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                           [0.957840000000000, 0.000000000000000, 0.000000000000000],
                           [-0.23995350000000, 0.927297000000000, 0.000000000000000]])*ang2bohr*1.05


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.arange(0, N_0)
        self.coords = np.array([coords_initial]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.


# Random walk of all the walkers
def Kinetic(Psi):
    randomwalkO = np.random.normal(0.0, sigmaO, size=(N_0, 3))
    randomwalkH = np.random.normal(0.0, sigmaH, size=(N_0, 2, 3))
    Psi.coords[:, 0, :] += randomwalkO
    Psi.coords[:, 1:3, :] += randomwalkH
    return Psi


def Potential(Psi):
    V = wm.PatrickShinglePotential(Psi.coords, 0)
    Psi.V = np.array(V)
    return Psi


# Calculate V_ref for the weighting calculation and to determine the ground state energy
def V_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    V_ref = sum(Psi.weights*Psi.V)/P - alpha*(sum((Psi.weights-Psi.weights_i))/P0)
    return V_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vref, Psi, DW):
    Psi.weights = Psi.weights * np.exp(-(Psi.V - Vref) * dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1./float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        if DW is True:
            Biggo_num = float(Psi.walkers[ind])
            Psi.walkers[i[0]] = Biggo_num
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
    return Psi


# Calculates the descendant weight for the walkers before descendant weighting
def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < N_0:
        d = np.append(d, 0.)
    return d


def run(propagation, test_number):
    DW = False  # a parameter that will implement descendant weighting when True
    psi = Walkers(N_0)
    Psi = Kinetic(psi)
    Psi = Potential(Psi)
    Eref = np.array([])
    time = np.array([])
    weights = np.array([])
    Vref = V_ref_calc(Psi)
    Eref = np.append(Eref, Vref)
    new_psi = Weighting(Vref, Psi, DW)
    time = np.append(time, 1)
    weights = np.append(weights, np.sum(new_psi.weights))

    # initial parameters before running the calculation

    Psi_tau = 0  #
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)
        Psi = Kinetic(new_psi)
        Psi = Potential(Psi)

        if DW is False:
            prop = float(propagation)

        elif DW is True:
            prop -= 1.
            if Psi_tau == 0:
                Psi_tau = copy.deepcopy(Psi)
        new_psi = Weighting(Vref, Psi, DW)

        Vref = V_ref_calc(new_psi)
        Eref = np.append(Eref, Vref)
        time = np.append(time, 2 + i)
        weights = np.append(weights, np.sum(new_psi.weights))

        if i >= (time_steps - 1. - float(propagation)) and prop > 0:  # start of descendant weighting
            DW = True
        elif i >= (time_steps - 1. - float(propagation)) and prop == 0:  # end of descendant weighting
            d_values = descendants(new_psi)
    np.save(f'DMC_water_coords_{N_0}_walkers_{test_number}', Psi_tau.coords)
    np.save(f'DMC_water_weights_{N_0}_walkers_{test_number}', np.vstack((Psi_tau.weights, d_values)))
    np.save(f'DMC_water_energy_{N_0}_walkers_{test_number}', np.vstack((time, Eref, weights)))
    return Eref


# tests = [100, 200, 500, 1000, 2000, 5000, 10000]
# for j in range(5):
#     for i in range(7):
#         N_0 = tests[i]
#         run(250, j + 6)
#         print(f'{tests[i]} Walker Test {j + 1} is done!')
for i in range(10):
    N_0 = 20000
    run(250, i+1)
    print(f'{N_0} Walker Test {i+1} is done!')



