import copy
import numpy as np

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
m_OD = (m_O*m_D)/(m_D+m_O)
m_OH = (m_O*m_H)/(m_H+m_O)
har2wave = 219474.6


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, atoms, coords_initial):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial]*walkers)*1.01
        self.weights = np.zeros(walkers) + 1.
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.atoms = atoms


# Random walk of all the walkers
def Kinetic(Psi, sigma):
    N = len(Psi.coords)
    randomwalk = np.random.normal(0.0, sigma, size=(N, sigma.shape[0], sigma.shape[1]))
    Psi.coords += randomwalk
    return Psi


# Calculate V_ref for the weighting calculation and to determine the ground state energy
def V_ref_calc(Psi, dtau):
    alpha = 1./(2.*dtau)
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    V_ref = sum(Psi.weights * Psi.V) / P - alpha * (sum((Psi.weights - Psi.weights_i)) / P0)
    return V_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vref, Psi, DW, dtau):
    N_0 = len(Psi.coords)
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
        Biggo_pot = float(Psi.V[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
    return Psi


# Calculates the descendant weight for the walkers before descendant weighting
def descendants(Psi):
    N_0 = len(Psi.coords)
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < N_0:
        d = np.append(d, 0.)
    return d


def simulation_time(psi, sigmaCH, time_steps, dtau, equilibration, wait_time, propagation, multicore):
    DW = False
    num_o_collections = int((time_steps - equilibration) / wait_time) + 1
    time = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))
    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Vref_array = np.zeros(time_steps)

    for i in range(int(time_steps)):
        if DW is False:
            prop = float(propagation)
            wait -= 1.
        else:
            prop -= 1.
        if i == 0:
            if multicore is True:
                psi = Parr_Potential(psi)
            else:
                psi.V = get_pot(psi.coords)

            V_ref = V_ref_calc(psi, dtau)

        psi = Kinetic(psi, sigmaCH)

        if multicore is True:
            psi = Parr_Potential(psi)
        else:
            psi.V = get_pot(psi.coords)

        psi = Weighting(V_ref, psi, DW, dtau)
        V_ref = V_ref_calc(psi, dtau)

        Vref_array[i] = V_ref
        time[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)
        if i >= int(equilibration)-1 and wait <= 0. < prop:
            DW = True
            wait = float(wait_time)
            Psi_tau = copy.deepcopy(psi)
            coords[num] = Psi_tau.coords
            weights[num] = Psi_tau.weights
        elif prop == 0:
            DW = False
            des[num] = descendants(psi)
            num += 1

    return coords, weights, time, Vref_array, sum_weights, des







