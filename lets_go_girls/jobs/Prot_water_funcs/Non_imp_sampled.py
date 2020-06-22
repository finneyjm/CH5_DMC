import copy
import numpy as np
import os, sys
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from ProtWaterPES import Potential

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
        half = int(walkers / 2)
        self.coords = np.array([coords_initial] * walkers) * 1.05
        if len(atoms) == 13:
            self.coords[:half, :4, 2] *= -1
        elif len(atoms) == 4:
            self.coords[:, 3, 2] *= -1
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
def V_ref_calc(Psi, dtau, weight='continuous'):
    alpha = 1./(2.*dtau)
    P0 = sum(Psi.weights_i)
    if weight == 'continuous':
        P = sum(Psi.weights)
        V_ref = sum(Psi.weights * Psi.V) / P - alpha * (sum((Psi.weights - Psi.weights_i)) / P0)
    elif weight == 'discrete':
        P = len(Psi.coords)
        V_ref = np.mean(Psi.V) - alpha*(P-P0)/P0
    return V_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vref, Psi, DW, dtau, threshold, max_thresh):
    Psi.weights = Psi.weights * np.nan_to_num(np.exp(-(Psi.V - Vref) * dtau))
    # Conditions to prevent one walker from obtaining all the weight
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

    death = np.argwhere(Psi.weights > max_thresh)
    for i in death:
        ind = np.argmin(Psi.weights)
        if DW is True:
            Biggo_num = float(Psi.walkers[i[0]])
            Psi.walkers[ind] = Biggo_num
        Biggo_weight = float(Psi.weights[i[0]])
        Biggo_pos = np.array(Psi.coords[i[0]])
        Biggo_pot = float(Psi.V[i[0]])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[ind] = Biggo_pos
        Psi.V[ind] = Biggo_pot
    return Psi


def Discrete_weighting(Vref, Psi, DW, dtau):
    probs = np.nan_to_num(np.exp(-(Psi.V - Vref) * dtau))
    check = np.random.random(len(Psi.coords))
    death = np.logical_or((1-probs) < check, Psi.V < Vref)
    Psi.coords = Psi.coords[death]
    Psi.V = Psi.V[death]
    check = check[death]
    probs = probs[death]
    if DW:
        Psi.walkers = Psi.walkers[death]
    birth = np.logical_and((probs-1) > check, Psi.V < Vref)
    Psi.coords = np.concatenate((Psi.coords, Psi.coords[birth]))
    Psi.V = np.concatenate((Psi.V, Psi.V[birth]))
    if DW:
        Psi.walkers = np.concatenate((Psi.walkers, Psi.walkers[birth]))
    else:
        Psi.walkers = np.arange(0, len(Psi.coords))
    return Psi


# Calculates the descendant weight for the walkers before descendant weighting
def descendants(Psi, weighting, N_0=None):
    if weighting == 'discrete':
        d = np.bincount(Psi.walkers)
    else:
        N_0 = len(Psi.coords)
        d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < N_0:
        d = np.append(d, 0.)
    Psi.walkers = np.arange(0, len(Psi.walkers))
    return d


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def Parr_Potential(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


def simulation_time(psi, sigmaCH, time_steps, dtau, equilibration, wait_time,
                    propagation, multicore, threshold, max_thresh, weighting='continuous', output='before_output_patch'):
    DW = False
    time = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    num_o_collections = int((time_steps - equilibration) / (propagation + wait_time)) + 1
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))
    if weighting == 'discrete':
        buffer = int(len(psi.coords))
        coords = np.hstack((coords, np.zeros((num_o_collections, buffer, psi.coords.shape[1], psi.coords.shape[2]))))
        weights = np.hstack((weights, np.zeros((num_o_collections, buffer))))
        des = np.hstack((des, np.zeros((num_o_collections, buffer))))

    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Vref_array = np.zeros(time_steps)
    a = 0
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

            V_ref = V_ref_calc(psi, dtau, weighting)

        psi = Kinetic(psi, sigmaCH)

        if multicore is True:
            psi = Parr_Potential(psi)
        else:
            psi.V = get_pot(psi.coords)

        if weighting == 'discrete':
            psi = Discrete_weighting(V_ref, psi, DW, dtau)
        else:
            psi = Weighting(V_ref, psi, DW, dtau, threshold, max_thresh)
        V_ref = V_ref_calc(psi, dtau, weighting)

        Vref_array[i] = V_ref
        time[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)

        if (i+1) % 500 == 0:
            print(f'time step {i+1} Energy = {Vref_array[i]}')
            if i > 500:
                print(f'Average E ref = {np.mean(Vref_array[500:])}')
            if a == 0:
                np.savez(f'wvfn_1_{output}', coords = psi.coords, weights=psi.weights, eref=V_ref)
                a = 1
            elif a == 1:
                np.savez(f'wvfn_2_{output}', coords=psi.coords, weights=psi.weights, eref=V_ref)
                a = 0

        if i >= int(equilibration)-1 and wait <= 0. < prop:
            DW = True
            wait = float(wait_time)
            Psi_tau = copy.deepcopy(psi)
            if weighting == 'discrete':
                coords[num, :len(Psi_tau.coords)] = Psi_tau.coords
            else:
                coords[num] = Psi_tau.coords
                weights[num] = Psi_tau.weights
        elif prop == 0:
            DW = False
            if weighting == 'discrete':
                des[num, :len(Psi_tau.coords)] = descendants(psi, weighting, len(Psi_tau.coords))
            else:
                des[num] = descendants(psi, weighting)
            num += 1

    return coords, weights, time, Vref_array, sum_weights, des


pool = mp.Pool(mp.cpu_count()-1)
