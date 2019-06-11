import numpy as np
import copy
import CH5pot
import Timing_p3 as tm

# DMC parameters
dtau = 5.
N_0 = 500
time_total = 1000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
har2wave = 219474.6

# Values for Simulation
sigmaH = np.sqrt(dtau/m_H)
sigmaC = np.sqrt(dtau/m_C)
# Starting orientation of walkers
coords_inital = ([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                  [-0.8247121421923925, -0.6295306113384560, 1.775332267901544],
                  [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                  [1.786540362044548, -1.386051328559878, 0.000000000000000],
                  [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                  [-0.8247121421923925, -0.6295306113384560, -1.775332267901544]])


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.array([coords_inital]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.


psi = Walkers(N_0)


# Random walk of all the walkers
def Kinetic(Psi):
    randomwalkC = np.random.normal(0.0, sigmaC, size=(N_0, 3))
    randomwalkH = np.random.normal(0.0, sigmaH, size=(N_0, 5, 3))
    Psi.coords[:, 0, :] += randomwalkC
    Psi.coords[:, 1:6, :] += randomwalkH
    return Psi


# Using the potential call to calculate the potential of all the walkers
def Potential(Psi):
    V = CH5pot.mycalcpot(Psi.coords, N_0)
    Vi = np.array(V)
    return Vi


# Calculate V_ref for the weighting calculation and to determine the ground state energy
def V_ref_calc(Vi,Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    V_ref = sum(Psi.weights*Vi)/P - alpha*(sum((Psi.weights-Psi.weights_i))/P0)
    return V_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vi, Vref, Psi):
    Psi.weights = Psi.weights * np.exp(-(Vi - Vref) * dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
    return Psi


# Descendant weighting where the descendants of the walkers that replace those that die are kept track of
def desWeight(Vi, Vref, Psi):
    Psi.weights = Psi.weights*np.exp(-(Vi-Vref)*dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_num = float(Psi.walkers[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.walkers[i] = Biggo_num
        Psi.coords[i[0]] = Biggo_pos
    return Psi


# Calculates the descendant weight for the walkers before descendant weighting
def descendants(Psi):
    for i in range(N_0):
        Psi.d[i] = np.sum(Psi.weights[Psi.walkers == i])
    return Psi.d


def run(propagation):
    Psi = Kinetic(psi)
    Vi = Potential(Psi)
    Eref = np.array([])
    Vref = V_ref_calc(Vi, Psi)
    Eref = np.append(Eref, Vref)
    new_psi = Weighting(Vi, Vref, Psi)

    # initial parameters before running the calculation
    DW = False  # a parameter that will implement descendant weighting when True
    Psi_dtau = 0  #
    for i in range(int(time_total)):
        if DW is False:
            prop = float(propagation)

        Psi = Kinetic(new_psi)
        Vref = V_ref_calc(Vi, Psi)
        Vi = Potential(Psi)

        Eref = np.append(Eref, Vref)

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

        if i >= (time_total - 1. - float(propagation)) and prop > 0:  # start of descendant weighting
            DW = True
        elif i >= (time_total - 1. - float(propagation)) and prop == 0:  # end of descendant weighting
            d_values = descendants(new_psi)
            Psi_tau.d += d_values
    E0 = np.mean(Eref[50:])
    np.save("DMC_CH5_Energy", Eref)
    print(d_values)
    return E0


E0, E0_time = tm.time_me(run, 50)
tm.print_time_list(run, E0_time)
