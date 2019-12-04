import numpy as np
import copy
import CH5pot
# import Timing_p3 as tm
import matplotlib.pyplot as plt
import multiprocessing as mp

# DMC parameters
dtau = 1.
# N_0 = 10000
time_total = 20000.
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
coords_inital = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                  [-0.8247121421923925, -0.6295306113384560, 1.775332267901544],
                  [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                  [1.786540362044548, -1.386051328559878, 0.000000000000000],
                  [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                  [-0.8247121421923925, -0.6295306113384560, -1.775332267901544]])


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.arange(0, N_0)
        self.coords = np.array([coords_inital]*walkers)
        rand_idx = np.random.rand(N_0, 5).argsort(axis=1) + 1
        b = self.coords[np.arange(N_0)[:, None], rand_idx]
        self.coords[:, 1:6, :] = b
        self.weights = np.zeros(walkers) + 1.
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)


# Random walk of all the walkers
def Kinetic(Psi):
    randomwalkC = np.random.normal(0.0, sigmaC, size=(N_0, 3))
    randomwalkH = np.random.normal(0.0, sigmaH, size=(N_0, 5, 3))
    Psi.coords[:, 0, :] += randomwalkC
    Psi.coords[:, 1:6, :] += randomwalkH
    return Psi


# Function for the potential for the mp to use
def get_pot(coords):
    V = CH5pot.mycalcpot(coords, len(coords))
    return V


# Split up those coords to speed up dat potential
def Potential(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    # Psi.V = np.array(CH5pot.mycalcpot(Psi.coords, len(Psi.coords)))
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
    for i in range(int(time_total)):
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

        if i >= (time_total - 1. - float(propagation)) and prop > 0:  # start of descendant weighting
            DW = True
        elif i >= (time_total - 1. - float(propagation)) and prop == 0:  # end of descendant weighting
            d_values = descendants(new_psi)
    # E0 = np.mean(Eref[50:])
    np.save(f'DMC_CH5_randomly_sampled_coords_{N_0}_walkers_{test_number}', Psi_tau.coords)
    np.save(f'DMC_CH5_randomly_sampled_weights_{N_0}_walkers_{test_number}', np.vstack((Psi_tau.weights, d_values)))
    np.save(f"Non_imp_sampled/DMC_CH5_Energy_{N_0}_walkers_{test_number}", np.vstack((time, Eref, weights)))
    return Eref


# N_0 = 10000
# run(200, 1)
pool = mp.Pool(mp.cpu_count()-1)
# run(100, 'get_dat_wvfn')
test = [2500, 3000, 3500, 4000, 4500, 5500]
for j in range(5):
    for i in range(6):
        N_0 = test[i]
        run(500, j+1)
        print(f'{test[i]} walker test {j+1} is done!')
# Eref, time = tm.time_me(run, 0)
# tm.print_time_list(run, time)
# plt.plot(Eref*har2wave)
# plt.xlabel('Time')
# plt.ylabel('Energy (cm^-1)')
# plt.ylim(0, 12000)
# plt.savefig('Non_Importance_sampling_Eref_full.png')
# print(np.mean(Eref[1000:])*har2wave)
# test = Walkers(N_0)
# te, te_time = tm.time_me(Potential, test)
# tm.print_time_list(Potential, te_time)
# E0, E0_time = tm.time_me(run, 0)
# tm.print_time_list(run, E0_time)
# run(50)
# print(np.mean(walker_reaper))
