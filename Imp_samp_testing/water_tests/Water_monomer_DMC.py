import copy
from scipy import interpolate
import numpy as np
import Water_monomer_pot_fns as wm
import multiprocessing as mp

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_OH = (m_H*m_O)/(m_H+m_O)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                           [0.957840000000000, 0.000000000000000, 0.000000000000000],
                           [-0.23995350000000, 0.927297000000000, 0.000000000000000]])*ang2bohr

np.save('monomer_coords', coords_initial)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, num_waters):
        self.walkers = np.arange(0, N_0)
        self.coords = np.array([coords_initial]*walkers*num_waters).reshape(walkers, num_waters*3, 3)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.


# Random walk of all the walkers
def Kinetic(Psi, sigma):
    N = len(Psi.coords)
    randomwalk = np.random.normal(0.0, sigma, size=(N, sigma.shape[0], sigma.shape[1]))
    Psi.coords += randomwalk
    return Psi


def Single_Potential(coords):
    V = wm.PatrickShinglePotential(coords)
    V = np.array(V)
    return V

def Potential(Psi, num_waters):
    coords = Psi.coords.reshape(N_0*num_waters, 3, 3)
    coords = np.array_split(coords, mp.cpu_count()-1)
    V = pool.map(Single_Potential, coords)
    V = np.concatenate(V)
    if num_waters is not 1:
        V = np.add.reduceat(V, np.arange(0, len(V), num_waters))
    Psi.V = V
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


def run(propagation, test_number, num_waters):
    DW = False
    psi = Walkers(N_0, num_waters)
    Eref = np.array([])
    time = np.array([])
    weights = np.array([])
    sigma = np.zeros((num_waters*3, 3))
    for i in range(num_waters*3):
        if (i+1) % 3 == 0:
            sigma[i] = np.array([[sigmaO]*3])
        else:
            sigma[i] = np.array([[sigmaH]*3])
    sigma = np.flip(sigma, axis=0)

    # initial parameters before running the calculation
    prop = float(propagation)
    Psi_tau = 0  #
    print('starting sim')
    Vref = 0
    for i in range(int(time_steps)):
        if (i+1) % 500 == 0:
            print(i+1)
            print(Vref*har2wave/num_waters)


        if DW is False:
            prop = float(propagation)
        else:
            prop -= 1.

        if i == 0:
            psi = Potential(psi, num_waters)
            Vref = V_ref_calc(psi)

        psi = Kinetic(psi, sigma)
        psi = Potential(psi, num_waters)

        psi = Weighting(Vref, psi, DW)
        Vref = V_ref_calc(psi)

        Eref = np.append(Eref, Vref)
        time = np.append(time, 2 + i)
        weights = np.append(weights, np.sum(psi.weights))

        if i >= (time_steps - 1. - float(propagation)) and prop > 0:  # start of descendant weighting
            DW = True
            Psi_tau = copy.deepcopy(psi)
        elif i >= (time_steps - 1. - float(propagation)) and prop == 0:  # end of descendant weighting
            d_values = descendants(psi)
    np.save(f'DMC_water_coords_{N_0}_walkers_{dtau}_au_{test_number}_{num_waters}_waters', Psi_tau.coords)
    np.save(f'DMC_water_weights_{N_0}_walkers_{dtau}_au_{test_number}_{num_waters}_waters', np.vstack((Psi_tau.weights, d_values)))
    np.save(f'DMC_water_energy_{N_0}_walkers_{dtau}_au_{test_number}_{num_waters}_waters', np.vstack((time, Eref, weights)))
    return Eref

pool = mp.Pool(mp.cpu_count()-1)
# DMC parameters
dtau = 1.
time_steps = 20000.
alpha = 1./(2.*dtau)
# Values for Simulation
sigmaH = np.sqrt(dtau/m_H)
sigmaO = np.sqrt(dtau/m_O)

N_0 = 20000
for i in range(4):
    test_num = i+1
    num_waters = 1
    run(250, test_num, num_waters)
    energy = np.load(f'DMC_water_energy_{N_0}_walkers_{dtau}_au_{test_num}_{num_waters}_waters.npy')
    print(np.mean(energy[1, 500:])*har2wave/num_waters)
#    print(f'{N_0} Walker Test {i+1} is done!')



