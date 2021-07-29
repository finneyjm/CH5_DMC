import numpy as np
import multiprocessing as mp
from ProtWaterPES import *
import copy

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_OD = (m_D*m_O)/(m_D+m_O)
m_OH = (m_H*m_O)/(m_H+m_O)
dtau = 1
alpha = 1./(2.*dtau)
sigmaH = np.sqrt(dtau/m_H)
sigmaO = np.sqrt(dtau/m_O)
sigmaD = np.sqrt(dtau/m_D)
sigma = np.broadcast_to(np.array([sigmaH, sigmaO, sigmaH, sigmaO, sigmaH])[:, None], (5, 3))
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

omega_asym = 3815.044564/har2wave


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([initial_struct]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)


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


def pot(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    dis = all_dists(Psi.coords)
    Psi.V[dis > 0] = 10
    return Psi


pool = mp.Pool(mp.cpu_count()-1)


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0], [3, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    # z = 1/2*(dis[:, -2] - dis[:, -1])
    mid = dis[:, 2]/2
    sp = -mid + dis[:, -2]*np.cos(roh_roo_angle(coords, dis[:, -3], dis[:, -2]))
    combo = -0.60594644269321474*sp + 42.200232187251913*a_oh
    combo2 = 41.561937672470521*sp + 1.0206303697659393*a_oh
    return np.array([combo, combo2, sp, a_oh])


def roh_roo_angle(coords, roo_dist, roh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (coords[:, 1]-coords[:, 0])/np.broadcast_to(roh_dist[:, None], (len(roh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


def run(N_0, time_steps, propagation, equilibration, wait_time, initial_struct):
    DW = False
    psi = Walkers(N_0, initial_struct)
    num_o_collections = int((time_steps - equilibration) / (propagation + wait_time)) + 1
    time = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))

    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Eref_array = np.zeros(time_steps)

    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)

        if DW is False:
            prop = float(propagation)
            wait -= 1.
        else:
            prop -= 1.

        if i == 0:
            psi = pot(psi)
            Vref = V_ref_calc(psi, dtau)

        psi = Kinetic(psi, sigma)
        psi = pot(psi)

        psi = Weighting(Vref, psi, DW, dtau)
        Vref = V_ref_calc(psi, dtau)

        Eref_array[i] = Vref
        time[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)

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

    return coords, weights, time, Eref_array, sum_weights, des


test_structure2 = np.array([
        [ 2.45704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.47502706, -1.46894299, -0.69579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.84391277,  0.14767547,  1.4669121 ],
])

test_structure = np.array([
        [ 2.75704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.87502706, -1.66894299, -0.79579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.84391277,  0.14767547,  1.4669121 ],
])

balsadfh = np.array([[ 2.53115388, -0.70229191,  0.09419782],
  [ 0.24733467, -0.43385458, -0.42577512],
  [-0.37487855, -2.12955265, -0.85581918],
  [ 4.82504351, -0.30846811, -0.3546176 ],
  [ 5.6966473,   0.96754918,  0.41819534]])

d = all_dists(np.array([test_structure2]*1))
print(d)
print(all_dists(np.array([test_structure]*1)))
print(all_dists(np.array([balsadfh]*1)))

# coords, weights, time, Eref_array, sum_weights, des = run(20000, 5000, 250, 500, 500, test_structure2)
#
# print(np.mean(Eref_array[500:]*har2wave))
