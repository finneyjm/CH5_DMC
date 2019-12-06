import numpy as np

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.000000000 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_CH = (m_C*m_H)/(m_H+m_C)
m_CD = (m_C*m_D)/(m_D+m_C)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# Starting orientation of walkers
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, atoms=None, rand_samp=False):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial]*walkers)
        if rand_samp is True:
            rand_idx = np.random.rand(walkers, 5).argsort(axis=1) + 1
            b = self.coords[np.arange(walkers)[:, None], rand_idx]
            self.coords[:, 1:6, :] = b
        else:
            self.coords *= 1.01
        self.weights = np.zeros(walkers) + 1.
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        if atoms is None:
            atoms = ['C', 'H', 'H', 'H', 'H', 'H']
        self.atoms = atoms


# Random walk of all the walkers
def Kinetic(Psi, sigmaCH):
    N_0 = len(Psi.coords)
    randomwalkC = np.random.normal(0.0, sigmaCH[0, 0], size=(N_0, 3))
    randomwalkH = np.random.normal(0.0, sigmaCH[1, 0], size=(N_0, 5, 3))
    Psi.coords[:, 0, :] += randomwalkC
    Psi.coords[:, 1:6, :] += randomwalkH
    return Psi


# Calculate V_ref for the weighting calculation and to determine the ground state energy
def V_ref_calc(Psi, dtau):
    alpha = 1./(2.*dtau)
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    V_ref = sum(Psi.weights*Psi.V)/P - alpha*(sum((Psi.weights-Psi.weights_i))/P0)
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




