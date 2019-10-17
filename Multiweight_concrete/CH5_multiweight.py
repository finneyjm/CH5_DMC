import numpy as np
import copy
import CH5pot

# DMC parameters
dtau = 5.
N_0 = 20000
time_steps = 10000.
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

    def __init__(self, walkers, cuts):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.array([coords_inital]*walkers)
        self.weights = np.zeros((cuts, walkers)) + 1.
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros((cuts, walkers))
        self.Diff = np.zeros((cuts, walkers))


# Random walk of all the walkers
def Kinetic(Psi):
    randomwalkC = np.random.normal(0.0, sigmaC, size=(N_0, 3))
    randomwalkH = np.random.normal(0.0, sigmaH, size=(N_0, 5, 3))
    Psi.coords[:, 0, :] += randomwalkC
    Psi.coords[:, 1:6, :] += randomwalkH
    return Psi


# Using the potential call to calculate the potential of all the walkers
def Potential(Psi, Ecut):
    Ecut = Ecut/har2wave
    V = CH5pot.mycalcpot(Psi.coords, N_0)
    V_orig = np.array(V)
    cuts = len(Ecut)
    for i in range(cuts):
        ind = np.argwhere(V_orig < Ecut[i])
        Psi.V[i, :] = np.array(V)
        Psi.V[i, ind] = Ecut[i]
        Psi.Diff[i, :] = V_orig - Psi.V[i, :]
    return Psi


def V_ref_calc(Psi):
    cuts = len(Psi.weights[:, 0])
    V_ref = np.zeros(cuts)
    for i in range(cuts):
        V_ref[i] += np.average(Psi.V[i, :], weights=Psi.weights[i, :]) - alpha * (sum(Psi.weights[i, :] - Psi.weights_i)/N_0)
    return V_ref


def Weighting(Vref, Psi, DW):
    cuts = len(Vref)
    for i in range(cuts):
        Psi.weights[i, :] *= np.nan_to_num(np.exp(-(Psi.V[i, :] - Vref[i]) * dtau))
    threshold = 1./float(N_0)
    for j in range(cuts):
        death = np.argwhere(Psi.weights[j, :] < threshold)
        for i in death:
            ind = np.unravel_index(Psi.weights.argmax(), Psi.weights.shape)
            if DW is True:
                Biggo_num = float(Psi.walkers[ind[1]])
                Psi.walkers[i] = Biggo_num
            Biggo_weight = np.array(Psi.weights[:, ind[1]])
            Biggo_pos = np.array(Psi.coords[ind[1]])
            Biggo_pot = np.array(Psi.V[:, ind[1]])
            Biggo_diff = np.array(Psi.Diff[:, ind[1]])
            Psi.coords[i[1]] = Biggo_pos
            Psi.weights[:, i[1]] = Biggo_weight/2.
            Psi.weights[:, ind[1]] = Biggo_weight/2.
            Psi.V[:, i[1]] = Biggo_pot
            Psi.Diff[:, i[1]] = Biggo_diff
    return Psi


def descendants(Psi):
    d = np.zeros((len(Psi.weights[:, 0]), N_0))
    for i in range(N_0):
        d[:, i] = np.sum(Psi.weights[:, Psi.walkers == i])
    return d


def run(equilibration, wait_time, propagation, Ecut, naming):
    cuts = len(Ecut)
    DW = False
    new_psi = Walkers(N_0, cuts)
    Eref = np.zeros((cuts, int(time_steps) + 1))
    Psi_tau = 0.
    wait = float(wait_time)
    prop = float(propagation)
    j = 0
    num_of_dw = int(round((time_steps - equilibration) / (wait_time + propagation)))
    des_weights = np.zeros((num_of_dw, cuts, N_0))
    differences = np.zeros((num_of_dw, cuts, N_0))
    positions = np.zeros((num_of_dw, N_0, 6, 3))
    weights = np.zeros((num_of_dw, cuts, N_0))
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)
        Psi = Kinetic(new_psi)
        Psi = Potential(Psi, Ecut)

        if i == 0:
            Vref = V_ref_calc(Psi)
            Eref[:, 0] += Vref

        if DW is False:
            prop = float(propagation)
            wait -= 1.
        elif DW is True:
            prop -= 1.
            if Psi_tau == 0:
                Psi_tau = copy.deepcopy(Psi)

        new_psi = Weighting(Vref, Psi, DW)

        Vref = V_ref_calc(new_psi)
        Eref[:, i+1] += Vref

        if i >= int(equilibration) and wait <= 0. < prop:  # start of descendant weighting
            DW = True
        elif prop == 0.:  # end of descendant weighting
            d_values = descendants(new_psi)
            if np.all(des_weights[-1, :, :] == 0.):
                des_weights[j, :, :] += d_values
                differences[j, :, :] += Psi_tau.Diff
                positions[j, :] += Psi_tau.coords
                weights[j, :, :] += Psi_tau.weights
                j += 1
            else:
                des_weights = np.vstack((des_weights, d_values[None, ...]))
                differences = np.vstack((differences, Psi_tau.Diff[None, ...]))
                positions = np.vstack((positions, Psi_tau.coords[None, ...]))
                weights = np.vstack((weights, Psi_tau.weights[None, ...]))
            Psi_tau = 0.
            wait = float(wait_time)
            DW = False

    np.save("DMC_CH5_descendants_concrete_multiweight%s" %naming, des_weights)
    np.save("DMC_CH5_Diffs_concrete_multiweight%s" %naming, differences)
    np.save("DMC_CH5_Energy_concrete_multiweight%s" %naming, Eref)
    np.save("DMC_CH5_Psi_pos_concrete_multiweight%s" %naming, positions)
    np.save("DMC_CH5_Psi_weights_concrete_multiweight%s" %naming, weights)
    return


def acquire_dis_data():
    for i in range(5):
        Ecut_array = np.linspace(0, 2000*(i+1), num=11)
        run(4000, 500, 50, Ecut_array, '_to_%s' %Ecut_array[-1])
        print('Done with Ecut to %s' %Ecut_array[-1])


acquire_dis_data()












