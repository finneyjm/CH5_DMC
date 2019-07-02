import numpy as np
import copy

# DMC parameters
dtau = 1.
N_0 = 2000
time_steps = 10000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6

sigma = np.sqrt(dtau/m_red)


class Walkers(object):

    def __init__(self, walkers, cuts):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.zeros(walkers)
        self.weights = np.zeros((cuts, walkers)) + 1.
        # self.d = np.zeros((cuts, walkers))
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros((cuts, walkers))
        self.Diff = np.zeros((cuts, walkers))


def Kinetic(Psi):
    randomwalk = np.random.normal(0.0, sigma, N_0)
    Psi.coords += randomwalk
    return Psi


def Potential(Psi, bh, spacing, Ecut):
    bh = bh/har2wave
    Ecut = Ecut/har2wave
    A = bh * 8. / spacing ** 2
    B = bh * (4. / spacing ** 2) ** 2
    Vi = bh - A * Psi.coords ** 2 + B * (Psi.coords ** 4)
    cuts = len(Ecut)
    for i in range(cuts):
        ind = np.argwhere(Vi < Ecut[i])
        Psi.V[i, :] = np.array(Vi)
        Psi.V[i, ind] = Ecut[i]
        Psi.Diff[i, :] = Vi - Psi.V[i, :]
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
        # if Psi.weights[i[0], i[1]] == np.max(Psi.weights[:, i[1]]):
            ind = np.unravel_index(Psi.weights.argmax(), Psi.weights.shape)
            if DW is True:
                Biggo_num = float(Psi.walkers[ind[1]])
                Psi.walkers[i] = Biggo_num
            Biggo_weight = np.array(Psi.weights[:, ind[1]])
            Biggo_pos = float(Psi.coords[ind[1]])
            Biggo_pot = np.array(Psi.V[:, ind[1]])
            Biggo_diff = np.array(Psi.Diff[:, ind[1]])
            Psi.coords[i[0]] = Biggo_pos
            Psi.weights[:, i[0]] = Biggo_weight/2.
            Psi.weights[:, ind[1]] = Biggo_weight/2.
            Psi.V[:, i[0]] = Biggo_pot
            Psi.Diff[:, i[0]] = Biggo_diff
    return Psi


def descendants(Psi):
    d = np.zeros((len(Psi.weights[:, 0]), N_0))
    for i in range(N_0):
        d[:, i] = np.sum(Psi.weights[:, Psi.walkers == i])
    return d


def run(equilibration, wait_time, propagation, Ecut, naming):
    barrier = 100.
    spacing = 2.
    cuts = len(Ecut)
    DW = False
    psi = Walkers(N_0, cuts)
    Psi = Kinetic(psi)
    Psi = Potential(Psi, barrier, spacing, Ecut)
    Eref = np.zeros((cuts, int(time_steps) + 1))
    Vref = V_ref_calc(Psi)
    Eref[:, 0] += Vref
    new_psi = Weighting(Vref, Psi, DW)

    Psi_tau = 0.
    wait = float(wait_time)
    j = 0
    num_of_dw = int(round((time_steps - equilibration) / (wait_time + propagation)))
    des_weights = np.zeros((num_of_dw, cuts, N_0))
    differences = np.zeros((num_of_dw, cuts, N_0))
    positions = np.zeros((num_of_dw, N_0))
    weights = np.zeros((num_of_dw, cuts, N_0))
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)
        Psi = Kinetic(new_psi)
        Psi = Potential(Psi, barrier, spacing, Ecut)

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

    np.save("DMC_HO_descendants_concrete_multiweight%s" %naming, des_weights)
    np.save("DMC_HO_Diffs_concrete_multiweight%s" %naming, differences)
    np.save("DMC_HO_Energy_concrete_multiweight%s" %naming, Eref)
    np.save("DMC_HO_Psi_pos_concrete_multiweight%s" %naming, positions)
    np.save("DMC_HO_Psi_weights_concrete_multiweight%s" %naming, weights)
    return


def acquire_dis_data():
    Ecut_array = np.linspace(0, 100, num=11)
    run(4000, 500, 50, Ecut_array, '_all_cuts')
    print("All the cuts are done")
    for i in range(len(Ecut_array)):
        run(4000, 500, 50, np.array([Ecut_array[i]]), '_Ecut%s' %Ecut_array[i])
        print("Cut %s is done" %(i+1))


acquire_dis_data()






