import numpy as np
import copy

# DMC parameters
dtau = 5.
N_0 = 10000
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

    def __init__(self, walkers):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.zeros(walkers)
        self.weights = np.zeros(walkers) + 1.
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.Diff = np.zeros(walkers)


def Kinetic(Psi):
    randomwalk = np.random.normal(0.0, sigma, N_0)
    Psi.coords += randomwalk
    return Psi


def Potential(Psi, bh, spacing, Ecut):
    bh = bh/har2wave
    Ecut = Ecut/har2wave
    A = bh * 8. / (spacing * spacing)
    B = bh * (4. / (spacing * spacing)) * (4. / (spacing * spacing))
    Vi = bh - A * Psi.coords * Psi.coords + B * (Psi.coords * Psi.coords * Psi.coords * Psi.coords)
    ind = np.argwhere(Vi < Ecut)
    Psi.V = np.array(Vi)
    Psi.V[ind] = Ecut
    Psi.Diff = Vi - Psi.V
    return Psi


# Calculate V_ref for the weighting calculation and to determine the ground state energy
def V_ref_calc(Psi):
    P = sum(Psi.weights)
    V_ref = sum(Psi.weights*Psi.V)/P - alpha*(sum((Psi.weights-Psi.weights_i))/N_0)
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
    d = np.zeros(N_0)
    for i in range(N_0):
        d[i] = np.sum(Psi.weights[Psi.walkers == i])
    return d


def run(equilibration, wait_time, propagation, Ecut, naming):
    barrier = 1000.
    spacing = 2.
    DW = False
    psi = Walkers(N_0)
    Psi = Kinetic(psi)
    Psi = Potential(Psi, barrier, spacing, Ecut)
    Eref = np.zeros(int(time_steps) + 1)
    Vref = V_ref_calc(Psi)
    Eref[0] += Vref
    new_psi = Weighting(Vref, Psi, DW)

    Psi_tau = 0.
    wait = float(wait_time)
    j = 0
    num_of_dw = int(round((time_steps - equilibration) / (wait_time + propagation)))
    des_weights = np.zeros((num_of_dw, N_0))
    differences = np.zeros((num_of_dw, N_0))
    positions = np.zeros((num_of_dw, N_0))
    weights = np.zeros((num_of_dw, N_0))
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
        Eref[i + 1] += Vref

        if i >= int(equilibration) and wait <= 0. < prop:  # start of descendant weighting
            DW = True
        elif prop == 0.:  # end of descendant weighting
            d_values = descendants(new_psi)
            if np.all(des_weights[-1, :] == 0.):
                des_weights[j, :] += d_values
                differences[j, :] += Psi_tau.Diff
                positions[j, :] += Psi_tau.coords
                weights[j, :] += Psi_tau.weights
                j += 1
            else:
                des_weights = np.vstack((des_weights, d_values[None, ...]))
                differences = np.vstack((differences, Psi_tau.Diff[None, ...]))
                positions = np.vstack((positions, Psi_tau.coords[None, ...]))
                weights = np.vstack((weights, Psi_tau.weights[None, ...]))
            Psi_tau = 0.
            wait = float(wait_time)
            DW = False

    np.save("DMC_HO_descendants_concrete_%s" % naming, des_weights)
    np.save("DMC_HO_Diffs_concrete_%s" % naming, differences)
    np.save("DMC_HO_Energy_concrete_%s" % naming, Eref)
    np.save("DMC_HO_Psi_pos_concrete_%s" % naming, positions)
    np.save("DMC_HO_Psi_weights_concrete_%s" % naming, weights)

    return des_weights, differences, Eref, positions, weights


def acquire_dis_data():
    for j in range(5):
        Ecut_array = np.array([0., 25., 50., 75., 100., 125., 150., 200., 225., 250., 300., 375., 400., 500.])
        for l in range(len(Ecut_array)):
            run(4000, 500, 50, Ecut_array[l], '_Ecut_%s_job' %Ecut_array[l] + '_%s' %(j+1))
            print('Done with Ecut to %s' %Ecut_array[l] + ' job %s!' %(j+1))


acquire_dis_data()


