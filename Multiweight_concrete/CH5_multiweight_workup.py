import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6


def load_psi(which_one):
    Psi_coords = np.load("DMC_CH5_Psi_pos_concrete_multiweight%s.npy" %which_one)
    Psi_d = np.load("DMC_CH5_descendants_concrete_multiweight%s.npy" %which_one)
    Psi_weights = np.load("DMC_CH5_Psi_weights_concrete_multiweight%s.npy" %which_one)
    return Psi_coords, Psi_d, Psi_weights


def load_observables(which_one):
    Diffs = np.load("DMC_CH5_Diffs_concrete_multiweight%s.npy" %which_one)
    Energy = np.load("DMC_CH5_Energy_concrete_multiweight%s.npy" %which_one)
    return Diffs, Energy


class Walkers(object):

    def __init__(self, coords, d, weights):
        self.walkers = np.linspace(0, len(weights[0, 0, :])-1, num=len(weights[0, 0, :]))
        self.weights = weights
        self.coords = coords
        self.d = d


def lets_get_these_graphs(name, Ecut_array):
    for i in range(len(Ecut_array)):
        coords, d, weights = load_psi(name[i])
        Psi = Walkers(coords, d, weights)
        diffs, energy = load_observables(name[i])
        num_dw = len(Psi.d[:, 0, 0])
        cuts = len(Psi.d[0, :, 0])
        Ecut = np.linspace(0, Ecut_array[i], num=cuts)
        E_corrections = np.zeros((num_dw, cuts))
        E_correct_mean = np.zeros(cuts)
        E_correct_std = np.zeros(cuts)
        Energies = np.zeros(cuts)
        for j in range(cuts):
            for k in range(num_dw):
                E_corrections[k, j] += sum(Psi.d[k, j, :] * diffs[k, j, :]) / sum(Psi.d[k, j, :])
            E_correct_mean[j] += np.mean(E_corrections[:, j])
            E_correct_std[j] += np.std(E_corrections[:, j])
            Energies[j] += np.mean(energy[j, 500:]) + E_correct_mean[j]
        plt.plot(Ecut, Energies * har2wave, label='Cuts to %s' %Ecut_array[i])
    plt.xlabel('Ecut (cm^-1)')
    plt.ylabel('Ground State Energy (cm^-1)')
    plt.title('Ground State Energy with Different Flattening')
    plt.legend(loc=4)
    plt.savefig('Energy_after_flattening_multi_CH5.png')


def new_graphs(name, Ecut_array):
    for i in range(len(Ecut_array)):
        diffs, energy = load_observables(name[i])
        plt.figure()
        Ecut = np.linspace(0, Ecut_array[i], num=len(energy[:, 0]))
        for j in range(len(energy[:, 0])):
            plt.plot(energy[j, :] * har2wave, label='Cut at %s' %Ecut[j])
        plt.xlabel('Time')
        plt.ylabel('Ground State Energy(cm^-1)')
        plt.title('Eref Fluctuations')
        plt.legend(loc=4)
        plt.savefig('Eref_fluctuations_for_Ecut_%s' %Ecut_array[i])



names = ['_to_10000', '_to_8000', '_to_6000', '_to_4000', '_to_2000']
Ecuts = [10000, 8000, 6000, 4000, 2000]
# lets_get_these_graphs(names, Ecuts)
new_graphs(names, Ecuts)



