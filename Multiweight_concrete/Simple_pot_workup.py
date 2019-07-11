import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6
exact_value = [452.984883748748, 452.9857207971296]#453.58024614005933, 456.8115157826138, 464.03340693238886, 476.2524479756547, 494.31143851304546]
DVR_no_correct = [452.9848837487481, 479.86017397722145, 525.5312334774873, 580.1326581308361, 639.8628850982727, 702.5739445389472]


def load_psi(which_one):
    Psi_coords = np.load("DMC_HO_Psi_pos_concrete_multiweight%s.npy" %which_one)
    Psi_d = np.load("DMC_HO_descendants_concrete_multiweight%s.npy" %which_one)
    Psi_weights = np.load("DMC_HO_Psi_weights_concrete_multiweight%s.npy" %which_one)
    return Psi_coords, Psi_d, Psi_weights


def load_observables(which_one):
    Diffs = np.load("DMC_HO_Diffs_concrete_multiweight%s.npy" %which_one)
    Energy = np.load("DMC_HO_Energy_concrete_multiweight%s.npy" %which_one)
    return Diffs, Energy


class Walkers(object):

    def __init__(self, coords, d, weights):
        self.walkers = np.linspace(0, len(weights[0, 0, :])-1, num=len(weights[0, 0, :]))
        self.weights = weights
        self.coords = coords
        self.d = d


def lets_get_these_graphs(name, Ecut_array):
    # coords, d, weights = load_psi(name)
    # Psi = Walkers(coords, d, weights)
    # diffs, energy = load_observables(name)
    # num_dw = len(Psi.d[:, 0, 0])
    # cuts = len(Psi.d[0, :, 0])
    # Ecut = np.linspace(0, 100., num=cuts)
    # E_corrections = np.zeros((num_dw, cuts))
    # E_correct_mean = np.zeros(cuts)
    # E_correct_std = np.zeros(cuts)
    # Energies = np.zeros(cuts)
    # plt.figure()
    # for j in range(cuts):
    #     for k in range(num_dw):
    #         E_corrections[k, j] += sum(Psi.d[k, j, :] * diffs[k, j, :]) / sum(Psi.d[k, j, :])
    #     E_correct_mean[j] += np.mean(E_corrections[:, j])
    #     E_correct_std[j] += np.std(E_corrections[:, j])
    #     Energies[j] += np.mean(energy[j, 500:]) + E_correct_mean[j]
    # # plt.errorbar(Ecut, Energies*har2wave, yerr=(E_correct_std*har2wave), label='Multi-weight')
    # plt.plot(Ecut, Energies*har2wave, label='Multi-weight')
    # E_corrections_nm = np.zeros((num_dw, cuts))
    # E_correct_mean_nm = np.zeros(cuts)
    # E_correct_std_nm = np.zeros(cuts)
    # Energies_nm = np.zeros(cuts)
    # for j in range(cuts):
    #     coords, d, weights = load_psi('_Ecut%s' %Ecut[j])
    #     Psi = Walkers(coords, d, weights)
    #     diffs, energy = load_observables('_Ecut%s' %Ecut[j])
    #     for k in range(num_dw):
    #         E_corrections_nm[k, j] += sum(Psi.d[k, 0, :] * diffs[k, 0, :]) / sum(Psi.d[k, 0, :])
    #     E_correct_mean_nm[j] += np.mean(E_corrections_nm[:, j])
    #     E_correct_std_nm[j] += np.std(E_corrections_nm[:, j])
    #     Energies_nm[j] += np.mean(energy[0, 500:]) + E_correct_mean_nm[j]
    # # plt.errorbar(Ecut, Energies_nm*har2wave, yerr=(E_correct_std_nm*har2wave), label='Non_Multi_weight')
    # plt.plot(Ecut, Energies_nm*har2wave, label='Non_Multi_weight')
    for i in range(1):
        fig, axes = plt.subplots()
        for l in range(1):
            coords, d, weights = load_psi(name[i][l])
            Psi = Walkers(coords, d, weights)
            diffs, energy = load_observables(name[i][l])
            num_dw = len(Psi.d[:, 0, 0])
            cuts = len(Psi.d[0, :, 0])
            Ecut = np.linspace(0, Ecut_array[i, l], num=(l+2))
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
            axes.plot(Ecut, Energies * har2wave, label='Num of points %s' %(l+2))
        x = np.linspace(0, 10*(i+1), num=(i+2))
        DVR = np.array([])
        for j in range(i+2):
            DVR = np.append(DVR, exact_value[j])
        axes.plot(x, DVR, label='Values From DVR')
        axes.set_xlabel('Ecut (cm^-1)')
        axes.set_ylabel('Ground State Energy (cm^-1)')
        axes.set_title('Ground State Energy with Different Flattening')
        axes.set_ylim(452, 454)
        lgd = axes.legend(loc='center right', bbox_to_anchor=(1.37, 0.5))
        fig.savefig('Energy_after_flattening_multi_simple_more_walkers_other_method%s.png' %Ecut_array[i, 0], bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)


# def Psi_sqrd(how_many, what_factor):
#     for i in range(how_many):
#         coords, d, weights = load_psi(what_factor*(i+1))
#         Psi = Walkers(coords, d, weights)
#         amp, xx = np.histogram(Psi.coords[0, :], weights=Psi.d[0, 0, :], bins=10, range=(-1.75, 1.75), density=True)
#         bins = (xx[1:]+xx[:-1])/2.
#         plt.plot(bins, amp, label='Des. weight = %s' %(what_factor*(i+1)))
#     plt.xlabel('x')
#     plt.legend()
#     plt.show()
#     plt.close()


# Psi_sqrd(7, 25)
# names = [['']*5, ['']*5, ['']*5, ['']*5, ['']*5]
# Ecuts = np.zeros((5, 5))
# for i in range(5):
#     Ecuts[i, :] += 100*(i+1)
#     for j in range(5):
#         names[i][j] += '_to_%s_job' % Ecuts[i, j] + '_{0}'.format(j+1)
# lets_get_these_graphs(names, Ecuts)

name = [['_to_10.0_job_1']]
Ecuts = np.array([[10.]])
lets_get_these_graphs(name, Ecuts)




