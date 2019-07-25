import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6
exact_value = [452.984883748748, 453.58024614005933, 456.8115157826138, 464.03340693238886, 476.2524479756547, 494.31143851304546]
DVR_no_correct = [452.9848837487481, 479.86017397722145, 525.5312334774873, 580.1326581308361, 639.8628850982727, 702.5739445389472]


def load_psi(which_one):
    Psi_coords = np.load("DMC_HO_Psi_pos_concrete_%s.npy" %which_one)
    Psi_d = np.load("DMC_HO_descendants_concrete_%s.npy" %which_one)
    Psi_weights = np.load("DMC_HO_Psi_weights_concrete_%s.npy" %which_one)
    return Psi_coords, Psi_d, Psi_weights


def load_observables(which_one):
    Diffs = np.load("DMC_HO_Diffs_concrete_%s.npy" %which_one)
    Energy = np.load("DMC_HO_Energy_concrete_%s.npy" %which_one)
    return Diffs, Energy


class Walkers(object):

    def __init__(self, coords, d, weights):
        self.walkers = np.linspace(0, len(weights[0, :])-1, num=len(weights[0, :]))
        self.weights = weights
        self.coords = coords
        self.d = d


def lets_get_these_graphs(Ecut_array):
    for i in range(5):
        fig, axes = plt.subplots()
        E_correct_mean = np.zeros((5, 5))
        E_correct_std = np.zeros((5, 5))
        Energies = np.zeros((5, 5))
        energies = np.zeros(5)
        en_std = np.zeros(5)
        for l in range(5):
            for job in range(5):
                coords, d, weights = load_psi('_Ecut_%s_job' % Ecut_array[i, l] + '_{0}_desweight100'.format(job+1))
                Psi = Walkers(coords, d, weights)
                diffs, energy = load_observables('_Ecut_%s_job' % Ecut_array[i, l] + '_{0}_desweight100'.format(job+1))
                num_dw = len(Psi.d[:, 0])
                E_corrections = np.zeros(num_dw)
                for j in range(num_dw):
                    E_corrections[j] += sum(Psi.d[j, :] * diffs[j, :])/sum(Psi.d[j, :])
                E_correct_mean[l, job] += np.mean(E_corrections)
                E_correct_std[l, job] += np.std(E_corrections)
                Energies[l, job] += np.mean(energy[500:]) + E_correct_mean[l, job]
            energies[l] += np.mean(Energies[l, :])
            en_std[l] += np.mean(E_correct_std[l, :]) + np.std(Energies[l, :])
        axes.errorbar(Ecut_array[i, :], energies*har2wave, yerr=en_std*har2wave, label='DMC with correction')
        x = np.linspace(0, 100*(i+1), num=(i+2))
        DVR = np.array([])
        for DV in range(i+2):
            DVR = np.append(DVR, exact_value[DV])
        axes.plot(x, DVR, label='Values From DVR')
        axes.set_xlabel('Ecut (cm^-1)')
        axes.set_ylabel('Ground State Energy (cm^-1)')
        axes.set_title('Ground State Energy with Different Flattening')
        lgd = axes.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
        # axes.set_ylim(450, 500)
        fig.savefig('Energy_after_flattening_single_DMC_desweight100%s.png' % (str(Ecut_array[i, -1])), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)


# names = [[['']*5, ['']*5, ['']*5, ['']*5, ['']*5]]*5
Ecuts = np.zeros((5, 5))
for i in range(5):
    Ecuts[i, :] += np.linspace(0, 100*(i+1), num=5)
    # for j in range(5):
    #     for job in range(5):
    #         names[i][j][job][0] += str('_Ecut_%s_job' % Ecuts[i, j] + '_{0}'.format(job+1))
lets_get_these_graphs(Ecuts)









