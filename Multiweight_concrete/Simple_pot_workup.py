import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6
exact_value = [452.984883748748, 453.58024614005933, 456.8115157826138, 464.03340693238886, 476.2524479756547, 494.31143851304546]
DVR_no_correct = [452.9848837487481, 479.86017397722145, 525.5312334774873, 580.1326581308361, 639.8628850982727, 702.5739445389472]


def load_psi(which_one):
    Psi_coords = np.load("DMC_HO_Psi_pos_concrete_multiweight%s.npy" %which_one)
    Psi_d = np.load("DMC_HO_descendants_concrete_multiweight%s.npy" %which_one)
    Psi_weights = np.load("DMC_HO_Psi_weights_concrete_multiweight%s.npy" %which_one)
    # Psi_d = np.zeros(Psi_weights.shape) + 1.
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


def lets_get_these_graphs(names, Ecut_array):
    for i in range(5):
        fig, axes = plt.subplots()
        E_correct_mean = np.zeros((5, 5))
        E_correct_std = np.zeros((5, 5))
        Energies = np.zeros((5, 5))
        energies = np.zeros(5)
        en_std = np.zeros(5)
        for job in range(5):
            coords, d, weights = load_psi(names[i][job])
            Psi = Walkers(coords, d, weights)
            diffs, energy = load_observables(names[i][job])
            num_dw = len(Psi.d[:, 0, 0])
            cuts = len(Psi.d[0, :, 0])
            E_corrections = np.zeros((num_dw, cuts))
            for j in range(cuts):
                for l in range(num_dw):
                    E_corrections[l, j] += sum(Psi.d[l, j, :] * diffs[l, j, :]) / sum(Psi.d[l, j, :])
                E_correct_mean[job, j] += np.mean(E_corrections[:, j])
                E_correct_std[job, j] += np.std(E_corrections[:, j])
                Energies[job, j] += np.mean(energy[j, 500:]) + E_correct_mean[job, j]
        energies += np.mean(Energies, axis=0)
        en_std += np.mean(E_correct_std, axis=0) + np.std(Energies, axis=0)
        ecut = np.linspace(0, Ecut_array[i, 0], num=5)
        axes.errorbar(ecut, energies*har2wave, yerr= en_std*har2wave, label='DMC with correction')
        x = np.linspace(0, Ecut_array[i, 0], num=i+2)
        DVR = np.array([])
        for j in range(i+2):
            DVR = np.append(DVR, exact_value[j])
        axes.plot(x, DVR, label='Values from DVR')
        axes.set_xlabel('Ecut (cm^-1)')
        axes.set_ylabel('Grounds State Energy (cm^-1)')
        axes.set_title('Ground State Energy with Different Flattening')
        # axes.set_ylim(450, 500)
        lgd = axes.legend(loc='center right', bbox_to_anchor=(1.67, 0.5))
        fig.savefig('Energy_after_flattening_multi_simple_fixed_des100%s.png' %str(Ecut_array[i, 0]), bbox_extra_artists=(lgd,), bbox_inches='tight')
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
names = [['']*5, ['']*5, ['']*5, ['']*5, ['']*5]
Ecuts = np.zeros((5, 5))
for i in range(5):
    Ecuts[i, :] += 100*(i+1)
    for j in range(5):
        names[i][j] += '_to_%s_job' % Ecuts[i, j] + '_{0}'.format(j+1)
lets_get_these_graphs(names, Ecuts)

# name = [['_to_10.0_job_1']]
# Ecuts = np.array([[10.]])
# lets_get_these_graphs(name, Ecuts)




