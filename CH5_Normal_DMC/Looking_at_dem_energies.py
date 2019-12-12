import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from Coordinerds.CoordinateSystems import *

order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11


def lets_get_some_energies(non_imp_samp_walkers, imp_samp_walkers, trials_ni, trials_i, broad):
    N_i = len(imp_samp_walkers)
    N_n = len(non_imp_samp_walkers)
    energies_imp = np.zeros((N_i, trials_i))
    energies_non = np.zeros((N_n, trials_ni))
    for j in range(trials_ni):
        for i in range(N_n):
            Energy = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled_CD/Non_imp_sampled_CD_{non_imp_samp_walkers[i]}_' +
                             f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
            energies_non[i, j] += np.mean(Energy[1000:])
    for j in range(trials_i):
        for i in range(N_i):
            Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCH_min_wvfn/' +
                             f'HH_to_rCH_min_wvfn_{imp_samp_walkers[i]}_' +
                             f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
            energies_imp[i, j] += np.mean(Energy[1000:])

    avg_imp = np.mean(energies_imp, axis=1)
    print(avg_imp)
    std_imp = np.std(energies_imp, axis=1)
    print(std_imp)
    avg_non = np.mean(energies_non, axis=1)
    print(avg_non)
    std_non = np.std(energies_non, axis=1)
    print(std_non)
    fig, axes = plt.subplots(2, 1)
    axes[0].errorbar(non_imp_samp_walkers, avg_non, yerr=std_non, color='red', label='Non Imp Sampling')
    axes[0].plot(non_imp_samp_walkers, [10917]*len(non_imp_samp_walkers), color='purple')
    axes[1].errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='blue', label='Imp Sampling')
    axes[1].plot(imp_samp_walkers, [10917] * len(imp_samp_walkers), color='purple')
    axes[0].set_xlabel('Number of Walkers')
    axes[1].set_xlabel('Number of Walkers')
    axes[0].set_ylabel('Energy (cm^-1)')
    axes[1].set_ylabel('Energy (cm^-1)')
    axes[0].set_ylim(10900, 10950)
    axes[1].set_ylim(10900, 10950)
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f'Convergence_plots/Energy_convergence_CH5_HH_to_rCH_min_wvfn_correct_mass.png')
    # plt.close(fig)


walkers1 = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
walkers3 = [100, 200, 500, 1000, 2000, 5000, 10000]
walkers2 = [500, 1000, 2000, 5000, 10000]
walkers4 = [100, 200, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
            5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 20000]
braod = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1]
bro = [1.01, 1.05, 1.10, 1.50]
# for i in bro:
lets_get_some_energies(walkers1, walkers1, 5, 5, 4)
plt.show()