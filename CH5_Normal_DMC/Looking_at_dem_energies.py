import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6
trials = 1


def lets_get_some_energies(non_imp_samp_walkers, imp_samp_walkers, broad):
    N_i = len(imp_samp_walkers)
    N_n = len(non_imp_samp_walkers)
    energies_imp = np.zeros((N_i, trials))
    energies_non = np.zeros((N_n, trials))
    for j in range(trials):
        for i in range(N_n):
            Energy = np.load(f'Non_imp_sampled/DMC_CH5_Energy_{non_imp_samp_walkers[i]}_' +
                             f'walkers_{j+1}.npy')[1, :]*har2wave
            energies_non[i, j] += np.mean(Energy[1000:])
        for i in range(N_i):
            Energy = np.load(f'Trial_wvfn_testing/Avg_wvfn/energies/' +
                             f'Imp_samp_CH5_energy_full_avg_{imp_samp_walkers[i]}_' +
                             f'walkers_{j+1}.npy')[1, :]*har2wave
            energies_imp[i, j] += np.mean(Energy[1000:])

    avg_imp = np.mean(energies_imp, axis=1)
    print(avg_imp)
    avg_non = np.mean(energies_non, axis=1)
    print(avg_non)
    std_imp = np.std(energies_imp, axis=1)
    print(std_imp)
    std_non = np.std(energies_non, axis=1)
    print(std_non)
    fig, axes = plt.subplots(2, 1)
    axes[0].errorbar(non_imp_samp_walkers, avg_non, yerr=std_non, color='red', label='Non Imp Sampling')
    axes[0].plot(non_imp_samp_walkers, [10916]*len(non_imp_samp_walkers), color='purple')
    axes[1].errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='blue', label='Imp Sampling')
    axes[1].plot(imp_samp_walkers, [10916] * len(imp_samp_walkers), color='purple')
    axes[0].set_xlabel('Number of Walkers')
    axes[1].set_xlabel('Number of Walkers')
    axes[0].set_ylabel('Energy (cm^-1)')
    axes[1].set_ylabel('Energy (cm^-1)')
    axes[0].set_ylim(10875, 11000)
    axes[1].set_ylim(10875, 11000)
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f'Convergence_plots/Energy_convergence_CH5_full_avg_wvfn.png')
    plt.close(fig)


walkers1 = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
walkers2 = [500, 1000, 2000, 5000, 10000]
# braod = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
# for i in range(9):
lets_get_some_energies(walkers1, walkers1, 4)
