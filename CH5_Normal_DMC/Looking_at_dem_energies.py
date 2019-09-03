import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6


def lets_get_some_energies(non_imp_samp_walkers, imp_samp_walkers):
    N_i = len(imp_samp_walkers)
    N_n = len(non_imp_samp_walkers)
    energies_imp = np.zeros((N_i, 5))
    energies_non = np.zeros((N_n, 5))
    for j in range(5):
        for i in range(N_i):
            Energy = np.load(f'Biased_DMC_imp_samp_CH5_energy_alpha_3_{imp_samp_walkers[i]}_walkers_{j+1}.npy')[1, :]*har2wave
            energies_imp[i, j] += np.mean(Energy[1000:])
        for i in range(N_n):
            Energy = np.load(f'DMC_CH5_Energy_{non_imp_samp_walkers[i]}_walkers_{j+1}.npy')[1, :]*har2wave
            energies_non[i, j] += np.mean(Energy[1000:])

    avg_imp = np.mean(energies_imp, axis=1)
    avg_non = np.mean(energies_non, axis=1)
    std_imp = np.std(energies_imp, axis=1)
    std_non = np.std(energies_non, axis=1)
    fig, axes = plt.subplots(2, 1)
    axes[0].errorbar(non_imp_samp_walkers, avg_non, yerr=std_non, color='red', label='Non Imp Sampling')
    axes[0].plot(non_imp_samp_walkers, [10916]*len(non_imp_samp_walkers), color='purple')
    axes[1].errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='blue', label='Imp Sampling')
    axes[1].plot(imp_samp_walkers, [10916] * len(imp_samp_walkers), color='purple')
    axes[0].set_xlabel('Number of Walkers')
    axes[1].set_xlabel('Number of Walkers')
    axes[0].set_ylabel('Energy (cm^-1)')
    axes[1].set_ylabel('Energy (cm^-1)')
    axes[0].set_ylim(10900, 11100)
    # axes[1].set_ylim(10900, 11100)
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    fig.savefig('Biased_Energy_convergence_importance_samp_alpha_3.png')


walkers1 = [100, 200, 500, 1000, 2000, 5000, 10000]
walkers2 = [100, 200, 500, 1000, 2000, 5000, 10000]
lets_get_some_energies(walkers1, walkers2)
