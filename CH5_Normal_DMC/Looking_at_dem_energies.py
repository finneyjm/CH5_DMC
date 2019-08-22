import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6

def lets_get_some_energies(non_imp_samp_walkers, imp_samp_walkers):
    fig, axes = plt.subplots(2, 1)
    for i in range(len(non_imp_samp_walkers)):
        Energy = np.load(f'DMC_CH5_Energy_{non_imp_samp_walkers[i]}_walkers.npy')
        # axes[0].plot(Energy*har2wave, label=f'{non_imp_samp_walkers[i]} walkers')
        axes[0].scatter(non_imp_samp_walkers[i], np.mean(Energy[5000:]*har2wave), color='red')
    axes[0].plot(non_imp_samp_walkers, [10916]*len(non_imp_samp_walkers), color='purple')
    for j in range(len(imp_samp_walkers)):
        Energy = np.load(f'DMC_imp_samp_CH5_energy_{imp_samp_walkers[j]}_walkers_{j+1}.npy')
        # axes[1].plot(Energy*har2wave, label=f'{imp_samp_walkers[j]} walkers')
        axes[1].scatter(imp_samp_walkers[j], np.mean(Energy[5000:]*har2wave), color='blue')
        print(np.mean(Energy[5000:])*har2wave)
    axes[1].plot(imp_samp_walkers, [10916] * len(imp_samp_walkers), color='purple')
    axes[0].set_xlabel('Number of Walkers')
    axes[1].set_xlabel('Number of Walkers')
    axes[0].set_ylabel('Energy (cm^-1)')
    axes[1].set_ylabel('Energy (cm^-1)')
    # axes[0].set_ylim(10800, 11100)
    # axes[1].set_ylim(10800, 11100)
    # axes[0].legend()
    # axes[1].legend()
    plt.tight_layout()
    fig.savefig('Energy_convergence_importance_samp.png')


walkers1 = [500, 1000, 5000, 10000, 20000]
walkers2 = [100, 200, 500, 1000, 2000]
lets_get_some_energies(walkers1, walkers2)
