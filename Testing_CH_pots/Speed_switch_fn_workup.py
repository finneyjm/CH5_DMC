import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6

indices = ['min', 'cs', 'c2v']
energy_array = np.zeros((3, 3, 6, 5, 11))
DVR = np.load('DVR_energies_each_CH_stretch.npy')
DMC_nis = np.load('Non_imp_samp_DMC_energies.npy')
Avg_en = np.load('Average_GSW_DMC_energies.npy')

for n, wvfn in enumerate(indices):
    for m, pot in enumerate(indices):
        energy_array[n, m, :, :, :] += np.load(f'Imp_{pot}_energies_{wvfn}_switch.npy')

energy_stats = np.zeros((2, 3, 3, 5, 11))
energy_stats[0, :] += np.mean(energy_array, axis=2)
energy_stats[1, :] += np.std(energy_array, axis=2)

speed = np.linspace(0.5, 5.5, 11)

for i in range(3):
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    for l in range(3):
        for j in range(5):
            axes[l][j].errorbar(speed, energy_stats[0, i, l, j, :], yerr=energy_stats[1, i, l, j, :], color=f'C{j}',
                                label=f'DMC CH stretch {j+1}')
            axes[l][j].errorbar(speed, [DMC_nis[0, l, j]]*len(speed), yerr=[DMC_nis[1, l, j]]*len(speed), color='cyan',
                                label=f'DMC w/o imp samp CH stretch {j+1}')
            axes[l][j].plot(speed, [DVR[l, j]]*len(speed), 'black', label=f'DVR CH stretch {j+1}')
            axes[l][j].set_xlabel('Steepness of Transistion')
            axes[l][j].set_ylabel('Ground State Energy (cm^-1)')
            axes[l][j].legend(loc='lower left')
            axes[l][j].set_ylim(DVR[l, j] - 5., DVR[l, j] + 5.)
    plt.tight_layout()
    fig.savefig(f'Energy_increase_steepness_{indices[i]}_wvfn.png')
    plt.close(fig)

for i in range(3):
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    for l in range(3):
        for j in range(5):
            axes[l][j].errorbar(speed, energy_stats[0, i, l, j, :], yerr=energy_stats[1, i, l, j, :], color=f'C{j}',
                                label=f'DMC CH stretch {j+1}')
            axes[l][j].errorbar(speed, [Avg_en[i, 0, l, j]]*len(speed), yerr=[Avg_en[i, 1, l, j]]*len(speed),
                                color='cyan', label=f'DMC w/o imp samp CH stretch {j+1}')
            axes[l][j].plot(speed, [DVR[l, j]]*len(speed), 'black', label=f'DVR CH stretch {j+1}')
            axes[l][j].set_xlabel('Steepness of Transistion')
            axes[l][j].set_ylabel('Ground State Energy (cm^-1)')
            axes[l][j].legend(loc='lower left')
            axes[l][j].set_ylim(DVR[l, j] - 5., DVR[l, j] + 5.)
    plt.tight_layout()
    fig.savefig(f'Energy_increase_steepness_vs_avg_{indices[i]}_wvfn.png')
    plt.close(fig)
