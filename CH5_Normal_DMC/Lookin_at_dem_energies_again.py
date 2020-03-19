import numpy as np
import matplotlib as plt

har2wave = 219474.6

thresh = ['half', 'one', 'five', 'ten', 'twenty']

energies = np.zeros((len(thresh), 3))

for i in range(len(thresh)):
    for j in range(3):
        Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10_thresh_{thresh[i]}/' +
                         f'ptetramer_non_imp_samp_ts_10_thresh_{thresh[i]}_{5000}_' +
                         f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
        a = -0.122146858971399 * har2wave
        energies[i, j] = np.mean(Energy[5000:]) - a

avg_non = np.mean(energies, axis=1)
print(avg_non)
std_non = np.std(energies, axis=1)
print(std_non)





