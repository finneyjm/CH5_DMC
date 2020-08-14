import numpy as np
import matplotlib.pyplot as plt
har2wave = 219474.6

Energy = np.zeros(5)
for i in range(5):
    blah = np.load(f'pmonomer_non_imp_samp_ts_10_20000_Walkers_Test_{i+1}.npz')
    energy = blah['Eref']*har2wave
    Energy[i] = np.mean(energy[500:])

print(np.mean(Energy))
print(Energy)