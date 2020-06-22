import numpy as np

har2wave = 219474.6


Blah = np.load('../../../CH5_Normal_DMC/Trial_wvfn_testing/results/pmonomer_full_imp_samp_water/pmonomer_full_imp_samp_water_100_Walkers_Test_1.npz')
energy = Blah['Eref']*har2wave
coords = Blah['coords']
weights = Blah['weights']
des = Blah['des']
import matplotlib.pyplot as plt

plt.plot(energy)
plt.show()