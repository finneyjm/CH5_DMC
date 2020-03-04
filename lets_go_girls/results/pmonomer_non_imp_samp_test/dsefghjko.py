import numpy as np

har2wave = 219474.6
a = np.load('pmonomer_non_imp_samp_2000_Walkers_Test_1.npz')


energy = np.mean(a['Eref'][5000:])
print(energy*har2wave)










