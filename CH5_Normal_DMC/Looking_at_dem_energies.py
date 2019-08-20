import numpy as np
import matplotlib.pyplot as plt


har2wave = 219474.6

energy = np.load('DMC_CH5_Energy.npy')
plt.plot(energy*har2wave)
plt.xlabel('Time')
plt.ylabel('Energy (cm^-1)')
plt.ylim(0, 12000)
plt.savefig('Non_Importance_sampling_Eref_full.png')


