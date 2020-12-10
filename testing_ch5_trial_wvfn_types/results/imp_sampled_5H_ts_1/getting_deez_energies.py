import numpy as np
import matplotlib.pyplot as plt
har2wave = 219474.6

energy = np.load('imp_sampled_5H_ts_1_2000_Walkers_Test_2.npz')['Eref']*har2wave

print(np.mean(energy[5000:]))

# plt.plot(energy)
# plt.show()