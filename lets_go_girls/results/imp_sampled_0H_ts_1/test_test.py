import numpy as np
import matplotlib.pyplot as plt
har2wave = 219474.6
a = np.load('Non_imp_sampled_0H_ts_1_200_Walkers_Test_1.npz')
b = a['Eref']*har2wave
plt.plot(b)
plt.show()