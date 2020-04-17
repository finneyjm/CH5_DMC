import numpy as np
import matplotlib.pyplot as plt
har2wave = 219474.6

energies = np.load('energies.npy')*har2wave

plt.plot(energies, label='Whatever Mark gave me')
plt.plot(np.array([10917]*len(energies)), label='Right answer')
plt.xlabel(r'$\tau$')
plt.ylabel(r'Energy (cm$^{-1}$)')
plt.legend()
plt.show()
plt.close()

