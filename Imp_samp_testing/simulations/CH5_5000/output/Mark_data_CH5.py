import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

energy = np.load('energies.npy')
plt.plot(energy*har2wave)
plt.plot(np.array([10917]*len(energy)))
plt.show()
plt.close()

wvfn = np.load('wavefunction_11.npz')
coords = wvfn['coords.npy']
weights = wvfn['weights.npy']
a = np.linalg.norm(coords[:, 1]-coords[:, 0], axis=1)
amp, xx = np.histogram(np.linalg.norm(coords[:, 1]-coords[:, 0], axis=1)/ang2bohr, weights=weights, bins=50)
bin = (xx[1:] + xx[:-1]) / 2.

# plt.plot(bin, amp)
# plt.show()
