import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
har2wave = 219474.6

wvfn = np.load('wavefunction_1.npz')

coords = wvfn['coords.npy']
weights = wvfn['weights.npy']


def dists(coords, ind1, ind2):
    return la.norm(coords[:, ind1]-coords[:, ind2], axis=1)


dist_mat = np.zeros((len(weights), 5))
for i in range(5):
    dist_mat[:, i] = dists(coords, 0, i+1)

amp, xx = np.histogram(dist_mat[:, 0], weights=weights, bins=40)
bin = (xx[1:] + xx[:-1]) / 2.
plt.plot(bin, amp)
plt.show()
