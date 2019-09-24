import numpy as np
from scipy import stats
from Coordinerds.CoordinateSystems import *
import matplotlib.pyplot as plt
ang2bohr = 1.e-10/5.291772106712e-11

order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
x = np.linspace(0.6, 4., num=500)/ang2bohr
amp_all = np.zeros((5, 5, len(x)))
for i in range(5):
    coords = np.load(f'Non_imp_sampled/DMC_CH5_coords_10000_walkers_{i+1}.npy')
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    weights = np.load(f'Non_imp_sampled/DMC_CH5_weights_10000_walkers_{i+1}.npy')
    for CH in range(5):
        density = stats.gaussian_kde(zmat[:, CH, 1]/ang2bohr, weights=weights[0, :])
        amp_all[i, CH, :] += density(x)

amp = np.mean(np.mean(amp_all, axis=0), axis=0)
plt.plot(x, amp, label='phi')

# amp_all = np.zeros((5, 5, len(x)))
# for i in range(5):
#     coords = np.load(f'Non_imp_sampled/DMC_CH5_coords_10000_walkers_{i+1}.npy')
#     zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
#     weights = np.load(f'Non_imp_sampled/DMC_CH5_weights_10000_walkers_{i+1}.npy')
#     for CH in range(5):
#         density = stats.gaussian_kde(zmat[:, CH, 1]/ang2bohr, weights=weights[1, :])
#         amp_all[i, CH, :] += density(x)
#
# amp = np.mean(np.mean(amp_all, axis=0), axis=0)
# plt.plot(x, amp, label='phi^2')
# plt.plot(x, amp**(0.5)*1.615, label='sqrt phi^2')

x = np.linspace(0.4, 6., 5000)/ang2bohr
psi = np.zeros((5, 5000))
type = 'min'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')

min_psi = np.mean(psi, axis=0)
type = 'cs'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')

cs_psi = np.mean(psi, axis=0)

type = 'c2v'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')

c2v_psi = np.mean(psi, axis=0)

new_psi = np.mean(np.vstack((min_psi, cs_psi, c2v_psi)), axis=0)

lkj = np.load('Fits_CH_stretch_wvfns/Average_no_fit.npy')
plt.plot(x, lkj[1, :]*42., label='average?')
plt.plot(x, new_psi*42., label='averaged psi')
plt.xlim(0.8, 1.8)
plt.legend()
plt.savefig('testingiasdfk.png')

