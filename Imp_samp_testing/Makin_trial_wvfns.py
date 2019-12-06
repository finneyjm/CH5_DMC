import numpy as np
from scipy import stats
import scipy.optimize
from Coordinerds.CoordinateSystems import *
import matplotlib.pyplot as plt
ang2bohr = 1.e-10/5.291772106712e-11

order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
x = np.linspace(0.4, 6., 500)/ang2bohr
amp_all = np.zeros((5, 5, len(x)))
for i in range(5):
    coords = np.load(f'Non_imp_sampled/DMC_CH5_coords_20000_walkers_{i+1}.npy')
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    weights = np.load(f'Non_imp_sampled/DMC_CH5_weights_20000_walkers_{i+1}.npy')
    for CH in range(5):
        density = stats.gaussian_kde(zmat[:, CH, 1]/ang2bohr, weights=weights[0, :])
        amp_all[i, CH, :] += density(x)

amp = np.mean(np.mean(amp_all, axis=0), axis=0)
plt.plot(x, amp, label='phi')

x = np.linspace(0.4, 6., 5000)/ang2bohr
psi = np.zeros((5, 5000))
type = 'min'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')
    if np.max(psi[i, :]) < 0.02:
        psi[i, :] *= -1.
trial_psi = np.mean(psi, axis=0)
# other_psi = np.mean(psi[0:3], axis=0)

psi = np.zeros((5, 5000))
type = 'min'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')


min_psi = np.mean(psi, axis=0)
psi = np.zeros((5, 5000))
type = 'cs'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')
    if np.max(psi[i, :]) < 0.02:
        psi[i, :] *= -1.


cs_psi = np.mean(psi, axis=0)
psi = np.zeros((5, 5000))
type = 'c2v'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')
    if np.max(psi[i, :]) < 0.02:
        psi[i, :] *= -1.


c2v_psi = np.mean(psi, axis=0)

a = np.max(amp)
#
#
# def get_this_fit(x, *args):
#     w1, w2, w3 = args
#     blah = np.average(np.vstack((min_psi, cs_psi, c2v_psi)), weights=np.abs([w1, w2, w3])/np.linalg.norm([w1, w2, w3]), axis=0)
#     b = np.max(blah)
#     return blah*a/b


# params = [0.74, 0.20, 0.06]
# fitted_params, _ = scipy.optimize.curve_fit(get_this_fit, x, amp, p0=params)
# print(np.abs(fitted_params)/np.linalg.norm(fitted_params))

# new_psi = np.average(np.vstack((min_psi, cs_psi, c2v_psi)), axis=0)
# b = np.argmax(new_psi)
# plt.plot(x, new_psi*a/new_psi[b], label='average')
b_new = np.argmax(min_psi)
# plt.plot(x, min_psi*a/min_psi[b_new], label='incorrect min average')
b = np.argmax(trial_psi)
# plt.plot(x, trial_psi*a/trial_psi[b], label='min average')
np.save('params/min_wvfns/Average_min_broadening_1.0x', trial_psi)
# b = np.argmax(other_psi)
# plt.plot(x, other_psi*a/other_psi[b], label='other psi')
# b = np.argmax(c2v_psi)
# plt.plot(x, c2v_psi*a/c2v_psi[b], label='c2v average')
# b = np.argmax(cs_psi)
# plt.plot(x, cs_psi*a/cs_psi[b], label='cs average')
# b = np.argmax(-psi[0, :])
# plt.plot(x, -psi[0, :]*a/(-psi[0, b]), label='CH stretch 1')
# b = np.argmax(psi[1, :])
# plt.plot(x, psi[1, :]*a/psi[1, b], label='CH stretch 2')
# b = np.argmax(psi[4, :])
# plt.plot(x, psi[4, :]*a/psi[4, b], label='CH stretch 5')
# for i in range(10):
# bro = 1.0
# new_x = (x - x[b_new])*(bro) + x[b_new]
# np.save(f'min_wvfns/Average_min_broadening_{bro}x', np.vstack((new_x*ang2bohr, min_psi)))
# b = np.max(trial_psi)
# plt.plot(new_x, min_psi*a/min_psi[b_new], label=f'min with {bro}x broadening')
# new_psi = np.mean(np.vstack((min_psi, cs_psi, c2v_psi)), axis=0)
# plt.plot(x, get_this_fit(x, *fitted_params), label='fit')
# lkj = np.load('Fits_CH_stretch_wvfns/Average_no_fit.npy')
# plt.plot(x, lkj[1, :]*42., label='average?')
# # plt.plot(x, new_psi*42., label='averaged psi')
# plt.xlim(0.8, 1.8)
# plt.xlabel('rCH (Angstrom)')
# plt.legend()
# plt.show()

