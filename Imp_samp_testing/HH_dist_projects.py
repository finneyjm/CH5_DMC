import numpy as np
import matplotlib.pyplot as plt

ang2bohr = 1.e-10/5.291772106712e-11


def hh_dist(coords):
    N = len(coords[:, 0, 0])
    hh = np.zeros((N, 10))
    for i in range(4):
        hh[:, i] += np.linalg.norm(coords[:, 1, :] - coords[:, 2+i, :], axis=1)
    for i in range(3):
        hh[:, i+4] += np.linalg.norm(coords[:, 2+i, :] - coords[:, 3+i, :], axis=1)
    for i in range(2):
        hh[:, i+7] += np.linalg.norm(coords[:, 3+i, :] - coords[:, 4+i, :], axis=1)
    hh[:, 9] += np.linalg.norm(coords[:, 4, :] - coords[:, 5, :], axis=1)
    return hh.flatten('F')


N_0 = 100
alpha = 61
bining = 50
amp_all = np.zeros((bining, 5))
for j in range(5):
    coords = np.load(f'Trial_wvfn_testing/alpha_{alpha}/coords/Imp_samp_DMC_CH5_coords_alpha_{alpha}_{N_0}_walkers_{j+1}.npy')
    weights = np.load(f'Trial_wvfn_testing/alpha_{alpha}/weights/Imp_samp_DMC_CH5_weights_alpha_{alpha}_{N_0}_walkers_{j+1}.npy')
    asdf = hh_dist(coords)
    lots_o_weights = np.array([weights[0, :]]*10)
    flat_weights = lots_o_weights.flatten()
    amp, xx = np.histogram(asdf/ang2bohr, weights=flat_weights, bins=bining, range=(0.5, 2.5), density=True)
    amp_all[:, j] += amp
bins = (xx[1:] + xx[:-1])/2.
plt.plot(bins, np.mean(amp_all, axis=1))
plt.savefig(f'HH_distances/alpha_{alpha}/hh_dist_projection_alpha_{alpha}_{N_0}_walkers.png')
plt.close()

amp_all = np.zeros((bining, 5))
for j in range(5):
    coords = np.load(f'Non_imp_sampled/DMC_CH5_randomly_sampled_coords_{N_0}_walkers_{j+1}.npy')
    weights = np.load(f'Non_imp_sampled/DMC_CH5_randomly_sampled_weights_{N_0}_walkers_{j+1}.npy')
    asdf = hh_dist(coords)
    lots_o_weights = np.array([weights[0, :]]*10)
    flat_weights = lots_o_weights.flatten()
    amp, xx = np.histogram(asdf/ang2bohr, weights=flat_weights, bins=bining, range=(0.5, 2.5), density=True)
    amp_all[:, j] += amp
bins = (xx[1:] + xx[:-1])/2.
plt.plot(bins, np.mean(amp_all, axis=1))
plt.savefig(f'HH_distances/non_imp_samp/hh_dist_projection_non_imp_samp_{N_0}_walkers.png')
plt.close()




