import numpy as np
import matplotlib.pyplot as plt

ang2bohr = 1.e-10/5.291772106712e-11


def hh_dist(coords):
    N = len(coords)
    hh = np.zeros((N, 5, 4))
    for i in range(4):
        for j in np.arange(i, 4):
            hh[:, i, j] = np.sqrt((coords[:, j + 2, 0] - coords[:, i + 1, 0]) ** 2 +
                                  (coords[:, j + 2, 1] - coords[:, i + 1, 1]) ** 2 +
                                  (coords[:, j + 2, 2] - coords[:, i + 1, 2]) ** 2)
            hh[:, j + 1, i] = hh[:, i, j]
    return hh


N_0 = 20000
alpha = 61
bining = 30
isotop = 5
test_num = 5
ni = ['_CD', '_1H', '_2H', '_3H', '_4H', '']
wvfn_num = -1
atoms = ['D', 'D', 'D', 'D', 'D']
for hs in range(isotop):
    atoms[hs] = 'H'
amp_all = np.zeros((bining, 5))
coords = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled{ni[isotop]}/Non_imp_sampled{ni[isotop]}_{N_0}_Walkers_Test_{test_num}.npz')['coords'][wvfn_num]
weights = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled{ni[isotop]}/Non_imp_sampled{ni[isotop]}_{N_0}_Walkers_Test_{test_num}.npz')['weights'][wvfn_num]
asdf = hh_dist(coords)
for j in range(5):
    lots_o_weights = np.array([weights]*4)
    flat_weights = lots_o_weights.flatten()
    amp, xx = np.histogram(asdf[:, j].flatten('F')/ang2bohr, weights=flat_weights, bins=bining, range=(0.5, 2.5), density=True)
    amp_all[:, j] += amp
bins = (xx[1:] + xx[:-1])/2.
plt.figure(1)
# for i in range(5):
#     plt.plot(bins, amp_all[:, i], label=atoms[i])
plt.plot(bins, np.mean(amp_all, axis=1))
# plt.xlabel(r'r ($\AA$)')
# plt.ylabel('P(r)')
# plt.legend(fontsize=14)
plt.xlabel(r'r$_{HH}$ ($\AA$)', fontsize=16)
plt.ylabel('P(r)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
# plt.legend()
# plt.show()
plt.savefig(f'HH_distances_CH5_averaged.png')
# plt.close()

amp_all = np.zeros((bining, 5))
coords = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_{isotop}H_GSW2/HH_to_rCHrCD_{isotop}H_GSW2_{N_0}_Walkers_Test_{test_num}.npz')['coords'][wvfn_num]
weights = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_{isotop}H_GSW2/HH_to_rCHrCD_{isotop}H_GSW2_{N_0}_Walkers_Test_{test_num}.npz')['weights'][wvfn_num]
asdf = hh_dist(coords)
for j in range(5):
    lots_o_weights = np.array([weights]*4)
    flat_weights = lots_o_weights.flatten()
    amp, xx = np.histogram(asdf[:, j].flatten('F')/ang2bohr, weights=flat_weights, bins=bining, range=(0.5, 2.5), density=True)
    amp_all[:, j] += amp
bins = (xx[1:] + xx[:-1])/2.
# plt.figure(2)
#for i in range(5):
  #  plt.plot(bins, amp_all[:, i], label=atoms[i])
# plt.savefig(f'HH_distances/non_imp_samp/hh_dist_projection_non_imp_samp_{N_0}_walkers.png')
# plt.legend()
# plt.show()
# plt.close()




