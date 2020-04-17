import numpy as np
import matplotlib.pyplot as plt

def dists(coords):
    bonds = [[4, 7], [4, 10]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds)-1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds)-1)]
    dis = np.linalg.norm(cd2-cd1, axis=2)
    return dis


def angles(coords):
    dist = dists(coords)
    v1 = (coords[:, 3] - coords[:, 6]) / np.broadcast_to(dist[:, 0, None], (len(dist), 3))
    v2 = (coords[:, 3] - coords[:, 9]) / np.broadcast_to(dist[:, 1, None], (len(dist), 3))


    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))

    ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())


    return np.rad2deg(ang1).T

for i in range(20):
    walkers = 10000
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp/' +
                                 f'ptetramer_full_imp_samp_{walkers}_' +
                                 f'Walkers_Test_{1}.npz')
    coords = wvfn1['coords'][i]
    # coords1 = np.reshape(coords, (coords.shape[0]*coords.shape[1], coords.shape[2], coords.shape[3]))
    coords1 = coords
    des = wvfn1['des'][i]
    weights1 = des
    # weights1 = np.reshape(des, (des.shape[0]*des.shape[1]))
    ang1 = angles(coords1)

    # wvfn2 = np.load(f'Trial_wvfn_testing/results/ptrimer_full_imp_samp/' +
    #                              f'ptrimer_full_imp_samp_{walkers}_' +
    #                              f'Walkers_Test_{1}.npz')

    # coords = wvfn2['coords'][:20]
    # coords2 = np.reshape(coords, (coords.shape[0]*coords.shape[1], coords.shape[2], coords.shape[3]))
    # des = wvfn2['des'][:20]
    # weights2 = np.reshape(des, (des.shape[0]*des.shape[1]))
    # ang2 = angles(coords2)
    #
    # wvfn3 = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp/' +
    #                              f'ptrimer_non_imp_samp_{walkers+30000}_' +
    #                              f'Walkers_Test_{1}.npz')
    #
    # coords = wvfn3['coords'][:20]
    # coords3 = np.reshape(coords, (coords.shape[0]*coords.shape[1], coords.shape[2], coords.shape[3]))
    # des = wvfn3['des'][:20]
    # weights3 = np.reshape(des, (des.shape[0]*des.shape[1]))
    # ang3 = angles(coords3)

    amp1, xx = np.histogram(ang1, weights=weights1, bins=40, density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.

    # amp2, xx = np.histogram(ang2, weights=weights2, bins=40, density=True)
    # bins2 = (xx[1:] + xx[:-1]) / 2.

    # amp3, xx = np.histogram(ang3, weights=weights3, bins=40, density=True)
    # bins3 = (xx[1:] + xx[:-1]) / 2.

    plt.plot(bins1, amp1, label='non imp. samp.')
    # plt.plot(bins2, amp2, label='waters imp. samp.')
    # plt.plot(bins3, amp3, label='Non imp. samp.')
    plt.legend()
    plt.xlabel(r'$\theta_{OOO}$')
plt.show()