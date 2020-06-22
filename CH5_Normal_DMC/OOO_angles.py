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


bins = 60
# new_amps = np.zeros((3, bins, 20))
# walk = [10000, 20000, 40000]
# samp = ['full', 'non', 'non']
# for trial, w, s in zip(range(len(walk)), walk, samp):
#     for i in range(20):
#         wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{s}_imp_samp/' +
#                         f'ptrimer_{s}_imp_samp_{w}_' +
#                         f'Walkers_Test_{5}.npz')
#         coords = wvfn1['coords'][i]
#         coords1 = coords
#         des = wvfn1['des'][i]
#         weights1 = des
#         ang1 = angles(coords1)
#         amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
#         bins1 = (xx[1:] + xx[:-1]) / 2.
#         new_amps[trial, :, i] = amp1
# avg = np.average(new_amps, axis=2)
# std = np.std(new_amps, axis=2)
# samp = ['Guided', 'Unguided', 'Unguided']
# colors = ['red', 'blue', 'green']
# order = [2, 1, 0]
# sub = [-1, 0, 1]
# for i in order:
#     plt.plot(bins1, avg[i], color=colors[i], linewidth=2.5, label=fr'{samp[i]} N$_{{\rmw}}$ = {walk[i]}')
#     std_x = [110, 120, 130]
#     plt.errorbar(std_x[0]-sub[i], avg[i, 20-sub[i]], yerr=std[i, 20-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
#     plt.errorbar(std_x[1]-sub[i], avg[i, 30-sub[i]], yerr=std[i, 30-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
#     plt.errorbar(std_x[2]-sub[i], avg[i, 40-sub[i]], yerr=std[i, 40-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
# leg = plt.legend(loc='upper left', fontsize=14)
# leg.get_frame().set_edgecolor('white')
# plt.ylim(-0.005, 0.125)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                 bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
# plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
# plt.tight_layout()
# plt.show()


walkers = 40000

imp_samp = 'non'
collect_coords = np.zeros((walkers, 20))
collect_weights = np.zeros((walkers, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}_imp_samp/' +
                                 f'ptrimer_{imp_samp}_imp_samp_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    # coords1 = np.reshape(coords, (coords.shape[0]*coords.shape[1], coords.shape[2], coords.shape[3]))
    coords1 = coords
    des = wvfn1['des'][i]
    weights1 = des
    # weights1 = np.reshape(des, (des.shape[0]*des.shape[1]))
    ang1 = angles(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1
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

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    # amp2, xx = np.histogram(ang2, weights=weights2, bins=40, density=True)
    # bins2 = (xx[1:] + xx[:-1]) / 2.

    # amp3, xx = np.histogram(ang3, weights=weights3, bins=40, density=True)
    # bins3 = (xx[1:] + xx[:-1]) / 2.
    # if i % 3 == 0:
    plt.plot(bins1, amp1)
# amp1, xx = np.histogram(collect_coords.flatten(), weights=collect_weights.flatten(), bins=75, density=True)
# bins1 = (xx[1:] + xx[:-1]) / 2.
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=2.5, label=fr'Average')
std = np.std(amps, axis=1)
std_x = [110, 120, 130]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color='black', zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color='black', zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color='black', zorder=3)

leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005, 0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.plot(bins2, amp2, label='waters imp. samp.')
# plt.plot(bins3, amp3, label='Non imp. samp.')
# plt.legend()
plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
plt.tight_layout()
plt.show()