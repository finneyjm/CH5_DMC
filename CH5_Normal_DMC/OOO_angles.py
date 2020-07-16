import numpy as np
import matplotlib.pyplot as plt


def dists(coords):
    bonds = [[4, 7], [4, 10]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds)-1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds)-1)]
    dis = np.linalg.norm(cd2-cd1, axis=2)
    return dis


def dists2(coords):
    bonds = [[4, 3], [4, 2]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis

def dists3(coords):
    bonds = [[7, 5], [7, 6], [10, 8], [10, 9]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def angles(coords):
    dist = dists(coords)
    v1 = (coords[:, 3] - coords[:, 6]) / np.broadcast_to(dist[:, 0, None], (len(dist), 3))
    v2 = (coords[:, 3] - coords[:, 9]) / np.broadcast_to(dist[:, 1, None], (len(dist), 3))


    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))

    ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())


    return np.rad2deg(ang1).T


def angles2(coords): # bound HOH angles
    dist = dists2(coords)
    v1 = (coords[:, 3] - coords[:, 2]) / np.broadcast_to(dist[:, 0, None], (len(dist), 3))
    v2 = (coords[:, 3] - coords[:, 1]) / np.broadcast_to(dist[:, 1, None], (len(dist), 3))

    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))

    ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())

    return np.rad2deg(ang1).T


def angles3(coords): # outer HOH angles
    dist = dists3(coords)
    v1 = (coords[:, 6] - coords[:, 4]) / np.broadcast_to(dist[:, 0, None], (len(dist), 3))
    v2 = (coords[:, 6] - coords[:, 5]) / np.broadcast_to(dist[:, 1, None], (len(dist), 3))
    v3 = (coords[:, 9] - coords[:, 7]) / np.broadcast_to(dist[:, 2, None], (len(dist), 3))
    v4 = (coords[:, 9] - coords[:, 8]) / np.broadcast_to(dist[:, 3, None], (len(dist), 3))

    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    v3_new = np.reshape(v3, (v3.shape[0], 1, v3.shape[1]))
    v4_new = np.reshape(v4, (v4.shape[0], v4.shape[1], 1))

    ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    ang2 = np.arccos(np.matmul(v3_new, v4_new).squeeze())

    return np.rad2deg(np.vstack((ang1, ang2)).flatten())

######## OOO angles #######
bins = 60
new_amps = np.zeros((3, bins, 20))
walk = [10000, 20000, 40000]
samp = ['imp_samp_waters', 'non_imp_samp', 'non_imp_samp']
for trial, w, s in zip(range(len(walk)), walk, samp):
    for i in range(20):
        wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{s}/' +
                        f'ptrimer_{s}_{w}_' +
                        f'Walkers_Test_{5}.npz')
        coords = wvfn1['coords'][i]
        coords1 = coords
        des = wvfn1['des'][i]
        weights1 = des
        ang1 = angles(coords1)
        amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
        bins1 = (xx[1:] + xx[:-1]) / 2.
        new_amps[trial, :, i] = amp1
avg = np.average(new_amps, axis=2)
std = np.std(new_amps, axis=2)
samp = ['Guided', 'Unguided', 'Unguided']
colors = ['red', 'blue', 'green']
order = [2, 1, 0]
sub = [-1, 0, 1]
for i in order:
    plt.plot(bins1, avg[i], color='black', linewidth=3)
    plt.plot(bins1, avg[i], color=colors[i], linewidth=2.5, label=fr'{samp[i]} N$_{{\rmw}}$ = {walk[i]}')
    std_x = [110, 120, 130]
    plt.errorbar(std_x[0] - sub[i], avg[i, 20 - sub[i]], yerr=std[i, 20 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[1] - sub[i], avg[i, 30 - sub[i]], yerr=std[i, 30 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[2] - sub[i], avg[i, 40 - sub[i]], yerr=std[i, 40 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[0]-sub[i], avg[i, 20-sub[i]], yerr=std[i, 20-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    plt.errorbar(std_x[1]-sub[i], avg[i, 30-sub[i]], yerr=std[i, 30-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    plt.errorbar(std_x[2]-sub[i], avg[i, 40-sub[i]], yerr=std[i, 40-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)

leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005, 0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
plt.tight_layout()
plt.show()


walkers = 10000

imp_samp = 'imp_samp_waters'
color = 'red'
collect_coords = np.zeros((walkers, 20))
collect_weights = np.zeros((walkers, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    weights1 = des

    ang1 = angles(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [110, 120, 130]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005, 0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
plt.tight_layout()
plt.show()

walkers = 20000

imp_samp = 'non_imp_samp'
color = 'blue'
collect_coords = np.zeros((walkers, 20))
collect_weights = np.zeros((walkers, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    weights1 = des

    ang1 = angles(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [110, 120, 130]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005, 0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
plt.tight_layout()
plt.show()

walkers = 40000

imp_samp = 'non_imp_samp'
color = 'green'
collect_coords = np.zeros((walkers, 20))
collect_weights = np.zeros((walkers, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    weights1 = des

    ang1 = angles(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [110, 120, 130]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005, 0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
plt.tight_layout()
plt.show()

######## bound HOH angle ######
bins = 60
new_amps = np.zeros((3, bins, 20))
walk = [10000, 20000, 40000]
samp = ['imp_samp_waters', 'non_imp_samp', 'non_imp_samp']
for trial, w, s in zip(range(len(walk)), walk, samp):
    for i in range(20):
        wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{s}/' +
                        f'ptrimer_{s}_{w}_' +
                        f'Walkers_Test_{5}.npz')
        coords = wvfn1['coords'][i]
        coords1 = coords
        des = wvfn1['des'][i]
        weights1 = des
        ang1 = angles2(coords1)
        amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
        bins1 = (xx[1:] + xx[:-1]) / 2.
        new_amps[trial, :, i] = amp1
avg = np.average(new_amps, axis=2)
std = np.std(new_amps, axis=2)
samp = ['Guided', 'Unguided', 'Unguided']
colors = ['red', 'blue', 'green']
order = [2, 1, 0]
sub = [-1, 0, 1]
for i in order:
    plt.plot(bins1, avg[i], color='black', linewidth=3)
    plt.plot(bins1, avg[i], color=colors[i], linewidth=2.5, label=fr'{samp[i]} N$_{{\rmw}}$ = {walk[i]}')
    std_x = [110, 120, 130]
    plt.errorbar(std_x[0] - sub[i], avg[i, 20 - sub[i]], yerr=std[i, 20 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[1] - sub[i], avg[i, 30 - sub[i]], yerr=std[i, 30 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[2] - sub[i], avg[i, 40 - sub[i]], yerr=std[i, 40 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[0]-sub[i], avg[i, 20-sub[i]], yerr=std[i, 20-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    plt.errorbar(std_x[1]-sub[i], avg[i, 30-sub[i]], yerr=std[i, 30-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    plt.errorbar(std_x[2]-sub[i], avg[i, 40-sub[i]], yerr=std[i, 40-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)

leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005,  0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmHOH,b}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH,b})$', fontsize=22)
plt.tight_layout()
plt.show()


walkers = 10000

imp_samp = 'imp_samp_waters'
color = 'red'
collect_coords = np.zeros((walkers, 20))
collect_weights = np.zeros((walkers, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    weights1 = des

    ang1 = angles2(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [110, 120, 130]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005,  0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmHOH,b}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH,b})$', fontsize=22)
plt.tight_layout()
plt.show()

walkers = 20000

imp_samp = 'non_imp_samp'
color = 'blue'
collect_coords = np.zeros((walkers, 20))
collect_weights = np.zeros((walkers, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    weights1 = des

    ang1 = angles2(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [110, 120, 130]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005,  0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmHOH,b}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH,b})$', fontsize=22)
plt.tight_layout()
plt.show()

walkers = 40000

imp_samp = 'non_imp_samp'
color = 'green'
collect_coords = np.zeros((walkers, 20))
collect_weights = np.zeros((walkers, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    weights1 = des

    ang1 = angles2(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [110, 120, 130]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005,  0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmHOH,b}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH,b})$', fontsize=22)
plt.tight_layout()
plt.show()


# ######## outer HOH angle ######
bins = 60
new_amps = np.zeros((3, bins, 20))
walk = [10000, 20000, 40000]
samp = ['imp_samp_waters', 'non_imp_samp', 'non_imp_samp']
for trial, w, s in zip(range(len(walk)), walk, samp):
    for i in range(20):
        wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{s}/' +
                        f'ptrimer_{s}_{w}_' +
                        f'Walkers_Test_{5}.npz')
        coords = wvfn1['coords'][i]
        coords1 = coords
        des = wvfn1['des'][i]
        des = np.vstack((des, des)).flatten()
        weights1 = des
        ang1 = angles3(coords1)
        amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(74.5, 134.5), density=True)
        bins1 = (xx[1:] + xx[:-1]) / 2.
        new_amps[trial, :, i] = amp1
avg = np.average(new_amps, axis=2)
std = np.std(new_amps, axis=2)
samp = ['Guided', 'Unguided', 'Unguided']
colors = ['red', 'blue', 'green']
order = [2, 1, 0]
sub = [-1, 0, 1]
for i in order:
    plt.plot(bins1, avg[i], color='black', linewidth=3)
    plt.plot(bins1, avg[i], color=colors[i], linewidth=2.5, label=fr'{samp[i]} N$_{{\rmw}}$ = {walk[i]}')
    std_x = [95, 105, 115]
    plt.errorbar(std_x[0] - sub[i], avg[i, 20 - sub[i]], yerr=std[i, 20 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[1] - sub[i], avg[i, 30 - sub[i]], yerr=std[i, 30 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[2] - sub[i], avg[i, 40 - sub[i]], yerr=std[i, 40 - sub[i]], elinewidth=3.5, color='black',
                 capsize=6.5, capthick=3.5)
    plt.errorbar(std_x[0]-sub[i], avg[i, 20-sub[i]], yerr=std[i, 20-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    plt.errorbar(std_x[1]-sub[i], avg[i, 30-sub[i]], yerr=std[i, 30-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    plt.errorbar(std_x[2]-sub[i], avg[i, 40-sub[i]], yerr=std[i, 40-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)

leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005,  0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmHOH}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH})$', fontsize=22)
plt.tight_layout()
plt.show()


walkers = 10000

imp_samp = 'imp_samp_waters'
color = 'red'
collect_coords = np.zeros((walkers*2, 20))
collect_weights = np.zeros((walkers*2, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    des = np.vstack((des, des)).flatten()
    weights1 = des

    ang1 = angles3(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(74.5, 134.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [95, 105, 115]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005,  0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmHOH}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH})$', fontsize=22)
plt.tight_layout()
plt.show()

walkers = 20000

imp_samp = 'non_imp_samp'
color = 'blue'
collect_coords = np.zeros((walkers*2, 20))
collect_weights = np.zeros((walkers*2, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    des = np.vstack((des, des)).flatten()
    weights1 = des

    ang1 = angles3(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(74.5, 134.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [95, 105, 115]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005,  0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmHOH}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH})$', fontsize=22)
plt.tight_layout()
plt.show()

walkers = 40000

imp_samp = 'non_imp_samp'
color = 'green'
collect_coords = np.zeros((walkers*2, 20))
collect_weights = np.zeros((walkers*2, 20))
amps = np.zeros((bins, 20))

for i in range(20):
    wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
                                 f'ptrimer_{imp_samp}_{walkers}_' +
                                 f'Walkers_Test_{5}.npz')
    coords = wvfn1['coords'][i]
    coords1 = coords
    des = wvfn1['des'][i]
    des = np.vstack((des, des)).flatten()
    weights1 = des

    ang1 = angles3(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(74.5, 134.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    plt.plot(bins1, amp1)
plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
std = np.std(amps, axis=1)
std_x = [95, 105, 115]
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)


leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005,  0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmHOH}$', fontsize=22)
plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH})$', fontsize=22)
plt.tight_layout()
plt.show()