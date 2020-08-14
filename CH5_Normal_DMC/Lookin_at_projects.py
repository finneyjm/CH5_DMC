import numpy as np
import matplotlib.pyplot as plt
import copy


def rotateBackToFrame(coordz, a, b, c,
                      dips=None):  # use the rotation matrices that I always use to reshape each coordinate back to its reference frame
    # print(coordz[1])
    print('RotatingWalkers')
    numWalkers = coordz.shape[0]
    # translation back to Origin
    o3 = coordz[:, a - 1].reshape(numWalkers, 1, 3)
    trCoordz = copy.deepcopy(coordz - o3)
    if dips is not None:
        dips = dips - o3[:, 0, :]
    # Rotation of O2 to x axis
    o2 = trCoordz[:, b - 1, :].reshape(numWalkers, 1, 3)
    z = o2[:, 0, 2]
    y = o2[:, 0, 1]
    x = o2[:, 0, 0]
    theta = np.arctan2(-1 * z, y)
    alpha = np.arctan2((-1 * (
            y * np.cos(theta) - np.sin(theta) * z)), x)
    stheta = np.sin(theta)
    ctheta = np.cos(theta)
    salpha = np.sin(alpha)
    calpha = np.cos(alpha)
    r1 = np.zeros((len(trCoordz), 3, 3))
    r1[:, 0, :] = np.tile([1, 0, 0], len(trCoordz)).reshape(len(trCoordz), 3)
    r1[:, 1, :] = np.column_stack((np.zeros(len(trCoordz)), ctheta, -1 * stheta))
    r1[:, 2, :] = np.column_stack((np.zeros(len(trCoordz)), stheta, ctheta))
    r2 = np.zeros((len(trCoordz), 3, 3))
    r2[:, 0, :] = np.column_stack((calpha, -1 * salpha, np.zeros(len(trCoordz))))
    r2[:, 1, :] = np.column_stack((salpha, calpha, np.zeros(len(trCoordz))))
    r2[:, 2, :] = np.tile([0, 0, 1], len(trCoordz)).reshape(len(trCoordz), 3)
    rotM = np.matmul(r2, r1)
    xaxtrCoordz = np.matmul(rotM, trCoordz.transpose(0, 2, 1)).transpose(0, 2, 1)
    if dips is not None:
        for i in range(len(dips)):
            dips[i] = np.dot(rotM[i], dips[i])
    # Rotation of O1 to xyplane
    o1 = xaxtrCoordz[:, c - 1, :]
    z = o1[:, 2]
    y = o1[:, 1]
    beta = np.arctan2(-1 * z, y)
    cbeta = np.cos(beta)
    sbeta = np.sin(beta)
    r = np.zeros((len(trCoordz), 3, 3))
    r[:, 0, :] = np.tile([1, 0, 0], len(trCoordz)).reshape(len(trCoordz), 3)
    r[:, 1, :] = np.column_stack((np.zeros(len(trCoordz)), cbeta, -1 * sbeta))
    r[:, 2, :] = np.column_stack((np.zeros(len(trCoordz)), sbeta, cbeta))
    finalCoords = np.matmul(r, xaxtrCoordz.transpose(0, 2, 1)).transpose(0, 2, 1)
    if dips is not None:
        for i in range(len(dips)):
            dips[i] = np.dot(r[i], dips[i])
    if dips is not None:
        return finalCoords, dips
    else:
        return finalCoords

def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, 5))
    for i in range(5):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch


bins = 60
new_amps = np.zeros((2, bins, 20))
walk = [5000, 5000]
walk_leg = ['5000', '5000']
samp = ['imp_sampled_analytic_5H_ts_1', 'HH_to_rCHrCD_5H_GSW2']
for trial, w, s in zip(range(len(walk)), walk, samp):
    for i in range(20):
        wvfn1 = np.load(f'Trial_wvfn_testing/results/{s}/' +
                        f'{s}_{w}_' +
                        f'Walkers_Test_{5}.npz')
        coords = wvfn1['coords'][i]
        coords1 = coords
        if s == 'Non_imp_sampled':
            des = wvfn1['weights'][i]
        else:
            des = wvfn1['des'][i]
        weights1 = des
        # ang1 = angles(coords1)
        dists = ch_dist(coords)[:, 2]
        amp1, xx = np.histogram(dists, weights=weights1, bins=bins, range=(1.2, 3.2), density=True)
        bins1 = (xx[1:] + xx[:-1]) / 2.
        new_amps[trial, :, i] = amp1
avg = np.average(new_amps, axis=2)
std = np.std(new_amps, axis=2)
samp = ['Guided (harmonic)', r'Guided (DVR wave function)']
colors = ['red', 'blue']
order = [0, 1]
# sub = [-1, 0, 1, 2]
for i in order:
    plt.plot(bins1, avg[i], color='black', linewidth=3)
    plt.plot(bins1, avg[i], color=colors[i], linewidth=2.5, label=fr'{samp[i]} N$_{{\rmw}}$ = {walk_leg[i]}')
    # std_x = [110, 120, 130]
    # plt.errorbar(std_x[0] - sub[i], avg[i, 20 - sub[i]], yerr=std[i, 20 - sub[i]], elinewidth=3.5, color='black',
    #              capsize=6.5, capthick=3.5)
    # plt.errorbar(std_x[1] - sub[i], avg[i, 30 - sub[i]], yerr=std[i, 30 - sub[i]], elinewidth=3.5, color='black',
    #              capsize=6.5, capthick=3.5)
    # plt.errorbar(std_x[2] - sub[i], avg[i, 40 - sub[i]], yerr=std[i, 40 - sub[i]], elinewidth=3.5, color='black',
    #              capsize=6.5, capthick=3.5)
    # plt.errorbar(std_x[0]-sub[i], avg[i, 20-sub[i]], yerr=std[i, 20-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    # plt.errorbar(std_x[1]-sub[i], avg[i, 30-sub[i]], yerr=std[i, 30-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    # plt.errorbar(std_x[2]-sub[i], avg[i, 40-sub[i]], yerr=std[i, 40-sub[i]], elinewidth=2.5, color=colors[i], capsize=5, capthick=2.5)
    # plt.plot(110, 0.12, label='Averages')

leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005, 3)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rmr_{\rmCH}$', fontsize=22)
plt.ylabel(r'$\rmP(\rmr_{\rmCH})$', fontsize=22)
plt.tight_layout()
plt.show()


