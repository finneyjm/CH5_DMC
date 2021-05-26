import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
m_OD = (m_O*m_D)/(m_D+m_O)
m_OH = (m_O*m_H)/(m_H+m_O)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

wvfn = np.load("../lets_go_girls/jobs/Prot_water_params/wvfns/free_oh_wvfn.npy")
free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)


def psi_t_extra(coords, atoms, num_waters, interp_reg_oh, interp_hbond=None, interp_OO_shift=None,
                interp_OO_scale=None, interp_ang=None, reg_oh=None, hbond_oh=None, hbond_oo=None, angs=None):

    def angle_function(angs, interp, atoms):
        if interp is None:
            r1 = 0.95784 * ang2bohr
            r2 = 0.95783997 * ang2bohr
            theta = np.deg2rad(104.5080029)
        else:
            r1 = interp['r1']
            r2 = interp['r2']
            theta = interp['theta']
        muH = 1 / m_H
        muD = 1 / m_D
        muO = 1 / m_O
        if atoms[0].upper() == 'H':
            if atoms[1].upper() == 'H':
                G = gmat(muH, muH, muO, r1, r2, theta)
                freq = 1668.4590610594878
            else:
                G = gmat(muH, muD, muO, r1, r2, theta)
                freq = 1462.5810039828614
        else:
            if atoms[1].upper() == 'H':
                G = gmat(muD, muH, muO, r1, r2, theta)
                freq = 1462.5810039828614
            else:
                G = gmat(muD, muD, muO, r1, r2, theta)
                freq = 1222.5100195873742

        freq /= har2wave
        alpha = freq / G
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2)

    def gmat(mu1, mu2, mu3, r1, r2, ang):
        return mu1 / r1 ** 2 + mu2 / r2 ** 2 + mu3 * (1 / r1 ** 2 + 1 / r2 ** 2 - 2 * np.cos(ang) / (r1 * r2))

    def angles(coords, dists, num_waters):
        if num_waters == 2:
            v1 = (coords[:, 0] - coords[:, 3]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
            v2 = (coords[:, 1] - coords[:, 3]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))
            v3 = (coords[:, 4] - coords[:, 6]) / np.broadcast_to(dists[:, 2, None], (len(dists), 3))
            v4 = (coords[:, 5] - coords[:, 6]) / np.broadcast_to(dists[:, 3, None], (len(dists), 3))

            v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
            v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
            v3_new = np.reshape(v3, (v3.shape[0], 1, v3.shape[1]))
            v4_new = np.reshape(v4, (v4.shape[0], v4.shape[1], 1))

            ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())
            ang2 = np.arccos(np.matmul(v3_new, v4_new).squeeze())

            return np.vstack((ang1, ang2)).T

        elif num_waters == 0:
            v1 = (coords[:, 0] - coords[:, 2]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
            v2 = (coords[:, 1] - coords[:, 2]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))

            v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
            v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))

            ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())

            return ang1.T

        elif num_waters == 3:
            v1 = (coords[:, 4] - coords[:, 6]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))
            v2 = (coords[:, 5] - coords[:, 6]) / np.broadcast_to(dists[:, 2, None], (len(dists), 3))
            v3 = (coords[:, 7] - coords[:, 9]) / np.broadcast_to(dists[:, 3, None], (len(dists), 3))
            v4 = (coords[:, 8] - coords[:, 9]) / np.broadcast_to(dists[:, 4, None], (len(dists), 3))

            v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
            v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
            v3_new = np.reshape(v3, (v3.shape[0], 1, v3.shape[1]))
            v4_new = np.reshape(v4, (v4.shape[0], v4.shape[1], 1))

            ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())
            ang2 = np.arccos(np.matmul(v3_new, v4_new).squeeze())

            return np.vstack((ang1, ang2)).T

        elif num_waters == 4:
            v1 = (coords[:, 4] - coords[:, 6]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
            v2 = (coords[:, 5] - coords[:, 6]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))
            v3 = (coords[:, 7] - coords[:, 9]) / np.broadcast_to(dists[:, 2, None], (len(dists), 3))
            v4 = (coords[:, 8] - coords[:, 9]) / np.broadcast_to(dists[:, 3, None], (len(dists), 3))
            v5 = (coords[:, 10] - coords[:, 12]) / np.broadcast_to(dists[:, 4, None], (len(dists), 3))
            v6 = (coords[:, 11] - coords[:, 12]) / np.broadcast_to(dists[:, 5, None], (len(dists), 3))

            v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
            v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
            v3_new = np.reshape(v3, (v3.shape[0], 1, v3.shape[1]))
            v4_new = np.reshape(v4, (v4.shape[0], v4.shape[1], 1))
            v5_new = np.reshape(v5, (v5.shape[0], 1, v5.shape[1]))
            v6_new = np.reshape(v6, (v6.shape[0], v6.shape[1], 1))

            ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())
            ang2 = np.arccos(np.matmul(v3_new, v4_new).squeeze())
            ang3 = np.arccos(np.matmul(v5_new, v6_new).squeeze())

            return np.vstack((ang1, ang2, ang3)).T

    if reg_oh is None:
        reg_oh = dists(coords, num_waters, 'OH')
        if num_waters > 1:
            hbond_oh = dists(coords, num_waters, 'hbond_OH')
            hbond_oo = dists(coords, num_waters, 'hbond_OO')
            angs = angles(coords, reg_oh, num_waters)
        elif num_waters == 0:
            angs = angles(coords, reg_oh, num_waters)

    if num_waters > 1:
        shift = np.zeros((len(coords), int(num_waters-1)))
        scale = np.zeros((len(coords), int(num_waters-1)))
    if num_waters == 2:
        psi = np.zeros((len(coords), 7))
    elif num_waters == 0:
        psi = np.zeros((len(coords), 3))
    else:
        psi = np.zeros((len(coords), int(num_waters*3)))

    if num_waters == 1:
        for i in range(3):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
    elif num_waters == 0:
        for i in range(2):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
        psi[:, 2] = angle_function(angs, interp_ang, atoms[1:3])
    elif num_waters == 2:
        for i in range(4):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
        if interp_hbond is None:
            psi[:, 4] = np.ones((len(coords)))
        else:
            shift = shift_calc(hbond_oo, interp_OO_shift)
            scale = scale_calc(hbond_oo, interp_OO_scale)
            psi[:, 4] = interpolate.splev(scale*(hbond_oh-shift), interp_hbond, der=0)
        for k in range(2):
            a = k*2 + k + 1
            psi[:, k+5] = angle_function(angs[:, k], interp_ang, atoms[a:a+2])
    elif num_waters == 3:
        for i in range(5):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
        for j in range(2):
            if interp_hbond is None:
                psi[:, j+5] = np.ones((len(coords)))
            else:
                shift[:, j] = shift_calc(hbond_oo[:, j], interp_OO_shift)
                scale[:, j] = scale_calc(hbond_oo[:, j], interp_OO_scale)
                psi[:, j+5] = interpolate.splev(scale[:, j]*(hbond_oh[:, j]-shift[:, j]), interp_hbond, der=0)
        for k in range(2):
            p = k+1
            a = p*2 + p + 1
            psi[:, k+5+2] = angle_function(angs[:, k], interp_ang, atoms[a:a+2])
    elif num_waters == 4:
        for i in range(6):
            psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
        for j in range(3):
            if interp_hbond is None:
                psi[:, j+6] = np.ones((len(coords)))
            else:
                shift[:, j] = shift_calc(hbond_oo[:, j], interp_OO_shift)
                scale[:, j] = scale_calc(hbond_oo[:, j], interp_OO_scale)
                psi[:, j+6] = interpolate.splev(scale[:, j]*(hbond_oh[:, j]-shift[:, j]), interp_hbond, der=0)
        for k in range(3):
            p = k + 1
            a = p * 2 + p + 1
            psi[:, k+6+3] = angle_function(angs[:, k], interp_ang, atoms[a:a+2])
    return psi

def dists(coords, num_waters, dist_type):
    if num_waters == 1:
        bonds = [[4, 1], [4, 2], [4, 3]]
    elif num_waters == 0:
        bonds = [[3, 1], [3, 2]]
    elif num_waters == 2:
        if dist_type == 'OH':
            bonds = [[4, 2], [4, 3], [7, 6], [7, 5]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 1]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7]]
    elif num_waters == 3:
        if dist_type == 'OH':
            bonds = [[4, 1], [7, 6], [7, 5], [10, 9], [10, 8]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 3], [4, 2]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7], [4, 10]]
        elif dist_type == 'OO':
            bonds = [[7, 10]]
    elif num_waters == 4:
        if dist_type == 'OH':
            bonds = [[7, 6], [7, 5], [10, 9], [10, 8], [13, 12], [13, 11]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 3], [4, 2], [4, 1]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7], [4, 13], [4, 10]]
        elif dist_type == 'OO':
            bonds = [[7, 10], [7, 13], [10, 13]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds)-1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds)-1)]
    dis = np.linalg.norm(cd2-cd1, axis=2)
    return dis


def shift_calc(oo_dists, interp):
    if interp is None:
        return np.zeros(oo_dists.shape)
    else:
        f = np.poly1d(interp)
        oh_max = f(oo_dists)
        return oh_max


def scale_calc(oo_dists, interp):
    if interp is None:
        return np.ones(oo_dists.shape)
    else:
        f = np.poly1d(interp)
        oh_std = f(oo_dists)
        return oh_std


def distsoo(coords):
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
    dist = distsoo(coords)
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
walk_leg = ['10 000', '20 000', '40 000']
samp = ['imp_samp_waters', 'non_imp_samp', 'non_imp_samp']
for trial, w, s in zip(range(len(walk)), walk, samp):
    for i in range(20):
        wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{s}/' +
                        f'ptrimer_{s}_{w}_' +
                        f'Walkers_Test_{5}.npz')
        coords = wvfn1['coords'][i]
        coords1 = coords
        if s == 'imp_samp_waters':
            des = wvfn1['weights'][i]
            psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn),
                          axis=-1)
            # psi = np.vstack((psi, psi)).flatten()
        else:
            des = wvfn1['des'][i]
        weights1 = des
        ang1 = angles(coords1)
        if s == 'imp_samp_waters':
            amp1, xx = np.histogram(ang1, weights=(2*weights1 - psi**2), bins=bins, range=(89.5, 149.5), density=True)
        else:
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
    plt.plot(bins1, avg[i], color=colors[i], linewidth=2.5, label=fr'{samp[i]} N$_{{\rmw}}$ = {walk_leg[i]}')
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
    # plt.plot(110, 0.12, label='Averages')

leg = plt.legend(loc='upper left', fontsize=14)
leg.get_frame().set_edgecolor('white')
plt.ylim(-0.005, 0.125)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, labelsize=14)
plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
plt.ylabel(r'$\rmP(\rm\theta_{\rmOOO})$', fontsize=22)
plt.tight_layout()
plt.savefig('OOO_all_averages_2f.png')
# plt.show()
plt.close()


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
    des = wvfn1['weights'][i]
    weights1 = des

    psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn),
                  axis=-1)
    # psi = np.vstack((psi, psi)).flatten()

    ang1 = angles(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=(weights1), bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    # plt.plot(bins1, amp1)
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

    # psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn),
    #               axis=-1)
    # psi = np.vstack((psi, psi)).flatten()

    ang1 = angles(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=(weights1), bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    # plt.plot(bins1, amp1)
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
plt.savefig('OOO_non_imp_samp_f_and_descend.png')
# plt.show()
plt.close()

# walkers = 20000
#
# imp_samp = 'non_imp_samp'
# color = 'blue'
# collect_coords = np.zeros((walkers, 20))
# collect_weights = np.zeros((walkers, 20))
# amps = np.zeros((bins, 20))
#
# for i in range(20):
#     wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
#                                  f'ptrimer_{imp_samp}_{walkers}_' +
#                                  f'Walkers_Test_{5}.npz')
#     coords = wvfn1['coords'][i]
#     coords1 = coords
#     des = wvfn1['weights'][i]
#     weights1 = des
#
#     ang1 = angles(coords1)
#     collect_coords[:, i] = ang1
#     collect_weights[:, i] = weights1
#
#     amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
#     bins1 = (xx[1:] + xx[:-1]) / 2.
#     amps[:, i] = amp1
#     plt.plot(bins1, amp1)
# plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
# plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
# std = np.std(amps, axis=1)
# std_x = [110, 120, 130]
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
#
#
# leg = plt.legend(loc='upper left', fontsize=14)
# leg.get_frame().set_edgecolor('white')
# plt.ylim(-0.005, 0.125)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                 bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
# plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
# plt.tight_layout()
# plt.savefig('OOO_no_imp_samp_20000_f.png')
# plt.show()

# walkers = 20000
#
# imp_samp = 'non_imp_samp_ts_10'
# color = 'purple'
# collect_coords = np.zeros((walkers, 20))
# collect_weights = np.zeros((walkers, 20))
# amps = np.zeros((bins, 20))
#
# for i in range(20):
#     wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
#                                  f'ptrimer_{imp_samp}_{walkers}_' +
#                                  f'Walkers_Test_{5}.npz')
#     coords = wvfn1['coords'][i]
#     coords1 = coords
#     des = wvfn1['des'][i]
#     weights1 = des
#
#     ang1 = angles(coords1)
#     collect_coords[:, i] = ang1
#     collect_weights[:, i] = weights1
#
#     amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
#     bins1 = (xx[1:] + xx[:-1]) / 2.
#     amps[:, i] = amp1
#     plt.plot(bins1, amp1)
# plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
# plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
# std = np.std(amps, axis=1)
# std_x = [110, 120, 130]
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
#
#
# leg = plt.legend(loc='upper left', fontsize=14)
# leg.get_frame().set_edgecolor('white')
# plt.ylim(-0.005, 0.125)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                 bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
# plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
# plt.tight_layout()
# plt.savefig('OOO_no_imp_samp_20000_ts_10.png')
# plt.show()
#
# walkers = 40000
#
# imp_samp = 'non_imp_samp'
# color = 'green'
# collect_coords = np.zeros((walkers, 20))
# collect_weights = np.zeros((walkers, 20))
# amps = np.zeros((bins, 20))
#
# for i in range(20):
#     wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
#                                  f'ptrimer_{imp_samp}_{walkers}_' +
#                                  f'Walkers_Test_{5}.npz')
#     coords = wvfn1['coords'][i]
#     coords1 = coords
#     des = wvfn1['weights'][i]
#     weights1 = des
#
#     ang1 = angles(coords1)
#     collect_coords[:, i] = ang1
#     collect_weights[:, i] = weights1
#
#     amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
#     bins1 = (xx[1:] + xx[:-1]) / 2.
#     amps[:, i] = amp1
#     plt.plot(bins1, amp1)
# plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
# plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
# std = np.std(amps, axis=1)
# std_x = [110, 120, 130]
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
#
#
# leg = plt.legend(loc='upper left', fontsize=14)
# leg.get_frame().set_edgecolor('white')
# plt.ylim(-0.005, 0.125)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                 bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.xlabel(r'$\rm\theta_{\rmOOO}$', fontsize=22)
# plt.ylabel(r'$\Psi(\rm\theta_{\rmOOO})$', fontsize=22)
# plt.tight_layout()
# plt.savefig('OOO_no_imp_samp_40000_f.png')
# # plt.show()
#
# ######## bound HOH angle ######
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
        if s == 'imp_samp_waters':
            des = wvfn1['weights'][i]
            psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn),
                          axis=-1)
            # psi = np.vstack((psi, psi)).flatten()
        else:
            des = wvfn1['des'][i]
        weights1 = des
        ang1 = angles2(coords1)
        if s == 'imp_samp_waters':
            amp, xx = np.histogram(ang1, weights=(2*weights1 - psi**2), bins=bins, range=(89.5, 149.5), density=True)
        else:
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
plt.savefig('HOHb_all_averages_2f.png')
# plt.show()
plt.close()


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
    des = wvfn1['weights'][i]
    weights1 = des

    psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn),
                  axis=-1)
    # psi = np.vstack((psi, psi)).flatten()

    ang1 = angles2(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=(weights1), bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    # plt.plot(bins1, amp1)
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

    psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn),
                  axis=-1)
    # psi = np.vstack((psi, psi)).flatten()

    ang1 = angles2(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    amp1, xx = np.histogram(ang1, weights=(weights1), bins=bins, range=(89.5, 149.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    # plt.plot(bins1, amp1)
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
plt.savefig('HOHb_imp_samp_f_and_descend.png')
# plt.show()
plt.close()

# walkers = 20000
#
# imp_samp = 'non_imp_samp'
# color = 'blue'
# collect_coords = np.zeros((walkers, 20))
# collect_weights = np.zeros((walkers, 20))
# amps = np.zeros((bins, 20))
#
# for i in range(20):
#     wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
#                                  f'ptrimer_{imp_samp}_{walkers}_' +
#                                  f'Walkers_Test_{5}.npz')
#     coords = wvfn1['coords'][i]
#     coords1 = coords
#     des = wvfn1['weights'][i]
#     weights1 = des
#
#     ang1 = angles2(coords1)
#     collect_coords[:, i] = ang1
#     collect_weights[:, i] = weights1
#
#     amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
#     bins1 = (xx[1:] + xx[:-1]) / 2.
#     amps[:, i] = amp1
#     plt.plot(bins1, amp1)
# plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
# plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
# std = np.std(amps, axis=1)
# std_x = [110, 120, 130]
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
#
#
# leg = plt.legend(loc='upper left', fontsize=14)
# leg.get_frame().set_edgecolor('white')
# plt.ylim(-0.005,  0.125)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                 bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.xlabel(r'$\rm\theta_{\rmHOH,b}$', fontsize=22)
# plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH,b})$', fontsize=22)
# plt.tight_layout()
# plt.savefig('HOHb_no_imp_samp_20000_f.png')
# # plt.show()
#
# walkers = 40000
#
# imp_samp = 'non_imp_samp'
# color = 'green'
# collect_coords = np.zeros((walkers, 20))
# collect_weights = np.zeros((walkers, 20))
# amps = np.zeros((bins, 20))
#
# for i in range(20):
#     wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
#                                  f'ptrimer_{imp_samp}_{walkers}_' +
#                                  f'Walkers_Test_{5}.npz')
#     coords = wvfn1['coords'][i]
#     coords1 = coords
#     des = wvfn1['weights'][i]
#     weights1 = des
#
#     ang1 = angles2(coords1)
#     collect_coords[:, i] = ang1
#     collect_weights[:, i] = weights1
#
#     amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(89.5, 149.5), density=True)
#     bins1 = (xx[1:] + xx[:-1]) / 2.
#     amps[:, i] = amp1
#     plt.plot(bins1, amp1)
# plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
# plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
# std = np.std(amps, axis=1)
# std_x = [110, 120, 130]
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
#
#
# leg = plt.legend(loc='upper left', fontsize=14)
# leg.get_frame().set_edgecolor('white')
# plt.ylim(-0.005,  0.125)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                 bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.xlabel(r'$\rm\theta_{\rmHOH,b}$', fontsize=22)
# plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH,b})$', fontsize=22)
# plt.tight_layout()
# plt.savefig('HOHb_no_imp_samp_40000_f.png')
# plt.show()


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
        if s == 'imp_samp_waters':
            des = wvfn1['weights'][i]
            psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn),
                          axis=-1)
            psi = np.vstack((psi, psi)).flatten()
        else:
            des = wvfn1['des'][i]
        des = np.vstack((des, des)).flatten()
        weights1 = des
        ang1 = angles3(coords1)
        if s == 'imp_samp_waters':
            amp1, xx = np.histogram(ang1, weights=(2*weights1-psi**2), bins=bins, range=(74.5, 134.5), density=True)
        else:
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
plt.savefig('HOHo_all_averages_2f.png')
# plt.show()
plt.close()

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
    des = wvfn1['weights'][i]
    des = np.vstack((des, des)).flatten()
    weights1 = des
    psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn), axis=-1)
    psi = np.vstack((psi, psi)).flatten()
    ang1 = angles3(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    # amp1, xx = np.histogram(ang1, weights=(2*weights1-psi**2), bins=bins, range=(74.5, 134.5), density=True)
    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(74.5, 134.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    # plt.plot(bins1, amp1)
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
    psi = np.prod(psi_t_extra(coords1, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], 3, free_oh_wvfn), axis=-1)
    psi = np.vstack((psi, psi)).flatten()
    ang1 = angles3(coords1)
    collect_coords[:, i] = ang1
    collect_weights[:, i] = weights1

    # amp1, xx = np.histogram(ang1, weights=(2*weights1-psi**2), bins=bins, range=(74.5, 134.5), density=True)
    amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(74.5, 134.5), density=True)
    bins1 = (xx[1:] + xx[:-1]) / 2.
    amps[:, i] = amp1
    # plt.plot(bins1, amp1)
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
plt.savefig('HOHo_imp_samp_f_and_descend.png')
# plt.show()
plt.close()
# walkers = 20000
#
# imp_samp = 'non_imp_samp'
# color = 'blue'
# collect_coords = np.zeros((walkers*2, 20))
# collect_weights = np.zeros((walkers*2, 20))
# amps = np.zeros((bins, 20))
#
# for i in range(20):
#     wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
#                                  f'ptrimer_{imp_samp}_{walkers}_' +
#                                  f'Walkers_Test_{5}.npz')
#     coords = wvfn1['coords'][i]
#     coords1 = coords
#     des = wvfn1['weights'][i]
#     des = np.vstack((des, des)).flatten()
#     weights1 = des
#
#     ang1 = angles3(coords1)
#     collect_coords[:, i] = ang1
#     collect_weights[:, i] = weights1
#
#     amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(74.5, 134.5), density=True)
#     bins1 = (xx[1:] + xx[:-1]) / 2.
#     amps[:, i] = amp1
#     plt.plot(bins1, amp1)
# plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
# plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
# std = np.std(amps, axis=1)
# std_x = [95, 105, 115]
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
#
#
# leg = plt.legend(loc='upper left', fontsize=14)
# leg.get_frame().set_edgecolor('white')
# plt.ylim(-0.005,  0.125)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                 bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.xlabel(r'$\rm\theta_{\rmHOH}$', fontsize=22)
# plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH})$', fontsize=22)
# plt.tight_layout()
# plt.savefig('HOHo_no_imp_samp_20000_f.png')
# # plt.show()
#
# walkers = 40000
#
# imp_samp = 'non_imp_samp'
# color = 'green'
# collect_coords = np.zeros((walkers*2, 20))
# collect_weights = np.zeros((walkers*2, 20))
# amps = np.zeros((bins, 20))
#
# for i in range(20):
#     wvfn1 = np.load(f'Trial_wvfn_testing/results/ptrimer_{imp_samp}/' +
#                                  f'ptrimer_{imp_samp}_{walkers}_' +
#                                  f'Walkers_Test_{5}.npz')
#     coords = wvfn1['coords'][i]
#     coords1 = coords
#     des = wvfn1['weights'][i]
#     des = np.vstack((des, des)).flatten()
#     weights1 = des
#
#     ang1 = angles3(coords1)
#     collect_coords[:, i] = ang1
#     collect_weights[:, i] = weights1
#
#     amp1, xx = np.histogram(ang1, weights=weights1, bins=bins, range=(74.5, 134.5), density=True)
#     bins1 = (xx[1:] + xx[:-1]) / 2.
#     amps[:, i] = amp1
#     plt.plot(bins1, amp1)
# plt.plot(bins1, np.average(amps, axis=1), color='black', linewidth=3.5)
# plt.plot(bins1, np.average(amps, axis=1), color=color, linewidth=2.5)
# std = np.std(amps, axis=1)
# std_x = [95, 105, 115]
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=3.5, color='black', capsize=6.5, capthick=3.5, zorder=3)
# plt.errorbar(std_x[0], np.average(amps, axis=1)[20], yerr=std[20], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[1], np.average(amps, axis=1)[30], yerr=std[30], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
# plt.errorbar(std_x[2], np.average(amps, axis=1)[40], yerr=std[40], elinewidth=2.5, color=color, capsize=5, capthick=2.5, zorder=3)
#
#
# leg = plt.legend(loc='upper left', fontsize=14)
# leg.get_frame().set_edgecolor('white')
# plt.ylim(-0.005,  0.125)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                 bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.xlabel(r'$\rm\theta_{\rmHOH}$', fontsize=22)
# plt.ylabel(r'$\Psi(\rm\theta_{\rmHOH})$', fontsize=22)
# plt.tight_layout()
# plt.savefig('HOHo_no_imp_samp_40000_f.png')
# plt.show()