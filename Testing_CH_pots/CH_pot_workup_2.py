import numpy as np
import matplotlib.pyplot as plt

har2wave = 219474.6

min_mean = np.zeros((3, 5, 25))
min_std = np.zeros((3, 5, 25))
cs_mean = np.zeros((3, 5, 25))
cs_std = np.zeros((3, 5, 25))
c2v_mean = np.zeros((3, 5, 25))
c2v_std = np.zeros((3, 5, 25))

DVR_correct = np.array([[1488.113887, 1218.705882, 1224.753528, 1577.102303, 1577.102303],
                        [1626.480859, 1242.570938, 1242.570937, 1554.300194, 1554.300197],
                        [1670.720578, 1919.949044, 1919.949044, 1713.462511, 1713.462511]])

DMC_NIS_mean = np.array([[1487.808809, 1219.11116, 1224.977963, 1576.08457, 1578.273878],
                         [1626.902877, 1241.245382, 1241.122367, 1555.912652, 1555.832413],
                         [1671.779012, 1919.835793, 1919.602008, 1713.789883, 1714.563884]])

DMC_NIS_std = np.array([[3.384906654, 2.891356861, 2.486500513, 3.630803896, 2.97606898],
                        [2.432287891, 3.080632152, 2.728463369, 2.715065554, 4.11127682],
                        [2.517015584, 3.587758239, 3.258845895, 3.151204422, 2.211255841]])

min_avg_mean = np.array([[1488.841196, 1218.870477, 1226.80496, 1578.53795, 1579.23167],
                         [1629.004676, 1246.545113, 1245.6346, 1554.464394, 1557.041201],
                         [1670.913973, 1918.212943, 1920.91641, 1714.287856, 1714.2216]])

min_avg_std = np.array([[1.892487293, 4.950044284, 3.520631584, 3.578217554, 4.000050559],
                        [3.334903835, 2.734438471, 2.835197691, 2.219589131, 2.370601791],
                        [1.480962048, 2.915919779, 1.081169247, 1.118778114, 0.5879482001]])

cs_avg_mean = np.array([[1489.669353, 1220.571004, 1226.004738, 1575.835836, 1578.253036],
                        [1629.022312, 1244.324199, 1245.630058, 1555.487862, 1556.020127],
                        [1670.131028, 1923.139041, 1922.654189, 1713.303494, 1713.842955]])

cs_avg_std = np.array([[1.443539592, 1.828303184, 4.844162027, 1.254558203, 2.277342155],
                       [4.18311821, 1.399238635, 4.136594295, 1.041410196, 2.502471051],
                       [2.037188058, 2.461859611, 2.682077174, 0.4551682817, 1.692408828]])

c2v_avg_mean = np.array([[1488.098908, 1220.476282, 1225.737411, 1577.576425, 1578.46744],
                        [1628.147142, 1247.357994, 1245.979311, 1555.14488, 1556.85983],
                        [1670.917949, 1920.034879, 1919.880642, 1713.811871, 1712.97294]])

c2v_avg_std = np.array([[0.9149840664, 5.105362483, 2.237728779, 2.637779663, 2.537770439],
                        [3.746882225, 8.33968364, 7.836876918, 0.5170682703, 8.707869296],
                        [3.678271741, 2.277934333, 3.062818096, 1.485448541, 1.253955976]])

energy1 = np.load('Imp_min_energies_min_low.npy')*har2wave
energy2 = np.load('Imp_cs_energies_min_low.npy')*har2wave
energy3 = np.load('Imp_c2v_energies_min_low.npy')*har2wave

energy1cs = np.load('Imp_min_energies_cs_low.npy')*har2wave
energy2cs = np.load('Imp_cs_energies_cs_low.npy')*har2wave
energy3cs = np.load('Imp_c2v_energies_cs_low.npy')*har2wave

energy1c2v = np.load('Imp_min_energies_c2v_low.npy')*har2wave
energy2c2v = np.load('Imp_cs_energies_c2v_low.npy')*har2wave
energy3c2v = np.load('Imp_c2v_energies_c2v_low.npy')*har2wave

for j in range(5):
    for l in range(8):
        min_mean[0, j, l] += np.mean(energy1[:, j, l])
        min_std[0, j, l] += np.std(energy1[:, j, l])

        min_mean[1, j, l] += np.mean(energy2[:, j, l])
        min_std[1, j, l] += np.std(energy2[:, j, l])

        min_mean[2, j, l] += np.mean(energy3[:, j, l])
        min_std[2, j, l] += np.std(energy3[:, j, l])

        cs_mean[0, j, l] += np.mean(energy1cs[:, j, l])
        cs_std[0, j, l] += np.std(energy1cs[:, j, l])

        cs_mean[1, j, l] += np.mean(energy2cs[:, j, l])
        cs_std[1, j, l] += np.std(energy2cs[:, j, l])

        cs_mean[2, j, l] += np.mean(energy3cs[:, j, l])
        cs_std[2, j, l] += np.std(energy3cs[:, j, l])

        c2v_mean[0, j, l] += np.mean(energy1c2v[:, j, l])
        c2v_std[0, j, l] += np.std(energy1c2v[:, j, l])

        c2v_mean[1, j, l] += np.mean(energy2c2v[:, j, l])
        c2v_std[1, j, l] += np.std(energy2c2v[:, j, l])

        c2v_mean[2, j, l] += np.mean(energy3c2v[:, j, l])
        c2v_std[2, j, l] += np.std(energy3c2v[:, j, l])

energy1 = np.load('Imp_min_energies.npy')*har2wave
energy2 = np.load('Imp_cs_energies_min_low.npy')*har2wave
energy3 = np.load('Imp_c2v_energies_min_low.npy')*har2wave

energy1cs = np.load('Imp_min_energies_cs.npy')*har2wave
energy2cs = np.load('Imp_cs_energies_cs.npy')*har2wave
energy3cs = np.load('Imp_c2v_energies_cs.npy')*har2wave

energy1c2v = np.load('Imp_min_energies_c2v.npy')*har2wave
energy2c2v = np.load('Imp_cs_energies_c2v.npy')*har2wave
energy3c2v = np.load('Imp_c2v_energies_c2v.npy')*har2wave

for j in range(5):
    for l in range(7):
        min_mean[0, j, l+8] += np.mean(energy1[:, j, l])
        min_std[0, j, l+8] += np.std(energy1[:, j, l])

        min_mean[1, j, l+8] += np.mean(energy2[:, j, l])
        min_std[1, j, l+8] += np.std(energy2[:, j, l])

        min_mean[2, j, l+8] += np.mean(energy3[:, j, l])
        min_std[2, j, l+8] += np.std(energy3[:, j, l])

        cs_mean[0, j, l+8] += np.mean(energy1cs[:, j, l])
        cs_std[0, j, l+8] += np.std(energy1cs[:, j, l])

        cs_mean[1, j, l+8] += np.mean(energy2cs[:, j, l])
        cs_std[1, j, l+8] += np.std(energy2cs[:, j, l])

        cs_mean[2, j, l+8] += np.mean(energy3cs[:, j, l])
        cs_std[2, j, l+8] += np.std(energy3cs[:, j, l])

        c2v_mean[0, j, l+8] += np.mean(energy1c2v[:, j, l])
        c2v_std[0, j, l+8] += np.std(energy1c2v[:, j, l])

        c2v_mean[1, j, l+8] += np.mean(energy2c2v[:, j, l])
        c2v_std[1, j, l+8] += np.std(energy2c2v[:, j, l])

        c2v_mean[2, j, l+8] += np.mean(energy3c2v[:, j, l])
        c2v_std[2, j, l+8] += np.std(energy3c2v[:, j, l])

energy1 = np.load('Imp_min_energies_min_high.npy')*har2wave
energy2 = np.load('Imp_cs_energies_min_high.npy')*har2wave
energy3 = np.load('Imp_c2v_energies_min_high.npy')*har2wave

energy1cs = np.load('Imp_min_energies_cs_high.npy')*har2wave
energy2cs = np.load('Imp_cs_energies_cs_high.npy')*har2wave
energy3cs = np.load('Imp_c2v_energies_cs_high.npy')*har2wave

energy1c2v = np.load('Imp_min_energies_c2v_high.npy')*har2wave
energy2c2v = np.load('Imp_cs_energies_c2v_high.npy')*har2wave
energy3c2v = np.load('Imp_c2v_energies_c2v_high.npy')*har2wave

for j in range(5):
    for l in range(10):
        min_mean[0, j, l+15] += np.mean(energy1[:, j, l])
        min_std[0, j, l+15] += np.std(energy1[:, j, l])

        min_mean[1, j, l+15] += np.mean(energy2[:, j, l])
        min_std[1, j, l+15] += np.std(energy2[:, j, l])

        min_mean[2, j, l+15] += np.mean(energy3[:, j, l])
        min_std[2, j, l+15] += np.std(energy3[:, j, l])

        cs_mean[0, j, l+15] += np.mean(energy1cs[:, j, l])
        cs_std[0, j, l+15] += np.std(energy1cs[:, j, l])

        cs_mean[1, j, l+15] += np.mean(energy2cs[:, j, l])
        cs_std[1, j, l+15] += np.std(energy2cs[:, j, l])

        cs_mean[2, j, l+15] += np.mean(energy3cs[:, j, l])
        cs_std[2, j, l+15] += np.std(energy3cs[:, j, l])

        c2v_mean[0, j, l+15] += np.mean(energy1c2v[:, j, l])
        c2v_std[0, j, l+15] += np.std(energy1c2v[:, j, l])

        c2v_mean[1, j, l+15] += np.mean(energy2c2v[:, j, l])
        c2v_std[1, j, l+15] += np.std(energy2c2v[:, j, l])

        c2v_mean[2, j, l+15] += np.mean(energy3c2v[:, j, l])
        c2v_std[2, j, l+15] += np.std(energy3c2v[:, j, l])


sp = np.linspace(0.6, 1.8, 25)
geos = ['Minimum', 'cs Saddle', 'c2v Saddle']
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for l in range(3):
    for j in range(5):
        axes[l][j].errorbar(sp, [DMC_NIS_mean[l, j]] * len(sp), yerr=[DMC_NIS_std[l, j]] * len(sp), color='cyan',
                            label='DMC w/o imp samp CH stretch %s' % (j + 1))
        axes[l][j].errorbar(sp, min_mean[l, j, :],  yerr=min_std[l, j, :], color='C%s' % j,
                            label='DMC CH stretch %s' % (j+1))
        axes[l][j].plot(sp, [DVR_correct[l, j]]*len(sp), 'black', label='DVR CH stretch %s' % (j+1))
        axes[l][j].set_xlabel('Scan Point (Angstroms)')
        axes[l][j].set_ylabel('Ground State Energy (cm^-1)')
        axes[l][j].set_title('Ground State Energy for %s Geometry' % geos[l])
        axes[l][j].legend(loc='lower left')
        axes[l][j].set_ylim(DVR_correct[l, j]-5., DVR_correct[l, j]+5.)
    # axes[l].legend()
plt.tight_layout()
# fig.suptitle('Ground State Energy for ')
fig.savefig('Energy_along_tanh_scan_%s_wvfn.png' % geos[0])
plt.close(fig)

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for l in range(3):
    for j in range(5):
        axes[l][j].errorbar(sp, [DMC_NIS_mean[l, j]]*len(sp), yerr=[DMC_NIS_std[l, j]]*len(sp), color='cyan',
                            label='DMC w/o imp samp CH stretch %s' % (j+1))
        axes[l][j].errorbar(sp, cs_mean[l, j, :], yerr=cs_std[l, j, :], color='C%s' % j,
                            label='DMC CH stretch %s' % (j+1))
        axes[l][j].plot(sp, [DVR_correct[l, j]]*len(sp), color='black', label='DVR CH stretch %s' % (j+1))
        axes[l][j].set_xlabel('Scan Point (Angstroms)')
        axes[l][j].set_ylabel('Ground State Energy (cm^-1)')
        axes[l][j].set_title('Ground State Energy for %s Geometry' % geos[l])
        axes[l][j].legend(loc='lower left')
        axes[l][j].set_ylim(DVR_correct[l, j]-5., DVR_correct[l, j]+5.)
    # axes[l].legend()
plt.tight_layout()
fig.savefig('Energy_along_tanh_scan_%s_wvfn.png' % 'cs_saddle')
plt.close(fig)

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for l in range(3):
    for j in range(5):
        axes[l][j].errorbar(sp, [DMC_NIS_mean[l, j]] * len(sp), yerr=[DMC_NIS_std[l, j]] * len(sp), color='cyan',
                            label='DMC w/o imp samp CH stretch %s' % (j + 1))
        axes[l][j].errorbar(sp, c2v_mean[l, j, :], yerr=c2v_std[l, j, :], color='C%s' % j,
                            label='DMC CH stretch %s' % (j+1))
        axes[l][j].plot(sp, [DVR_correct[l, j]]*len(sp), color='black', label='DVR CH stretch %s' % (j+1))
        axes[l][j].set_xlabel('Scan Point (Angstroms)')
        axes[l][j].set_ylabel('Ground State Energy (cm^-1)')
        axes[l][j].set_title('Ground State Energy for %s Geometry' % geos[l])
        axes[l][j].legend(loc='lower left')
        axes[l][j].set_ylim(DVR_correct[l, j]-5., DVR_correct[l, j]+5.)
    # axes[l].legend()
plt.tight_layout()
fig.savefig('Energy_along_tanh_scan_%s_wvfn.png' % 'c2v_saddle')
plt.close(fig)

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for l in range(3):
    for j in range(5):
        axes[l][j].errorbar(sp, [min_avg_mean[l, j]] * len(sp), yerr=[min_avg_std[l, j]] * len(sp), color='cyan',
                            label='DMC Avg Wvfn CH stretch %s' % (j + 1))
        axes[l][j].errorbar(sp, min_mean[l, j, :],  yerr=min_std[l, j, :], color='C%s' % j,
                            label='DMC CH stretch %s' % (j+1))
        axes[l][j].plot(sp, [DVR_correct[l, j]]*len(sp), 'black', label='DVR CH stretch %s' % (j+1))
        axes[l][j].set_xlabel('Scan Point (Angstroms)')
        axes[l][j].set_ylabel('Ground State Energy (cm^-1)')
        axes[l][j].set_title('Ground State Energy for %s Geometry' % geos[l])
        axes[l][j].legend(loc='lower left')
        axes[l][j].set_ylim(DVR_correct[l, j]-5., DVR_correct[l, j]+5.)
    # axes[l].legend()
plt.tight_layout()
# fig.suptitle('Ground State Energy for ')
fig.savefig('Energy_along_tanh_against_avg_scan_%s_wvfn.png' % geos[0])
plt.close(fig)

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for l in range(3):
    for j in range(5):
        axes[l][j].errorbar(sp, [cs_avg_mean[l, j]] * len(sp), yerr=[cs_avg_std[l, j]] * len(sp), color='cyan',
                            label='DMC Avg Wvfn CH stretch %s' % (j + 1))
        axes[l][j].errorbar(sp, cs_mean[l, j, :], yerr=cs_std[l, j, :], color='C%s' % j,
                            label='DMC CH stretch %s' % (j+1))
        axes[l][j].plot(sp, [DVR_correct[l, j]]*len(sp), color='black', label='DVR CH stretch %s' % (j+1))
        axes[l][j].set_xlabel('Scan Point (Angstroms)')
        axes[l][j].set_ylabel('Ground State Energy (cm^-1)')
        axes[l][j].set_title('Ground State Energy for %s Geometry' % geos[l])
        axes[l][j].legend(loc='lower left')
        axes[l][j].set_ylim(DVR_correct[l, j]-5., DVR_correct[l, j]+5.)
    # axes[l].legend()
plt.tight_layout()
fig.savefig('Energy_along_tanh_against_avg_scan_%s_wvfn.png' % 'cs_saddle')
plt.close(fig)

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for l in range(3):
    for j in range(5):
        axes[l][j].errorbar(sp, [c2v_avg_mean[l, j]] * len(sp), yerr=[c2v_avg_std[l, j]] * len(sp), color='cyan',
                            label='DMC Avg Wvfn CH stretch %s' % (j + 1))
        axes[l][j].errorbar(sp, c2v_mean[l, j, :], yerr=c2v_std[l, j, :], color='C%s' % j,
                            label='DMC CH stretch %s' % (j+1))
        axes[l][j].plot(sp, [DVR_correct[l, j]]*len(sp), color='black', label='DVR CH stretch %s' % (j+1))
        axes[l][j].set_xlabel('Scan Point (Angstroms)')
        axes[l][j].set_ylabel('Ground State Energy (cm^-1)')
        axes[l][j].set_title('Ground State Energy for %s Geometry' % geos[l])
        axes[l][j].legend(loc='lower left')
        axes[l][j].set_ylim(DVR_correct[l, j]-5., DVR_correct[l, j]+5.)
    # axes[l].legend()
plt.tight_layout()
fig.savefig('Energy_along_tanh_against_avg_scan_%s_wvfn.png' % 'c2v_saddle')
plt.close(fig)

# fig, axes = plt.subplots(3, 1, figsize=(8, 10))
# for l in range(3):
#     for j in range(5):
#         axes[l].errorbar()
