import numpy as np

har2wave = 219474.6
ground_erefs = np.zeros((10, 20000))


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    s_oh = 1/np.sqrt(2)*(dis[:, 0]+dis[:, 1])
    mid = dis[:, 2]/2
    sp = mid - dis[:, -1]*np.cos(roh_roo_angle(coords, dis[:, -2], dis[:, -1]))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], s_oh, dis[:, -2], sp)).T


def roh_roo_angle(coords, roo_dist, roh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (coords[:, 1]-coords[:, 0])/np.broadcast_to(roh_dist[:, None], (len(roh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


for i in range(10):
    blah = np.load(f'ground_state_full_h3o2_{i+1}.npz')
    eref = blah['Eref']
    ground_erefs[i] = eref

print(np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_zpe = np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_zpe = np.std(np.mean(ground_erefs[:, 5000:]*har2wave, axis=1))
print(std_zpe)

excite_neg_erefs = np.zeros((5, 20000))
for i in range(5):
    blah = np.load(f'Asym_excite_state_full_h3o2_left_{i+1}.npz')
    eref = blah['Eref']
    excite_neg_erefs[i] = eref

gtg = [0, 1, 2, 3, 4]

print(np.mean(np.mean(excite_neg_erefs[gtg, 5000:], axis=1), axis=0)*har2wave)
average_excite_neg_energy = np.mean(np.mean(excite_neg_erefs[gtg, 5000:], axis=1), axis=0)*har2wave
std_excite_neg_energy = np.std(np.mean(excite_neg_erefs[gtg, 5000:]*har2wave, axis=1))

excite_pos_erefs = np.zeros((5, 20000))
for i in range(5):
    blah = np.load(f'Asym_excite_state_full_h3o2_right_{i+1}.npz')
    eref = blah['Eref']
    excite_pos_erefs[i] = eref

gtg = [0, 1, 2, 3, 4]

print(np.mean(np.mean(excite_pos_erefs[gtg, 5000:], axis=1), axis=0)*har2wave)
average_excite_pos_energy = np.mean(np.mean(excite_pos_erefs[gtg, 5000:], axis=1), axis=0)*har2wave
std_excite_pos_energy = np.std(np.mean(excite_pos_erefs[gtg, 5000:]*har2wave, axis=1))

print(average_excite_neg_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_neg_energy**2))
print(average_excite_pos_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_pos_energy**2))

average_excite_energy = np.average(np.array([average_excite_pos_energy, average_excite_neg_energy]))
std_excite_energy = np.sqrt(std_excite_pos_energy**2 + std_excite_neg_energy**2)
print(average_excite_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_energy**2))

excite_neg_erefs = np.zeros((5, 20000))
sp = np.zeros((27, 5, 10000))
roo = np.zeros((27, 5, 10000))
weights = np.zeros((27, 5, 10000))
des = np.zeros((27, 5, 10000))
for i in range(5):
    blah = np.load(f'XH_excite_state_full_h3o2_left_{i+1}.npz')
    eref = blah['Eref']
    d = np.zeros((27, 6, 5000))
    for j in range(27):
        d[j] = all_dists(blah['coords'][j]).T
        sp[j, i, :5000] = d[j, -1]
        roo[j, i, :5000] = d[j, -2]
        weights[j, i, :5000] = blah['weights'][j]
        des[j, i, :5000] = blah['d'][j]
    excite_neg_erefs[i] = eref


gtg = [0, 1, 2, 3, 4]
print(np.mean(np.mean(excite_neg_erefs[gtg, 5000:], axis=1), axis=0)*har2wave)
average_excite_neg_energy = np.mean(np.mean(excite_neg_erefs[gtg, 5000:], axis=1), axis=0)*har2wave
std_excite_neg_energy = np.std(np.mean(excite_neg_erefs[gtg, 5000:]*har2wave, axis=1))

excite_pos_erefs = np.zeros((5, 20000))
for i in range(5):
    blah = np.load(f'XH_excite_state_full_h3o2_right_{i+1}.npz')
    eref = blah['Eref']
    d = np.zeros((27, 6, 5000))
    for j in range(27):
        d[j] = all_dists(blah['coords'][j]).T
        sp[j, i, 5000:] = d[j, -1]
        roo[j, i, 5000:] = d[j, -2]
        weights[j, i, 5000:] = blah['weights'][j]
        des[j, i, 5000:] = blah['d'][j]
    excite_pos_erefs[i] = eref

sp_left = sp[..., :5000].flatten()
roo_left = roo[..., :5000].flatten()
weights_left = weights[..., :5000].flatten()
des_left = des[..., :5000].flatten()

sp_right = sp[..., 5000:].flatten()
roo_right = roo[..., 5000:].flatten()
weights_right = weights[..., 5000:].flatten()
des_right = des[..., 5000:].flatten()

sp = sp.flatten()
roo = roo.flatten()
weights = weights.flatten()
des = des.flatten()

amp, xx, yy = np.histogram2d(sp, roo, weights=des, bins=75, density=True)
binx = (xx[1:] + xx[:-1]) / 2.
biny = (yy[1:] + yy[:-1]) / 2.
amp[amp>2] = 2

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
tcc = ax.contourf(binx, biny, amp.T)
fig.colorbar(tcc)
plt.show()

amp, xx, yy = np.histogram2d(sp_left, roo_left, weights=des_left, bins=75, density=True)
binx = (xx[1:] + xx[:-1]) / 2.
biny = (yy[1:] + yy[:-1]) / 2.

fig, ax = plt.subplots()
tcc = ax.contourf(binx, biny, amp)
fig.colorbar(tcc)
plt.show()

amp, xx, yy = np.histogram2d(sp_right, roo_right, weights=des_right, bins=75, density=True)
binx = (xx[1:] + xx[:-1]) / 2.
biny = (yy[1:] + yy[:-1]) / 2.
amp[amp>12] = 12

fig, ax = plt.subplots()
tcc = ax.contourf(binx, biny, amp)
fig.colorbar(tcc)
plt.show()

print(np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_excite_pos_energy = np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_excite_pos_energy = np.std(np.mean(excite_pos_erefs[:, 5000:]*har2wave, axis=1))

print(average_excite_neg_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_neg_energy**2))
print(average_excite_pos_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_pos_energy**2))

average_excite_energy = np.average(np.array([average_excite_pos_energy, average_excite_neg_energy]))
std_excite_energy = np.sqrt(std_excite_pos_energy**2 + std_excite_neg_energy**2)
print(average_excite_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_energy**2))

