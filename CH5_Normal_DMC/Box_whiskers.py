import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])

def hh_dist(coords):
    N = len(coords[:, 0, 0])
    hh = np.zeros((N, 5, 5))
    a = np.full((5, 5), True)
    np.fill_diagonal(a, False)
    mask = np.broadcast_to(a, (N, 5, 5))
    for i in range(4):
        for j in np.arange(i, 5):
            hh[:, i, j] += np.linalg.norm(coords[:, j+1, :] - coords[:, i+1, :], axis=1)
    hh += np.transpose(hh, (0, 2, 1))
    blah = hh[mask].reshape(N, 5, 4)
    return blah

def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, 5))
    for i in range(5):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch


coords = np.reshape(coords_initial, (1, 6, 3))
ch_bonds = ch_dist(coords).reshape(5)
hh_dists = hh_dist(coords).reshape(5, 4)

ind = np.argsort(ch_bonds)
ch_bonds = ch_bonds[ind]
ang2bohr = 1.e-10/5.291772106712e-11
print(ch_bonds/ang2bohr)

ch_bonds = ['1.0881', '1.0881', '1.1076', '1.1966', '1.1971']
ordering = ['4', '5', '1', '2', '3']

bp = plt.boxplot(hh_dists[ind].T/ang2bohr, patch_artist=True, labels=ch_bonds)
# hh_dists = hh_dists[ind]
colors = np.array(['purple', 'orange', 'blue', 'green', 'red'])
colors = colors[ind]
hh_dists = hh_dists[ind]


for median in bp['medians']:
    median.set(color='black', linewidth=2)

for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=2)

for cap in bp['caps']:
    cap.set(color='black', linewidth=2)

for box in bp['boxes']:
    box.set(color='black', linewidth=2)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

for patch, color in zip(bp['fliers'], colors):
    patch.set(marker='o', color=color, alpha=0.8)

plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                     bottom=True, top=False, left=True, right=False, labelsize=14)
plt.ylabel(r'$\rmr_{HH}$ ($\rm\AA$)', fontsize=22)
plt.xlabel(r'r$_{\rmCH} (\rm\AA)$', fontsize=22)
plt.tight_layout()
plt.show()
plt.close()

for i in range(4):
    plt.scatter(ordering, hh_dists[:, i]/ang2bohr, marker='x', color='black')
    print(np.vstack((ordering, hh_dists[:, i])))

plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                     bottom=True, top=False, left=True, right=False, labelsize=14)
plt.ylabel(r'$\rmr_{HH}$ ($\rm\AA$)', fontsize=22)
plt.xlabel(r'r$_{\rmCH} (\rm\AA)$', fontsize=22)
plt.tight_layout()
plt.show()
plt.close()
