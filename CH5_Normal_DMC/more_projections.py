import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from Lookin_at_projects import rotateBackToFrame

ang2bohr = 1.e-10/5.291772106712e-11


def calcD(xx, O, H1, H2, H3):
    O = xx[:, O]
    # Every walker's xyz coordinate for O
    H1 = xx[:, H1]
    H2 = xx[:, H2]
    H3 = xx[:, H3]
    # test = H1-O
    aOH1 = np.divide((H1 - O), la.norm(H1 - O, axis=1)[:, np.newaxis])  # broadcasting silliness
    aOH2 = np.divide((H2 - O), la.norm(H2 - O, axis=1)[:, np.newaxis])
    aOH3 = np.divide((H3 - O), la.norm(H3 - O, axis=1)[:, np.newaxis])
    # point in space along OH bonds that is 1 unit away from the O
    aH1 = O + aOH1
    aH2 = O + aOH2
    aH3 = O + aOH3
    # vectors between the points along the OH bonds that are 1 unit vector away from the O
    vaH1H2 = aH2 - aH1
    vaH2H3 = aH3 - aH2
    # calculate vector
    line = np.cross(vaH1H2, vaH2H3, axis=1)
    D = O + (line / la.norm(line, axis=1)[:, np.newaxis])
    return D


def dists(coords, ind1, ind2):
    return la.norm(coords[:, ind1]-coords[:, ind2], axis=1)


def angles(coords, ind1, ind2, ind3):
    v1 = (coords[:, ind2] - coords[:, ind1]) / np.broadcast_to(dists(coords, ind1, ind2)[:, None], (len(coords), 3))
    v2 = (coords[:, ind3] - coords[:, ind1]) / np.broadcast_to(dists(coords, ind1, ind3)[:, None], (len(coords), 3))

    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))

    ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())

    return ang1


def umbrellaDi(xx, O, H1, H2, H3):

    """O,H1,H2,H3 are indices. """
    # calculate d, the trisector point of the umbrella
    D = calcD(xx, O, H1, H2, H3)
    addedX = np.concatenate((xx, D[:, np.newaxis, :]), axis=1)  # Change D index
    getXYZ = False
    if getXYZ:
        wf = open('test_umb.xyz', 'w+')
        trim = ["H", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "F"]
        for wI, walker in enumerate(addedX):
            wf.write("14\n")
            wf.write("0.0 0.0 0.0 0.0 0.0\n")
            for aI, atm in enumerate(walker):
                wf.write("%s %5.12f %5.12f %5.12f\n" % (trim[aI], atm[0]/ang2bohr, atm[1]/ang2bohr, atm[2]/ang2bohr))
            wf.write("\n")
        wf.close()
    umbrell = np.rad2deg(angles(addedX, O, H2, -1) ) # 4 11 0 now O H1 0
    return umbrell

for j in range(2):
    Nw = 40000
    # j = 1

    blah = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
                             f'ptetramer_non_imp_samp_ts_10_{Nw}_' +
                             f'Walkers_Test_{j+1}.npz')

    coords = blah['coords'][:20]
    coords = np.reshape(coords, (coords.shape[0] * coords.shape[1], coords.shape[2], coords.shape[3]))
    des = blah['des'][:20]
    des = np.reshape(des, (des.shape[0] * des.shape[1]))


    coords = rotateBackToFrame(coords, 7, 10, 13)
    umb = umbrellaDi(coords, 3, 0, 1, 2)
    sym_coords = np.hstack((umb, 180-umb))
    sym_weights = np.hstack((des, des))

    amp, xx = np.histogram(sym_coords, weights=sym_weights, bins=40, range=(60, 120), density=True)
    bins = (xx[1:] + xx[:-1]) / 2.

    plt.plot(bins, amp)
    plt.xlabel(r'$\theta$ (deg)')
plt.show()