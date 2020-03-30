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

# thresh = ['half', 'one', 'five', 'ten', 'twenty', 'pone']
# i = -1
j = 2
#
#
blah = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
                         f'ptetramer_non_imp_samp_ts_10_{10000}_' +
                         f'Walkers_Test_{j + 1}.npz')

coords = blah['coords'][:20]
coords = np.reshape(coords, (coords.shape[0]*coords.shape[1], coords.shape[2], coords.shape[3]))
des = blah['des'][:20]
# print(des.shape)
des = np.reshape(des, (des.shape[0]*des.shape[1]))


coords = rotateBackToFrame(coords, 7, 10, 13)
amp, xx = np.histogram(coords[:, 4-1, 2], weights=des, range=(-2, 2), bins=40, density=True)
# amp, xx = np.histogram(coords[:, 4-1, 2], weights=des, bins=40, density=True)
bins = (xx[1:] + xx[:-1]) / 2.

plt.plot(bins, amp)
plt.xlabel('Z Displacement (Bohr)')
plt.show()
# ang2bohr = 1.e-10/5.291772106712e-11

# tetramer = [[-0.31106354,  -0.91215572,  -0.20184621],
#             [0.95094197,   0.18695800,  -0.20259538],
#             [-0.63272209,   0.72926470,  -0.20069859],
#             [0.00253217,   0.00164013,  -0.47227522],
#             [-1.96589559,   2.46292466,  -0.54312627],
#             [-2.13630186,   1.99106023,   0.90604777],
#             [-1.55099190,   1.94067311,   0.14704161],
#             [-1.18003749,  -2.92157144,  -0.54090532],
#             [-0.65410291,  -2.84939169,   0.89772271],
#             [-0.91203521,  -2.30896272,   0.14764850],
#             [2.79828182,   0.87002791,   0.89281564],
#             [3.12620054,   0.43432898,  -0.54032031],
#             [2.46079102,   0.36718848,   0.14815394]]

# coords = np.array([tetramer]*3)*ang2bohr

# from ProtWaterPES import Potential

# pot = Potential(13)
# v = pot.get_potential(coords[:-40000])
# a = 4
# pot = Potential(coords.shape[1])
# v1 = pot.get_potential(coords)
# coord_rot = rotateBackToFrame(coords, 7, 10, 13)
# v2 = pot.get_potential(coord_rot)
# print(coord_rot)
# coord = coord_rot
# coord[:, 0:4, 2] *= -1
# v3 = pot.get_potential(coord)
# print(v1)
# print(v2)
# print(v3)


