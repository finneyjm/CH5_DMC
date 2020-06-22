import numpy as np
from scipy import interpolate

ang2bohr = 1.e-10/5.291772106712e-11
me = 9.10938356e-31
Avo_num = 6.0221367e23
har2wave = 219474.6
m_H = 1.00782503223 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
wvfn = np.load("free_oh_wvfn.npy")
free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)


def angles(coords, dists):
    v1 = (coords[:, 0] - coords[:, 2]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
    v2 = (coords[:, 1] - coords[:, 2]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))

    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))

    ang1 = np.arccos(np.matmul(v1_new, v2_new).squeeze())

    return ang1.T


def angle_function(coords, dists):
    angs = angles(coords, dists)
    r1 = 0.9616 * ang2bohr
    r2 = 0.961610 * ang2bohr
    theta = np.deg2rad(104.175)
    muH = 1 / m_H
    muO = 1 / m_O
    G = gmat(muH, muH, muO, r1, r2, theta)
    freq = 1668.4590610594878
    freq /= har2wave
    alpha = freq / G
    return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2)


def gmat(mu1, mu2, mu3, r1, r2, ang):
    return mu1/r1**2 + mu2/r2**2 + mu3*(1/r1**2 + 1/r2**2 - 2*np.cos(ang)/(r1*r2))


def dists(coords):
    bonds = [[3, 1], [3, 2]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def trial_wvfn(coords):
    reg_oh = dists(coords)
    psi = np.zeros((len(coords), int(3)))
    for i in range(2):
        psi[:, i] = interpolate.splev(reg_oh[:, i], free_oh_wvfn, der=0)
    psi[:, -1] = angle_function(coords, reg_oh)
    return np.prod(psi, axis=1)
