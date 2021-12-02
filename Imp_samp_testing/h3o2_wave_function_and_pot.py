import numpy as np
from scipy import interpolate
from ProtWaterPES import *


def psi_t(coords, excite, shift):
    psi = np.zeros((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    dists[:, 0] = dists[:, 0] - shift[0]
    if excite == 'all':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_both_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & roo':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_both_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'sp & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'roo & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = excite_roo_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'roo':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_roo_no_der(dists[:, -1], dists[:, -2])[np.argsort(np.argsort(dists[:, -2])), np.argsort(np.argsort(dists[:, -1]))]
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    return psi


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

