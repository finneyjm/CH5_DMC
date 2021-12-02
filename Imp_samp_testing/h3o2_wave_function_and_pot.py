import numpy as np
from ProtWaterPES import *


def interp(x, y, poiuy):
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = poiuy(x[i], y[i])
    return out


def psi_t(coords, kwargs):
    excite = kwargs['excite']
    interp_func = kwargs['interp']
    psi = np.zeros((len(coords), 2))
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    m_OH = (m_H * m_O) / (m_H + m_O)
    har2wave = 219474.6

    omega_asym = 3070.648654929466 / har2wave

    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    if excite == 'sp & a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = interp(dists[:, -1], dists[:, -2], interp_func)
    elif excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = interp(dists[:, -1], dists[:, -2], interp_func)
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = interp(dists[:, -1], dists[:, -2], interp_func)
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = interp(dists[:, -1], dists[:, -2], interp_func)
    return psi


def dpsidasp(coords, excite, dists, dx1, dy1):
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    m_OH = (m_H * m_O) / (m_H + m_O)
    har2wave = 219474.6

    omega_asym = 3070.648654929466 / har2wave

    collect = np.zeros((len(coords), 3))
    mw_h = m_OH * omega_asym
    if excite == 'sp & a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], dx1)
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], dy1)
    elif excite == 'sp':
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], dx1)
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], dy1)
    elif excite == 'a':
        collect[:, 0] = (1 - mw_h*dists[:, 0]**2)/dists[:, 0]
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], dx1)
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], dy1)
    else:
        collect[:, 0] = -mw_h*dists[:, 0]
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], dx1)
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], dy1)
    return collect


def d2psidasp(coords, excite, dists, dx2, dy2, dx1_dy1):
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    m_OH = (m_H * m_O) / (m_H + m_O)
    har2wave = 219474.6

    omega_asym = 3070.648654929466 / har2wave

    collect = np.zeros((len(coords), 4))
    mw_h = m_OH * omega_asym
    if excite == 'sp & a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], dx2)
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], dy2)
        collect[:, 3] = interp(dists[:, -1], dists[:, -2], dx1_dy1)
    elif excite == 'sp':
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], dx2)
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], dy2)
        collect[:, 3] = interp(dists[:, -1], dists[:, -2], dx1_dy1)
    elif excite == 'a':
        collect[:, 0] = mw_h*(mw_h*dists[:, 0]**2 - 3)
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], dx2)
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], dy2)
        collect[:, 3] = interp(dists[:, -1], dists[:, -2], dx1_dy1)
    else:
        collect[:, 0] = mw_h**2*dists[:, 0]**2 - mw_h
        collect[:, 2] = interp(dists[:, -1], dists[:, -2], dx2)
        collect[:, 1] = interp(dists[:, -1], dists[:, -2], dy2)
        collect[:, 3] = interp(dists[:, -1], dists[:, -2], dx1_dy1)
    return collect


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


def dpsidx(coords, excite, dx1, dy1):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])
    dspx = dspdx(coords)
    dr = np.concatenate((droox, dspx[..., None]), axis=-1)
    collect = dpsidasp(coords, excite, dists, dx1, dy1)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def d2psidx2(coords, excite, dx1, dy1, dx2, dy2, dx1_dy1):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])
    dspx = dspdx(coords)
    dr1 = np.concatenate((droox, dspx[..., None]), axis=-1)
    droox2 = daroodx2(coords, dists[:, [1, 2, -2]])
    dspx2 = d2spdx2(coords, dists[:, -1])
    dr2 = np.concatenate((droox2, dspx2[..., None]), axis=-1)
    first_dir = dpsidasp(coords, excite, dists, dx1, dy1)
    second_dir = d2psidasp(coords, excite, dists, dx2, dy2, dx1_dy1)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1 ** 2, second_dir[:, None, 0:3, None]).squeeze()
    part3 = dr1[..., 0] * dr1[..., 1] * np.broadcast_to(second_dir[:, -1, None, None], (len(dr1), 5, 3)).squeeze()
    part4 = np.matmul(np.broadcast_to(dr1[..., 0, None], droox.shape)*dr1[..., [1, 2]],
                      (np.broadcast_to(first_dir[:, 0, None],
                       first_dir[:, [1, 2]].shape)*first_dir[:, [1, 2]])[:, None, :, None]).squeeze()
    return part1 + part2 + 2*part3 + 2*part4


def daroodx(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = (-1)**bond*1/np.sqrt(2)*(
                (coords[:, 2*bond + 1] - coords[:, 2*bond + 2]) / dists[:, bond, None])
        chain[:, 2*bond + 2, :, 0] = (-1)**bond*1/np.sqrt(2)*(
                (coords[:, 2*bond + 2] - coords[:, 2*bond + 1]) / dists[:, bond, None])
    chain[:, 1, :, 1] = ((coords[:, 1] - coords[:, 3]) / dists[:, -1, None])
    chain[:, 3, :, 1] = ((coords[:, 3] - coords[:, 1]) / dists[:, -1, None])
    return chain


def daroodx2(coords, dists):
    chain = np.zeros((len(coords), 5, 3, 2))
    for bond in range(2):
        chain[:, 2*bond + 1, :, 0] = (-1)**bond*1/np.sqrt(2)*(1./dists[:, bond, None] - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2/dists[:, bond, None]**3)
        chain[:, 2*bond + 2, :, 0] = (-1)**bond*1/np.sqrt(2)*(1./dists[:, bond, None] - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2/dists[:, bond, None]**3)
    chain[:, 1, :, 1] = (1./dists[:, -1, None] - (coords[:, 1]-coords[:, 3])**2/dists[:, -1, None]**3)
    chain[:, 3, :, 1] = (1./dists[:, -1, None] - (coords[:, 1]-coords[:, 3])**2/dists[:, -1, None]**3)
    return chain


def dspdx(coords):
    chain = np.zeros((len(coords), 5, 3, 4))
    dx = 1e-3  #Bohr
    coeffs = np.array([1/12, -2/3, 2/3, -1/12])/dx
    atoms = [0, 1, 3]  # the only atoms that affect the derivative of sp
    for atom in atoms:
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 2] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 3] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)


def d2spdx2(coords, sp):
    chain = np.zeros((len(coords), 5, 3, 5))
    chain[:, :, :, 2] = np.broadcast_to(sp[..., None, None], (len(coords), 5, 3))
    dx = 1e-3  #Bohr
    coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])/(dx**2)
    atoms = [0, 1, 3]  # the only atoms that affect the derivative of sp
    for atom in atoms:
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 3] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 4] = sp_calc_for_fd(coords)
            coords[:, atom, xyz] -= 2*dx
    chain[:, [2, 4]] = np.zeros((len(coords), 2, 3, 5))
    return np.dot(chain, coeffs)


def sp_calc_for_fd(coords):
    bonds = [[1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    mid = dis[:, 0] / 2
    sp = mid - dis[:, -1] * np.cos(roh_roo_angle(coords, dis[:, -2], dis[:, -1]))
    return sp


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot
