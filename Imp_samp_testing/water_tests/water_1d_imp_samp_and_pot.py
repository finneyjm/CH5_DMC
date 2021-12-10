from Potential.Water_monomer_pot_fns import PatrickShinglePotential
import numpy as np
from scipy import interpolate
import DMC_Tools as dt


def psi_t(coords, kwargs):
    excite = kwargs['excite']
    interp = kwargs['interp']
    shift = kwargs['shift']
    mass = kwargs['mass']
    water = np.load('paf_monomer_coords.npy')
    eck = dt.EckartsSpinz(water, coords, mass, planar=True)
    coords = eck.get_rotated_coords()
    try:
        timestep = kwargs['timestep']
        # print(timestep)
        shift_rate = kwargs['shift_rate']
        if timestep > 4999:
            shift = shift + (timestep-4999)*np.array(shift_rate)
    except (Exception,):
        pass
    psi = np.ones((len(coords), 3))
    dists = oh_dists(coords)
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    # print(shift[1])
    if shift is not None:
        anti = anti - shift[0]
        sym = sym - shift[1]
    psi[:, 0] = interpolate.splev(anti, interp[0], der=0)
    psi[:, 1] = interpolate.splev(sym, interp[1], der=0)
    psi[:, 2] = angle_function(coords, excite, shift)
    return np.prod(psi, axis=1)


def harm_psit(coords, kwargs):
    excite = kwargs['excite']
    omega_OH = 3890.7865072878913
    ang2bohr = 1.e-10 / 5.291772106712e-11
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    har2wave = 219474.6
    m_OH = (m_H * m_O) / (m_H + m_O)
    mw_h = m_OH * omega_OH / har2wave
    dists = harm_dist(coords)
    r1 = 0.9616036495623883 * ang2bohr
    # r2 = 0.9616119936423067 * ang2bohr
    # req = [r1, r2]
    dists = dists - r1
    if excite is True:
        psi = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists ** 2)) * (2 * mw_h) ** (
            1 / 2) * dists
    else:
        psi = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists ** 2))
    return psi


def harm_dpsi_dr(coords, excite):
    omega_OH = 3890.7865072878913
    ang2bohr = 1.e-10 / 5.291772106712e-11
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    har2wave = 219474.6
    m_OH = (m_H * m_O) / (m_H + m_O)
    mw_h = m_OH * omega_OH / har2wave
    dists = harm_dist(coords)
    r1 = 0.9616036495623883 * ang2bohr
    # r2 = 0.9616119936423067 * ang2bohr
    # req = [r1, r2]
    dists = dists - r1
    if excite is True:
        psi = (1 - mw_h*dists**2)/dists
    else:
        psi = -mw_h*dists
    return psi


def harm_d2psi_dr(coords, excite):
    omega_OH = 3890.7865072878913
    ang2bohr = 1.e-10 / 5.291772106712e-11
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    har2wave = 219474.6
    m_OH = (m_H * m_O) / (m_H + m_O)
    mw_h = m_OH * omega_OH / har2wave
    dists = harm_dist(coords)
    r1 = 0.9616036495623883 * ang2bohr
    # r2 = 0.9616119936423067 * ang2bohr
    # req = [r1, r2]
    dists = dists - r1
    if excite is True:
        psi = mw_h*(mw_h*dists**2 - 3)
    else:
        psi = mw_h**2*dists**2 - mw_h
    return psi


def harm_pot(coords):
    omega_OH = 3890.7865072878913
    ang2bohr = 1.e-10 / 5.291772106712e-11
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    har2wave = 219474.6
    m_OH = (m_H * m_O) / (m_H + m_O)
    k = m_OH*(omega_OH/har2wave)**2
    d = harm_dist(coords)
    r1 = 0.9616036495623883 * ang2bohr
    x = d - r1
    return 1/2*k*x**2


def harm_dpsi_dx(coords, excite):
    dists = harm_dist(coords)
    drx = harm_drdx(coords, dists)
    dr = drx[..., None]
    collect = harm_dpsi_dr(coords, excite)
    return np.matmul(dr, collect[:, None, None, None]).squeeze()


def harm_d2psi_dx2(coords, excite):
    dists = harm_dist(coords)
    drx = harm_drdx(coords, dists)
    dr = drx[..., None]
    collect = harm_d2psi_dr(coords, excite)
    dr2 = harm_drdx2(coords, dists)[..., None]
    collect2 = harm_dpsi_dr(coords, excite)
    return np.matmul(dr**2, collect[:, None, None, None]).squeeze() \
           + np.matmul(dr2, collect2[:, None, None, None]).squeeze()


def harm_derivatives(coords, kwargs):
    excite = kwargs['excite']
    deriv1 = harm_dpsi_dx(coords, excite)
    deriv2 = harm_d2psi_dx2(coords, excite)
    return deriv1, deriv2


def dpsidx(coords, excite, interp, shift):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)
    dthet = dthetadx(coords, shift)
    dr = np.concatenate((drx, dthet[..., None]), axis=-1)
    collect = dpsidrtheta(coords, dists, excite, interp, shift)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def duvdx(drx):
    return 1/np.sqrt(2)*np.concatenate(((drx[..., 1, None] - drx[..., 0, None]),
                                        (drx[..., 0, None] + drx[..., 1, None])), axis=-1)


def d2psidx2(coords, excite, interp, shift):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)
    dthet = dthetadx(coords, shift)
    dr1 = np.concatenate((drx, dthet[..., None]), axis=-1)
    drx2 = drdx2(coords, dists)
    drx2 = duvdx(drx2)
    dthet2 = dthetadx2(coords, angle(coords), shift)
    dr2 = np.concatenate((drx2, dthet2[..., None]), axis=-1)
    first_dir = dpsidrtheta(coords, dists, excite, interp, shift)
    second_dir = d2psidrtheta(coords, dists, excite, interp, shift)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1**2, second_dir[:, None, :, None]).squeeze()
    part3 = np.matmul(dr1*dr1[..., [1, 2, 0]], first_dir[:, None, :, None]*first_dir[:, None, [1, 2, 0], None]).squeeze()
    return part1+part2+2*part3
    # return part2


def cartesian_check_sym(coords, excite, interp, shift=None):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)[..., 1]
    dr1 = drx[..., None]
    drx2 = drdx2(coords, dists)
    drx2 = duvdx(drx2)[..., 1]
    dr2 = drx2[..., None]
    first_dir = dpsidrtheta(coords, dists, excite, interp, shift)[:, 1]
    second_dir = d2psidrtheta(coords, dists, excite, interp, shift)[:, 1]
    part1 = np.matmul(dr2, first_dir[:, None, None, None]).squeeze()
    part2 = np.matmul(dr1**2, second_dir[:, None, None, None]).squeeze()
    return part1 + part2, part1, part2


def angle_function(coords, excite, shift):
    angs = angle(coords)
    if shift is not None:
        angs = angs - shift[2]
    ang2bohr = 1.e-10 / 5.291772106712e-11
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    theta = np.deg2rad(104.50800290215986)
    har2wave = 219474.6
    muH = 1 / m_H
    muO = 1 / m_O
    G = gmat(muH, muH, muO, r1, r2, theta)
    freq = 1668.4590610594878
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2) * (2*alpha) ** (1/2) * (angs-theta)
    else:
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2)


def gmat(mu1, mu2, mu3, r1, r2, ang):
    return mu1/r1**2 + mu2/r2**2 + mu3*(1/r1**2 + 1/r2**2 - 2*np.cos(ang)/(r1*r2))


def dangle(coords, excite, shift):
    angs = angle(coords)
    if shift is not None:
        angs = angs - shift[2]
    ang2bohr = 1.e-10 / 5.291772106712e-11
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    har2wave = 219474.6
    theta = np.deg2rad(104.50800290215986)
    muH = 1 / m_H
    muO = 1 / m_O
    G = gmat(muH, muH, muO, r1, r2, theta)
    freq = 1668.4590610594878
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        return (1 - alpha * (angs-theta) ** 2) / (angs-theta)
    else:
        return -alpha*(angs-theta)


def d2angle(coords, excite, shift):
    angs = angle(coords)
    if shift is not None:
        angs = angs - shift[2]
    ang2bohr = 1.e-10 / 5.291772106712e-11
    har2wave = 219474.6
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num * me * 1000)
    m_H = 1.007825 / (Avo_num * me * 1000)
    theta = np.deg2rad(104.50800290215986)
    muH = 1 / m_H
    muO = 1 / m_O
    G = gmat(muH, muH, muO, r1, r2, theta)
    freq = 1668.4590610594878
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        return alpha * (alpha * (angs-theta) ** 2 - 3)
    else:
        return alpha**2*(angs-theta)**2 - alpha


def dthetadx(coords, shift):
    chain = np.zeros((len(coords), 3, 3, 4))
    dx = 1e-3  #Bohr
    coeffs = np.array([1/12, -2/3, 2/3, -1/12])/dx
    if shift is None:
        shift = np.zeros(3)
    for atom in range(3):
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = angle(coords) - shift[2]
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = angle(coords) - shift[2]
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 2] = angle(coords) - shift[2]
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 3] = angle(coords) - shift[2]
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)


def dthetadx2(coords, angs, shift):
    chain = np.zeros((len(coords), 3, 3, 5))
    chain[:, :, :, 2] = np.broadcast_to(angs[..., None, None], (len(coords), 3, 3))
    dx = 1e-3
    coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])/(dx**2)
    if shift is None:
        shift = np.zeros(3)
    for atom in range(3):
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = angle(coords) - shift[2]
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = angle(coords) - shift[2]
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 3] = angle(coords) - shift[2]
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 4] = angle(coords) - shift[2]
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)


def angle(coords):
    dists = oh_dists(coords)
    v1 = (coords[:, 1] - coords[:, 0]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
    v2 = (coords[:, 2] - coords[:, 0]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))

    ang1 = np.arccos(np.matmul(v1[:, None, :], v2[..., None]).squeeze())

    return ang1.T


def dpsidrtheta(coords, dists, excite, interp, shift=None):
    collect = np.zeros((len(coords), 3))
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    if shift is not None:
        anti = anti - shift[0]
        sym = sym - shift[1]
    collect[:, 0] = interpolate.splev(anti, interp[0], der=1)/interpolate.splev(anti, interp[0], der=0)
    collect[:, 1] = interpolate.splev(sym, interp[1], der=1)/interpolate.splev(sym, interp[1], der=0)
    collect[:, 2] = dangle(coords, excite, shift)
    return collect


def d2psidrtheta(coords, dists, excite, interp, shift=None):
    collect = np.zeros((len(coords), 3))
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    if shift is not None:
        anti = anti - shift[0]
        sym = sym - shift[1]
    collect[:, 0] = interpolate.splev(anti, interp[0], der=2)/interpolate.splev(anti, interp[0], der=0)
    collect[:, 1] = interpolate.splev(sym, interp[1], der=2)/interpolate.splev(sym, interp[1], der=0)
    collect[:, 2] = d2angle(coords, excite, shift)
    return collect


def oh_dists(coords):
    bonds = [[1, 2], [1, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def harm_dist(coords):
    bonds = [[1, 2]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis.squeeze()


def drdx(coords, dists):
    chain = np.zeros((len(coords), 3, 3, 2))
    for bond in range(2):
        chain[:, 0, :, bond] += ((coords[:, 0]-coords[:, bond+1])/dists[:, bond, None])
        chain[:, bond+1, :, bond] += ((coords[:, bond+1]-coords[:, 0])/dists[:, bond, None])
    return chain


def harm_drdx(coords, dists):
    chain = np.zeros((len(coords), 2, 3))
    chain[:, 0, :] += ((coords[:, 0] - coords[:, 1]) / dists[:, None])
    chain[:, 1, :] += ((coords[:, 1] - coords[:, 0]) / dists[:, None])
    return chain


def drdx2(coords, dists):
    chain = np.zeros((len(coords), 3, 3, 2))
    for bond in range(2):
        chain[:, 0, :, bond] = (1./dists[:, bond, None] - (coords[:, 0]-coords[:, bond+1])**2/dists[:, bond, None]**3)
        chain[:, bond + 1, :, bond] = (1./dists[:, bond, None] - (coords[:, bond + 1] - coords[:, 0])**2 / dists[:, bond, None]**3)
    return chain


def harm_drdx2(coords, dists):
    chain = np.zeros((len(coords), 2, 3))
    for bond in range(1):
        chain[:, 0, :] = (1./dists[:, None] - (coords[:, 0]-coords[:, bond+1])**2/dists[:, None]**3)
        chain[:, bond + 1, :] = (1./dists[:, None] - (coords[:, bond + 1] - coords[:, 0])**2 / dists[:, None]**3)
    return chain


def full_dpsi_dx(coords, kwargs):
    excite = kwargs['excite']
    interp = kwargs['interp']
    shift = kwargs['shift']
    mass = kwargs['mass']
    water = np.load('paf_monomer_coords.npy')
    eck = dt.EckartsSpinz(water, coords, mass, planar=True)
    coords = eck.get_rotated_coords()
    try:
        timestep = kwargs['timestep']
        shift_rate = kwargs['shift_rate']
        if timestep > 4999:
            shift = shift + (timestep-4999)*np.array(shift_rate)
    except (Exception,):
        pass
    first_derivative = dpsidx(coords, excite, interp, shift)
    second_derivative = d2psidx2(coords, excite, interp, shift)
    return first_derivative, second_derivative


def get_pot(coords):
    V = PatrickShinglePotential(coords)
    return V


# from Coordinerds.CoordinateSystems import *
# def linear_combo_stretch_grid(r1, r2, coords):
#     coords = np.array([coords] * 1)
#     zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
#                                                                         ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
#                                                                                    [2, 0, 1, 0]])).coords
#     N = len(r1)
#     zmat = np.array([zmat]*N).squeeze()
#     zmat[:, 0, 1] = r1
#     zmat[:, 1, 1] = r2
#     new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     return new_coords
#
#
# def finite_difference_local_kinetic(coords, kwargs, sigma):
#     much_psi = np.zeros((len(coords), 3, 3, 3))
#     much_psi[:, 1] = np.broadcast_to(psi_t(coords, kwargs)[:, None, None], (len(coords), 3, 3))
#     dx = 1e-3
#     for atom in range(3):
#         for xyz in range(3):
#             coords[:, atom, xyz] -= dx
#             much_psi[:, 0, atom, xyz] = psi_t(coords, kwargs)
#             coords[:, atom, xyz] += 2*dx
#             much_psi[:, 2, atom, xyz] = psi_t(coords, kwargs)
#             coords[:, atom, xyz] -= dx
#     d2psidx2 = ((much_psi[:, 0] - 2. * much_psi[:, 1] + much_psi[:, 2]) / dx ** 2) / much_psi[:, 1]
#     kin = -1. / 2. * np.sum(np.sum(sigma ** 2 * d2psidx2, axis=1), axis=1)
#     return kin
#
#
#
# anti = np.load('antisymmetric_stretch_water_wvfns.npz')
# sym = np.load('symmetric_stretch_water_wvfns.npz')
#
# interp_anti = interpolate.splrep(anti['grid'], np.abs(anti['ground']), s=0)
# interp_sym = interpolate.splrep(sym['grid'], sym['excite'], s=0)
# interp = [interp_anti, interp_sym]
#
# kwargs = {
#     "excite": "sym",
#     "interp": interp,
#     "shift": [0, 0.00, 0],
#     # "shift_rate": [0, -0.00001, 0]
# }
# ang2bohr = 1.e-10 / 5.291772106712e-11
# re = 0.95784 * ang2bohr
# re2 = 0.95783997 * ang2bohr
# num_points = 400
# anti = np.zeros(num_points)
# sym = np.linspace(-0.55, 0.85, num_points) + (re + re2)/np.sqrt(2)
# A = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
# X, Y = np.meshgrid(anti, sym, indexing='ij')
# eh = np.matmul(np.linalg.inv(A), np.vstack((X.flatten(), Y.flatten())))
# r1 = eh[0]
# r2 = eh[1]
# water = np.load('monomer_coords.npy')
# grid = linear_combo_stretch_grid(r1, r2, water)
# me = 9.10938356e-31
# Avo_num = 6.0221367e23
# m_O = 15.994915 / (Avo_num*me*1000)
# m_H = 1.007825 / (Avo_num*me*1000)
# m_D = 2.01410177812 / (Avo_num*me*1000)
#
# first, second = full_dpsi_dx(grid, kwargs)
# sigma = np.zeros((3, 3))
# dtau = 1
# sigma[0] = np.array([[np.sqrt(dtau/m_O)] * 3])
# sigma[1] = np.array([[np.sqrt(dtau/m_H)]*3])
# sigma[2] = np.array([[np.sqrt(dtau/m_H)]*3])
# m_OH = m_H*m_O/(m_H + m_O)
# ang = np.deg2rad(104.1747712)
# sym_gmat_one_over = 1/(1/m_OH + np.cos(ang)/m_O)
# sigma_sym = np.sqrt(dtau/sym_gmat_one_over)
#
# kin = -1/2 * np.sum(np.sum(sigma**2*second, axis=1), axis=1)
# oned_kin = -1/2 * sigma_sym**2*d2psidrtheta(grid, oh_dists(grid), 'sym', interp)[:, 1]
# full_deriv, part1, part2 = cartesian_check_sym(grid, "sym", interp)
# cart_kin = -1/2 * np.sum(np.sum(sigma**2*full_deriv, axis=1), axis=1)
# part1_kin = -1/2 * np.sum(np.sum(sigma**2*part1, axis=1), axis=1)
# part2_kin = -1/2 * np.sum(np.sum(sigma**2*part2, axis=1), axis=1)
# fd_kin = finite_difference_local_kinetic(grid, kwargs, sigma)
# v = get_pot(grid)
# har2wave = 219474.6
#
# import matplotlib.pyplot as plt
# el = kin + v
# el = el.reshape((num_points, num_points))[0]
# fd_el = (fd_kin + v).reshape((num_points, num_points))[0]
# oned_el = (oned_kin + v).reshape((num_points, num_points))[0]
# cart_el = (cart_kin + v).reshape((num_points, num_points))[0]
# part1_el = (part1_kin + v).reshape((num_points, num_points))[0]
# part2_el = (part2_kin + v).reshape((num_points, num_points))[0]
# v = v.reshape((num_points, num_points))[0]
#
# plt.plot(sym, oned_el*har2wave, label='1d local energy')
# plt.plot(sym, cart_el*har2wave, label='cartesian local energy')
# plt.plot(sym, fd_el*har2wave, label='fd cartesian local energy')
# # plt.plot(sym, part1_el*har2wave, label='part1 cartesian local energy')
# # plt.plot(sym, part2_el*har2wave, label='part2 cartesian local energy')
# plt.plot(sym, v*har2wave, label='Potential')
# plt.ylabel(r'Energy cm$^-1$')
# plt.xlabel('s (Bohr)')
# plt.legend()
# plt.ylim(0, 20000)
# plt.show()
