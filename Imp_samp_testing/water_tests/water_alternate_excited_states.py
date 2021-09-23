import numpy as np
from scipy import interpolate
from Potential.Water_monomer_pot_fns import PatrickShinglePotential
import multiprocessing as mp
from itertools import repeat
import copy
ang2bohr = 1.e-10/5.291772106712e-11
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_OD = (m_D*m_O)/(m_D+m_O)
m_OH = (m_H*m_O)/(m_H+m_O)
har2wave = 219474.6
omega_OD = 2832.531899782715
omega_OH = 3890.7865072878913
mw_d = m_OD * omega_OD/har2wave
mw_h = m_OH * omega_OH/har2wave
dtau = 1


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct, excite, initial_shifts, atoms, interp=None):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([initial_struct]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.excite = excite
        self.shift = initial_shifts
        self.atoms = atoms
        self.interp = interp


def psi_t(coords, excite, shift, atoms):
    dists = oh_dists(coords)
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            mw1 = mw_h
            mw2 = mw_h
        else:
            mw1 = mw_h
            mw2 = mw_d
    else:
        if atoms[2].upper() == 'H':
            mw1 = mw_d
            mw2 = mw_h
        else:
            mw1 = mw_d
            mw2 = mw_d
    dists = dists - shift[:2]
    if excite == 'asym' or excite == 'sym':
        psi = np.zeros((len(coords), 2))
        term1 = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2)) * \
                (2 * mw1) ** (1 / 2) * dists[:, 0]
        term1_2 = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
        term2 = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2))
        term2_2 = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2)) * \
                 (2 * mw2) ** (1 / 2) * dists[:, 1]
        if excite == 'asym':
            psi[:, 1] = 1/np.sqrt(2)*(term2*term2_2 - term1*term1_2)
        else:
            psi[:, 1] = 1/np.sqrt(2)*(term1*term1_2 + term2*term2_2)
    else:
        psi = np.zeros((len(coords), 3))
        psi[:, 1] = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2))
        psi[:, 2] = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
    psi[:, 0] = angle_function(coords, excite, shift, atoms)
    return np.prod(psi, axis=-1)


def dpsidx(coords, excite, shift, atoms):
    dists = oh_dists(coords)
    drx = drdx(coords, dists, shift)
    dthet = dthetadx(coords, shift)
    dr = np.concatenate((dthet[..., None], drx), axis=-1)
    collect = dpsidrtheta(coords, excite, dists, shift, atoms)

    if excite == 'asym' or excite == 'sym':
        term1 = np.matmul(dr, collect[:, None, [0, 1, 2], None]).squeeze()
        term2 = np.matmul(dr, collect[:, None, [0, 3, 4], None]).squeeze()
        if excite == 'asym':
            dpsi = 1/np.sqrt(2)*(term2 - term1)
        else:
            dpsi = 1/np.sqrt(2)*(term1 + term2)
    else:
        dpsi = np.matmul(dr, collect[:, None, :, None]).squeeze()

    return dpsi


def d2psidx2(coords, excite, shift, atoms):
    dists = oh_dists(coords)
    drx = drdx(coords, dists, shift)
    dthet = dthetadx(coords, shift)
    dr1 = np.concatenate((dthet[..., None], drx), axis=-1)
    drx2 = drdx2(coords, dists, shift)
    dthet2 = dthetadx2(coords, angle(coords), shift)
    dr2 = np.concatenate((dthet2[..., None], drx2), axis=-1)
    first_dir = dpsidrtheta(coords, excite, dists, shift, atoms)
    second_dir = d2psidrtheta(coords, excite, dists, shift, atoms)

    if excite == 'asym' or excite == 'sym':
        part1 = np.matmul(dr2, first_dir[:, None, [0, 1, 2], None]).squeeze()
        part2 = np.matmul(dr1**2, second_dir[:, None, [0, 1, 2], None]).squeeze()
        part3 = np.matmul(dr1*dr1[..., [1, 2, 0]], first_dir[:, None, [0, 1, 2], None]
                          *first_dir[:, None, [1, 2, 0], None]).squeeze()
        term1 = part1 + part2 + 2*part3
        part1 = np.matmul(dr2, first_dir[:, None, [0, 3, 4], None]).squeeze()
        part2 = np.matmul(dr1 ** 2, second_dir[:, None, [0, 3, 4], None]).squeeze()
        part3 = np.matmul(dr1 * dr1[..., [1, 2, 0]], first_dir[:, None, [0, 3, 4], None]
                          * first_dir[:, None, [3, 4, 0], None]).squeeze()
        term2 = part1 + part2 + 2*part3
        if excite == 'asym':
            dpsi = 1 / np.sqrt(2) * (term2 - term1)
        else:
            dpsi = 1 / np.sqrt(2) * (term1 + term2)
    else:
        part1 = np.matmul(dr2, first_dir[:, None, [0, 1, 2], None]).squeeze()
        part2 = np.matmul(dr1 ** 2, second_dir[:, None, [0, 1, 2], None]).squeeze()
        part3 = np.matmul(dr1 * dr1[..., [1, 2, 0]], first_dir[:, None, [0, 1, 2], None]
                          * first_dir[:, None, [1, 2, 0], None]).squeeze()
        dpsi = part1 + part2 + 2*part3

    return dpsi


def dpsidrtheta(coords, excite, dists, shift, atoms):
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            mw1 = mw_h
            mw2 = mw_h
        else:
            mw1 = mw_h
            mw2 = mw_d
    else:
        if atoms[2].upper() == 'H':
            mw1 = mw_d
            mw2 = mw_h
        else:
            mw1 = mw_d
            mw2 = mw_d
    dists = dists - shift[:2]
    if excite == 'asym' or excite == 'sym':
        psi = np.zeros((len(coords), 5))
        psi[:, 1] = (1 - mw1*dists[:, 0]**2)/dists[:, 0]
        psi[:, 2] = -mw2*dists[:, 1]
        psi[:, 3] = -mw1*dists[:, 0]
        psi[:, 4] = (1 - mw2*dists[:, 1]**2)/dists[:, 1]
    else:
        psi = np.zeros((len(coords), 3))
        psi[:, 1] = -mw1*dists[:, 0]
        psi[:, 2] = -mw2*dists[:, 1]
    psi[:, 0] = dangle(coords, excite, shift, atoms)
    return psi


def d2psidrtheta(coords, excite, dists, shift, atoms):
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            mw1 = mw_h
            mw2 = mw_h
        else:
            mw1 = mw_h
            mw2 = mw_d
    else:
        if atoms[2].upper() == 'H':
            mw1 = mw_d
            mw2 = mw_h
        else:
            mw1 = mw_d
            mw2 = mw_d
    dists = dists - shift[:2]
    if excite == 'asym' or excite == 'sym':
        psi = np.zeros((len(coords), 5))
        psi[:, 1] = mw1*(mw1*dists[:, 0]**2 - 3)
        psi[:, 2] = mw2**2*dists[:, 1]**2 - mw2
        psi[:, 3] = mw1**2*dists[:, 0]**2 - mw1
        psi[:, 4] = mw2*(mw2*dists[:, 1]**2 - 3)
    else:
        psi = np.zeros((len(coords), 3))
        psi[:, 1] = mw1**2*dists[:, 0]**2 - mw1
        psi[:, 2] = mw2**2*dists[:, 1]**2 - mw2
    psi[:, 0] = d2angle(coords, excite, shift, atoms)
    return psi


def angle_function(coords, excite, shift, atoms):
    angs = angle(coords)
    angs = angs - shift[2]
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    theta = np.deg2rad(104.1747712)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            G = gmat(muH, muH, muO, r1, r2, theta)
            freq = 1668.4590610594878
        else:
            G = gmat(muH, muD, muO, r1, r2, theta)
            freq = 1462.5810039828614
    else:
        if atoms[2].upper() == 'H':
            G = gmat(muD, muH, muO, r1, r2, theta)
            freq = 1462.5810039828614
        else:
            G = gmat(muD, muD, muO, r1, r2, theta)
            freq = 1222.5100195873742
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang':
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2) * (2*alpha) ** (1/2) * (angs-theta)
    else:
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2)


def dangle(coords, excite, shift, atoms):
    angs = angle(coords)
    angs = angs - shift[-1]
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    theta = np.deg2rad(104.1747712)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            G = gmat(muH, muH, muO, r1, r2, theta)
            freq = 1668.4590610594878
        else:
            G = gmat(muH, muD, muO, r1, r2, theta)
            freq = 1462.5810039828614
    else:
        if atoms[2].upper() == 'H':
            G = gmat(muD, muH, muO, r1, r2, theta)
            freq = 1462.5810039828614
        else:
            G = gmat(muD, muD, muO, r1, r2, theta)
            freq = 1222.5100195873742
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang':
        return (1 - alpha * (angs-theta) ** 2) / (angs-theta)
    else:
        return -alpha*(angs-theta)


def d2angle(coords, excite, shift, atoms):
    angs = angle(coords)
    angs = angs - shift[-1]
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    theta = np.deg2rad(104.1747712)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    if atoms[1].upper() == 'H':
        if atoms[2].upper() == 'H':
            G = gmat(muH, muH, muO, r1, r2, theta)
            freq = 1668.4590610594878
        else:
            G = gmat(muH, muD, muO, r1, r2, theta)
            freq = 1462.5810039828614
    else:
        if atoms[2].upper() == 'H':
            G = gmat(muD, muH, muO, r1, r2, theta)
            freq = 1462.5810039828614
        else:
            G = gmat(muD, muD, muO, r1, r2, theta)
            freq = 1222.5100195873742
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang':
        return alpha * (alpha * (angs-theta) ** 2 - 3)
    else:
        return alpha**2*(angs-theta)**2 - alpha


def oh_dists(coords):
    bonds = [[1, 2], [1, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def angle(coords):
    dists = oh_dists(coords)
    v1 = (coords[:, 1] - coords[:, 0]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
    v2 = (coords[:, 2] - coords[:, 0]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))

    ang1 = np.arccos(np.matmul(v1[:, None, :], v2[..., None]).squeeze())

    return ang1.T


def gmat(mu1, mu2, mu3, r1, r2, ang):
    return mu1/r1**2 + mu2/r2**2 + mu3*(1/r1**2 + 1/r2**2 - 2*np.cos(ang)/(r1*r2))


def drdx(coords, dists, shift):
    chain = np.zeros((len(coords), 3, 3, 2))
    dists = dists - shift[:2]
    for bond in range(2):
        chain[:, 0, :, bond] += ((coords[:, 0]-coords[:, bond+1])/dists[:, bond, None])
        chain[:, bond+1, :, bond] += ((coords[:, bond+1]-coords[:, 0])/dists[:, bond, None])
    return chain


def dthetadx(coords, shift):
    chain = np.zeros((len(coords), 3, 3, 4))
    dx = 1e-3  #Bohr
    coeffs = np.array([1/12, -2/3, 2/3, -1/12])/dx
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


def drdx2(coords, dists, shift):
    chain = np.zeros((len(coords), 3, 3, 2))
    dists = dists - shift[:2]
    for bond in range(2):
        chain[:, 0, :, bond] = (1./dists[:, bond, None] - (coords[:, 0]-coords[:, bond+1])**2/dists[:, bond, None]**3)
        chain[:, bond + 1, :, bond] = (1./dists[:, bond, None] - (coords[:, bond + 1] - coords[:, 0])**2 / dists[:, bond, None]**3)
    return chain


def dthetadx2(coords, angs, shift):
    chain = np.zeros((len(coords), 3, 3, 5))
    chain[:, :, :, 2] = np.broadcast_to(angs[..., None, None], (len(coords), 3, 3))
    dx = 1e-3
    coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])/(dx**2)
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


def local_kinetic(Psi, sigma):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    d2psi = pool.starmap(d2psidx2, zip(coords, repeat(Psi.excite), repeat(Psi.shift),
                                       repeat(Psi.atoms)))
    d2psi = np.concatenate(d2psi)
    kin = -1/2 * np.sum(np.sum(sigma**2/dtau*d2psi, axis=1), axis=1)
    return kin, d2psi


def get_pot(coords):
    V = PatrickShinglePotential(coords)
    return V


def pot(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


def E_loc(Psi, sigma):
    kin, d2psi = local_kinetic(Psi, sigma)
    Psi.El = kin + Psi.V
    return Psi, d2psi


pool = mp.Pool(mp.cpu_count()-1)


def all_da_psi(coords, excite, shift, atoms):
    dx = 1e-3
    psi = np.zeros((len(coords), 3, 3, 3))
    psi[:, 1] = np.broadcast_to(psi_t(coords, excite, shift, atoms)[:, None, None],
                                (len(coords), 3, 3))
    for atom in range(3):
        for xyz in range(3):
            coords[:, atom, xyz] -= dx
            psi[:, 0, atom, xyz] = psi_t(coords, excite, shift, atoms)
            coords[:, atom, xyz] += 2*dx
            psi[:, 2, atom, xyz] = psi_t(coords, excite, shift, atoms)
            coords[:, atom, xyz] -= dx
    return psi


def local_kinetic_finite(Psi):
    dx = 1e-3
    d2psidx2 = ((Psi.psit[:, 0] - 2. * Psi.psit[:, 1] + Psi.psit[:, 2]) / dx ** 2) / Psi.psit[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin, d2psidx2


from Coordinerds.CoordinateSystems import *


def linear_combo_stretch_grid(r1, r2, coords):
    re = np.linalg.norm(coords[0]-coords[1])
    re2 = np.linalg.norm(coords[0]-coords[2])
    re = 0.9616036495623883 * ang2bohr
    re2 = 0.9616119936423067 * ang2bohr

    coords = np.array([coords] * 1)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).squeeze()
    zmat[:, 0, 1] = re + r1
    zmat[:, 1, 1] = re2 + r2
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


def drift_fd(coords, excite, shift, atoms):
    dx = 1e-3
    psi = all_da_psi(coords, excite, shift, atoms)
    der = (psi[:, 2] - psi[:, 0])/dx/psi[:, 1]
    return der, psi


def grid_angle(a, b, num, coords):
    spacing = np.linspace(a, b, num)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    g = np.array([zmat]*num)
    g[:, 1, 3] = spacing
    new_coords = CoordinateSet(g, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


molecule = np.load('monomer_coords.npy')

anti = np.linspace(-0.75, 0.75, 200)
sym = np.zeros(200)
A = 1/np.sqrt(2)*np.array([[-1, 1], [1, 1]])
eh = np.matmul(np.linalg.inv(A), np.vstack((anti, sym)))
r1 = eh[0]
r2 = eh[1]

grid = linear_combo_stretch_grid(r1, r2, molecule)

psi = Walkers(50, molecule, 'asym', [0, 0, 0], ['O', 'D', 'H'])
psi.coords = grid

# d, _ = drift(psi.coords, 'sym', [0, 0, 0], ['O', 'H', 'H'], interp)
psi = pot(psi)
sigma = np.zeros((3, 3))
sigma[0] = np.array([[np.sqrt(dtau / m_O)] * 3])
sigma[1] = np.array([[np.sqrt(dtau / m_H)] * 3])
sigma[2] = np.array([[np.sqrt(dtau / m_H)] * 3])

psi, d2psi = E_loc(psi, sigma)

fd_d, psi.psit = drift_fd(psi.coords, psi.excite, psi.shift, psi.atoms)
# asdf, d2psi_fd = local_kinetic_finite(psi)
fd_eloc, d2psi_fd = local_kinetic_finite(psi)
fd_eloc = fd_eloc + psi.V
# theta = np.linspace(theta-1, theta+1, 50)
# anti = np.rad2deg(theta)
# anti = sym
import matplotlib.pyplot as plt
plt.plot(anti, psi.V*har2wave, label='potential')
plt.plot(anti, psi.El*har2wave, label='local energy')
plt.plot(anti, fd_eloc*har2wave, label='fd local energy')
plt.legend()
plt.show()
