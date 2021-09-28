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
        psi[:, 0] = angle_function(coords, excite, shift, atoms)
        term1 = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2)) * \
                (2 * mw1) ** (1 / 2) * dists[:, 0]
        term1_2 = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
        term2 = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2))
        term2_2 = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2)) * \
                 (2 * mw2) ** (1 / 2) * dists[:, 1]
        if excite == 'asym':
            psi[:, 1] = 1/np.sqrt(2)*(term1*term1_2 - term2*term2_2)
        else:
            psi[:, 1] = 1/np.sqrt(2)*(term1*term1_2 + term2*term2_2)
    else:
        psi = np.zeros((len(coords), 3))
        psi[:, 0] = angle_function(coords, excite, shift, atoms)
        psi[:, 1] = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2))
        psi[:, 2] = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
    return np.prod(psi, axis=-1)


def dpsidx(coords, excite, shift, atoms):
    dists = oh_dists(coords)
    drx = drdx(coords, dists, shift)
    dthet = dthetadx(coords, shift)
    dr = np.concatenate((dthet[..., None], drx), axis=-1)
    collect = dpsidrtheta(coords, excite, dists, shift, atoms)

    if excite == 'asym' or excite == 'sym':
        psi = psi_t(coords, excite, shift, atoms)
        psip = psi_parts(coords, excite, dists, shift, atoms)
        term1 = np.matmul(dr, collect[:, None, [0, 1, 2], None]).squeeze() * \
                np.prod(psip[:, [0, 1, 2]], axis=-1)[:, None, None]
        term2 = np.matmul(dr, collect[:, None, [0, 3, 4], None]).squeeze() * \
                np.prod(psip[:, [0, 3, 4]], axis=-1)[:, None, None]
        if excite == 'asym':
            dpsi = 1/np.sqrt(2)*(term1 - term2)
            dpsi = dpsi / np.broadcast_to(psi[:, None, None], (dpsi.shape))
        else:
            dpsi = 1/np.sqrt(2)*(term1 + term2)
            dpsi = dpsi / np.broadcast_to(psi[:, None, None], (dpsi.shape))
    else:
        dpsi = np.matmul(dr, collect[:, None, :, None]).squeeze()

    return dpsi


def d2psidx2(coords, excite, shift, atoms):
    # import pyvibdmc as pv
    # check = pv.ChainRuleHelper.dth_dx(coords, [[1, 0, 2]])
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
        psi = psi_t(coords, excite, shift, atoms)
        psip = psi_parts(coords, excite, dists, shift, atoms)
        part1 = np.matmul(dr2, first_dir[:, None, [0, 1, 2], None]).squeeze() * \
                np.prod(psip[:, [0, 1, 2]], axis=-1)[:, None, None]
        part2 = np.matmul(dr1**2, second_dir[:, None, [0, 1, 2], None]).squeeze() * \
                np.prod(psip[:, [0, 1, 2]], axis=-1)[:, None, None]
        part3 = np.matmul(dr1*dr1[..., [1, 2, 0]], first_dir[:, None, [0, 1, 2], None]
                          *first_dir[:, None, [1, 2, 0], None]).squeeze() * \
                np.prod(psip[:, [0, 1, 2]], axis=-1)[:, None, None]
        term1 = part1 + part2 + 2*part3
        part1_2 = np.matmul(dr2, first_dir[:, None, [0, 3, 4], None]).squeeze() * \
                np.prod(psip[:, [0, 3, 4]], axis=-1)[:, None, None]
        part2_2 = np.matmul(dr1 ** 2, second_dir[:, None, [0, 3, 4], None]).squeeze() * \
                np.prod(psip[:, [0, 3, 4]], axis=-1)[:, None, None]
        part3_2 = np.matmul(dr1 * dr1[..., [1, 2, 0]], first_dir[:, None, [0, 3, 4], None]
                          * first_dir[:, None, [3, 4, 0], None]).squeeze() * \
                np.prod(psip[:, [0, 3, 4]], axis=-1)[:, None, None]
        term2 = part1_2 + part2_2 + 2*part3_2
        if excite == 'asym':
            dpsi = 1 / np.sqrt(2) * (term1 - term2)
            dpsi = dpsi / np.broadcast_to(psi[:, None, None], (dpsi.shape))
        else:
            dpsi = 1 / np.sqrt(2) * (term1 + term2)
            dpsi = dpsi / np.broadcast_to(psi[:, None, None], (dpsi.shape))
    else:
        part1 = np.matmul(dr2, first_dir[:, None, [0, 1, 2], None]).squeeze()
        part2 = np.matmul(dr1 ** 2, second_dir[:, None, [0, 1, 2], None]).squeeze()
        part3 = np.matmul(dr1 * dr1[..., [1, 2, 0]], first_dir[:, None, [0, 1, 2], None]
                          * first_dir[:, None, [1, 2, 0], None]).squeeze()
        dpsi = part1 + part2 + 2*part3

    return dpsi


def psi_parts(coords, excite, dists, shift, atoms):
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
    psi = np.zeros((len(coords), 5))
    psi[:, 0] = angle_function(coords, excite, shift, atoms)
    psi[:, 1] = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2)) * \
                (2 * mw1) ** (1 / 2) * dists[:, 0]
    psi[:, 2] = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
    psi[:, 3] = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2))
    psi[:, 4] = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2)) * \
                (2 * mw2) ** (1 / 2) * dists[:, 1]
    return psi


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
        psi[:, 0] = dangle(coords, excite, shift, atoms)
    else:
        psi = np.zeros((len(coords), 3))
        psi[:, 0] = dangle(coords, excite, shift, atoms)
        psi[:, 1] = -mw1*dists[:, 0]
        psi[:, 2] = -mw2*dists[:, 1]
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
        psi[:, 0] = d2angle(coords, excite, shift, atoms)
    else:
        psi = np.zeros((len(coords), 3))
        psi[:, 0] = d2angle(coords, excite, shift, atoms)
        psi[:, 1] = mw1**2*dists[:, 0]**2 - mw1
        psi[:, 2] = mw2**2*dists[:, 1]**2 - mw2
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


def drift(coords, excite, shift, atoms):
    coordz = np.array_split(coords, mp.cpu_count()-1)
    psi = pool.starmap(psi_t, zip(coordz, repeat(excite), repeat(shift), repeat(atoms)))
    psi = np.concatenate(psi)
    der = 2*np.concatenate(pool.starmap(dpsidx, zip(coordz, repeat(excite), repeat(shift),
                                                    repeat(atoms))))
    return der, psi


def local_kinetic(Psi, sigma):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    d2psi = pool.starmap(d2psidx2, zip(coords, repeat(Psi.excite), repeat(Psi.shift),
                                       repeat(Psi.atoms)))
    d2psi = np.concatenate(d2psi)
    kin = -1/2 * np.sum(np.sum(sigma**2/dtau*d2psi, axis=1), axis=1)
    return kin


def get_pot(coords):
    V = PatrickShinglePotential(coords)
    return V


def pot(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


def E_loc(Psi, sigma):
    kin = local_kinetic(Psi, sigma)
    Psi.El = kin + Psi.V
    return Psi


def metropolis(Fqx, Fqy, x, y, psi_1, psi_2, sigma):
    psi_ratio = (psi_2/psi_1)**2
    a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
    a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
    remove = np.argwhere(psi_2 * psi_1 < 0)
    a[remove] = 0.
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, sigma):
    Drift = sigma**2/2.*Fqx
    randomwalk = np.random.normal(0.0, sigma, size=(len(Psi.coords), sigma.shape[0], sigma.shape[1]))
    y = randomwalk + Drift + np.array(Psi.coords)
    Fqy, psi = drift(y, Psi.excite, Psi.shift, Psi.atoms)
    a = metropolis(Fqx, Fqy, Psi.coords, y, Psi.psit, psi, sigma)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqx, acceptance


def E_ref_calc(Psi):
    alpha = 1. / (2. * dtau)
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/P0)
    return E_ref


def Weighting(Eref, Psi, DW, Fqx):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    threshold = 0.01
    max_thresh = 20
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        if DW is True:
            Biggo_num = int(Psi.walkers[ind])
            Psi.walkers[i[0]] = Biggo_num
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_el = float(Psi.El[ind])
        Biggo_force = np.array(Fqx[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Fqx[i[0]] = Biggo_force

    death = np.argwhere(Psi.weights > max_thresh)
    for i in death:
        ind = np.argmin(Psi.weights)
        if DW is True:
            Biggo_num = float(Psi.walkers[i[0]])
            Psi.walkers[ind] = Biggo_num
        Biggo_weight = float(Psi.weights[i[0]])
        Biggo_pos = np.array(Psi.coords[i[0]])
        Biggo_pot = float(Psi.V[i[0]])
        Biggo_el = float(Psi.El[i[0]])
        Biggo_force = np.array(Fqx[i[0]])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[ind] = Biggo_pos
        Psi.V[ind] = Biggo_pot
        Psi.El[ind] = Biggo_el
        Fqx[ind] = Biggo_force
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


def run(N_0, time_steps, propagation, equilibration, wait_time, excite, initial_struct,
        initial_shifts, shift_rate, atoms):
    DW = False
    psi = Walkers(N_0, initial_struct, excite, initial_shifts, atoms)
    sigma = np.zeros((3, 3))
    sigma[0] = np.array([[np.sqrt(dtau/m_O)] * 3])
    if atoms[1].upper() == 'H':
        sigma[1] = np.array([[np.sqrt(dtau/m_H)]*3])
    else:
        sigma[1] = np.array([[np.sqrt(dtau/m_D)]*3])
    if atoms[2].upper() == 'H':
        sigma[2] = np.array([[np.sqrt(dtau/m_H)]*3])
    else:
        sigma[2] = np.array([[np.sqrt(dtau/m_D)]*3])

    Fqx, psi.psit = drift(psi.coords, psi.excite, psi.shift, psi.atoms)
    num_o_collections = int((time_steps - equilibration) / (propagation + wait_time)) + 1
    timez = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    accept = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = np.zeros(np.append(num_o_collections, psi.weights.shape))

    num = 0
    prop = float(propagation)
    wait = float(wait_time)
    Eref_array = np.zeros(time_steps)

    shift = np.zeros((time_steps + 1, len(psi.shift)))
    shift[0] = psi.shift
    shift_rate = np.array(shift_rate)
    psi.shift = np.array(psi.shift)
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i, flush=True)

        if DW is False:
            prop = float(propagation)
            wait -= 1.
        else:
            prop -= 1.

        if i == 0:
            psi = pot(psi)
            psi = E_loc(psi, sigma)
            Eref = E_ref_calc(psi)

        psi, Fqx, acceptance = Kinetic(psi, Fqx, sigma)
        shift[i + 1] = psi.shift
        psi = pot(psi)
        psi = E_loc(psi, sigma)

        psi = Weighting(Eref, psi, DW, Fqx)
        Eref = E_ref_calc(psi)

        Eref_array[i] = Eref
        timez[i] = i + 1
        sum_weights[i] = np.sum(psi.weights)
        accept[i] = acceptance

        if i >= 5000:
            psi.shift = psi.shift + shift_rate

        if i >= int(equilibration) - 1 and wait <= 0. < prop:
            DW = True
            wait = float(wait_time)
            Psi_tau = copy.deepcopy(psi)
            coords[num] = Psi_tau.coords
            weights[num] = Psi_tau.weights
        elif prop == 0:
            DW = False
            des[num] = descendants(psi)
            num += 1

    return coords, weights, timez, Eref_array, sum_weights, accept, des


pool = mp.Pool(mp.cpu_count()-1)
