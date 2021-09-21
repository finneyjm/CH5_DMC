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
har2wave = 219474.6
dtau = 1


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct, excite, initial_shifts, atoms, interp):
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
        self.psit = np.zeros((walkers, 3))


def psi_t(coords, excite, shift, interp, atoms):
    psi = np.zeros((len(coords), 3))
    dists = oh_dists(coords)
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req

    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    anti = anti - shift[0]
    sym = sym - shift[0]
    psi[:, 0] = interpolate.splev(anti, interp[0], der=0)
    psi[:, 1] = interpolate.splev(sym, interp[1], der=0)
    psi[:, 2] = angle_function(coords, excite, shift, atoms)
    return psi


def dpsidx(coords, excite, shift, interp, atoms):
    dists = oh_dists(coords)
    drx = drdx(coords, dists, shift)
    drx = duvdx(drx)
    dthet = dthetadx(coords, shift)
    dr = np.concatenate((drx, dthet[..., None]), axis=-1)
    collect = dpsidrtheta(coords, excite, dists, shift, interp, atoms)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def duvdx(drx):
    return 1/np.sqrt(2)*np.concatenate(((drx[..., 1, None] - drx[..., 0, None]),
                                        (drx[..., 0, None] + drx[..., 1, None])), axis=-1)


def d2psidx2(coords, excite, shift, interp, atoms):
    dists = oh_dists(coords)
    drx = drdx(coords, dists, shift)
    drx = duvdx(drx)
    dthet = dthetadx(coords, shift)
    dr1 = np.concatenate((drx, dthet[..., None]), axis=-1)
    drx2 = drdx2(coords, dists, shift)
    drx2 = duvdx(drx2)
    dthet2 = dthetadx2(coords, angle(coords), shift)
    dr2 = np.concatenate((drx2, dthet2[..., None]), axis=-1)
    first_dir = dpsidrtheta(coords, excite, dists, shift, interp, atoms)
    second_dir = d2psidrtheta(coords, excite, dists, shift, interp, atoms)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1**2, second_dir[:, None, :, None]).squeeze()
    part3 = np.matmul(dr1*dr1[..., [1, 2, 0]], first_dir[:, None, :, None]*first_dir[:, None, [1, 2, 0], None]).squeeze()
    return part1+part2+2*part3


def dpsidrtheta(coords, excite, dists, shift, interp, atoms):
    collect = np.zeros((len(coords), 3))
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    anti = anti - shift[0]
    sym = sym - shift[0]
    collect[:, 0] = interpolate.splev(anti, interp[0], der=1)/interpolate.splev(anti, interp[0], der=0)
    collect[:, 1] = interpolate.splev(sym, interp[1], der=1)/interpolate.splev(sym, interp[1], der=0)
    collect[:, 2] = dangle(coords, excite, shift, atoms)
    return collect


def d2psidrtheta(coords, excite, dists, shift, interp, atoms):
    collect = np.zeros((len(coords), 3))
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    anti = anti - shift[0]
    sym = sym - shift[0]
    collect[:, 0] = interpolate.splev(anti, interp[0], der=2)/interpolate.splev(anti, interp[0], der=0)
    collect[:, 1] = interpolate.splev(sym, interp[1], der=2)/interpolate.splev(sym, interp[1], der=0)
    collect[:, 2] = d2angle(coords, excite, shift, atoms)
    return collect


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


def angle_function(coords, excite, shift, atoms):
    angs = angle(coords)
    angs = angs - shift[2]
    r1 = 0.9616036495623883*ang2bohr
    r2 = 0.9616119936423067*ang2bohr
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
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
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
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
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
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        return alpha * (alpha * (angs-theta) ** 2 - 3)
    else:
        return alpha**2*(angs-theta)**2 - alpha


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


def drift(coords, excite, shift, atoms, interp):
    coordz = np.array_split(coords, mp.cpu_count()-1)
    psi = pool.starmap(psi_t, zip(coordz, repeat(excite), repeat(shift), repeat(interp), repeat(atoms)))
    psi = np.concatenate(psi)
    der = 2*np.concatenate(pool.starmap(dpsidx, zip(coordz, repeat(excite), repeat(shift), repeat(interp),
                                                    repeat(atoms))))
    return der, psi


def local_kinetic(Psi, sigma):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    d2psi = pool.starmap(d2psidx2, zip(coords, repeat(Psi.excite), repeat(Psi.shift), repeat(Psi.interp),
                                       repeat(Psi.atoms)))
    d2psi = np.concatenate(d2psi)
    kin = -1/2 * np.sum(np.sum(sigma**2/dtau*d2psi, axis=1), axis=1)
    return kin


def metropolis(Fqx, Fqy, x, y, psi_1, psi_2, sigma):
    psi_ratio = np.prod((psi_2/psi_1)**2, axis=1)
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
    Fqy, psi = drift(y, Psi.excite, Psi.shift, Psi.atoms, Psi.interp)
    a = metropolis(Fqx, Fqy, Psi.coords, y, Psi.psit, psi, sigma)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqx, acceptance


def get_pot(coords):
    V = PatrickShinglePotential(coords)
    return V


def pot(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


def E_loc(Psi, sigma):
    Psi.El = local_kinetic(Psi, sigma) + Psi.V
    return Psi


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
        initial_shifts, shift_rate, atoms, interp):
    DW = False
    psi = Walkers(N_0, initial_struct, excite, initial_shifts, atoms, interp)
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

    Fqx, psi.psit = drift(psi.coords, psi.excite, psi.shift, psi.atoms, psi.interp)
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


def all_da_psi(coords, excite, shift, interp, atoms):
    dx = 1e-3
    psi = np.zeros((len(coords), 3, 3, 3))
    psi[:, 1] = np.broadcast_to(np.prod(psi_t(coords, excite, shift, interp, atoms), axis=1)[:, None, None],
                                (len(coords), 3, 3))
    for atom in range(3):
        for xyz in range(3):
            coords[:, atom, xyz] -= dx
            psi[:, 0, atom, xyz] = np.prod(psi_t(coords, excite, shift, interp, atoms), axis=1)
            coords[:, atom, xyz] += 2*dx
            psi[:, 2, atom, xyz] = np.prod(psi_t(coords, excite, shift, interp, atoms), axis=1)
            coords[:, atom, xyz] -= dx
    return psi


def local_kinetic_finite(Psi):
    dx = 1e-3
    d2psidx2 = ((Psi.psit[:, 0] - 2. * Psi.psit[:, 1] + Psi.psit[:, 2]) / dx ** 2) / Psi.psit[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin


def drift_fd(coords, excite, shift, interp, atoms):
    dx = 1e-3
    psi = all_da_psi(coords, excite, shift, interp, atoms)
    der = (psi[:, 2] - psi[:, 0])/dx/psi[:, 1]
    return der, psi


from Coordinerds.CoordinateSystems import *
def linear_combo_stretch_grid(r1, r2, coords):
    re = np.linalg.norm(coords[0]-coords[1])
    coords = np.array([coords] * 1)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).squeeze()
    zmat[:, 0, 1] = re + r1
    zmat[:, 1, 1] = re + r2
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


def grid_angle(a, b, num, coords):
    spacing = np.linspace(a, b, num)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    g = np.array([zmat]*num)
    g[:, 1, 3] = spacing
    new_coords = CoordinateSet(g, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


anti = np.load('antisymmetric_stretch_water_wvfns.npz')
sym = np.load('symmetric_stretch_water_wvfns.npz')
interp_anti = interpolate.splrep(anti['grid'], anti['excite'], s=0)
interp_sym = interpolate.splrep(sym['grid'], sym['ground'], s=0)
interp = [interp_anti, interp_sym]
molecule = np.array([
                        [0.,          0.,          0.],
                        [1.81005527,  0.00002,          0.],
                    [-0.4534464, 1.75233737, 0.],
])

anti = np.linspace(-0.75, 0.75, 50)
sym = np.zeros(50)
A = 1/np.sqrt(2)*np.array([[-1, 1], [1, 1]])
eh = np.matmul(np.linalg.inv(A), np.vstack((anti, sym)))
r1 = eh[0]
r2 = eh[1]

theta = np.deg2rad(104.1747712)
grid = grid_angle(theta-1, theta+1, 50, molecule)

# grid = linear_combo_stretch_grid(r1, r2, molecule)

# grid = np.array([molecule]*50)

psi = Walkers(50, molecule, None, [0, 0, 0], ['O', 'H', 'H'], interp)
psi.coords = grid

d, _ = drift(psi.coords, None, [0, 0, 0], ['O', 'H', 'H'], interp)
psi = pot(psi)
sigma = np.zeros((3, 3))
sigma[0] = np.array([[np.sqrt(dtau / m_O)] * 3])
sigma[1] = np.array([[np.sqrt(dtau / m_H)] * 3])
sigma[2] = np.array([[np.sqrt(dtau / m_H)] * 3])

psi = E_loc(psi, sigma)
# _, d2psi = local_kinetic(psi, sigma)

fd_d, psi.psit = drift_fd(psi.coords, psi.excite, psi.shift, psi.interp, psi.atoms)
# asdf, d2psi_fd = local_kinetic_finite(psi)
fd_eloc = local_kinetic_finite(psi) + psi.V
theta = np.linspace(theta-1, theta+1, 50)
anti = theta
import matplotlib.pyplot as plt
plt.plot(anti, psi.V*har2wave, label='potential')
plt.plot(anti, psi.El*har2wave, label='local energy')
plt.plot(anti, fd_eloc*har2wave, label='fd local energy')
plt.legend()
plt.show()




