import copy
from scipy import interpolate
import numpy as np
import multiprocessing as mp
import os
os.chdir('ProtWaterPES')
from ProtWaterPES.ProtonatedWaterPot import Potential
os.chdir('../')
from itertools import repeat
import time

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_OD = (m_D*m_O)/(m_D+m_O)
m_OH = (m_H*m_O)/(m_H+m_O)
dtau = 1
alpha = 1./(2.*dtau)
sigmaH = np.sqrt(dtau/m_H)
sigmaO = np.sqrt(dtau/m_O)
sigmaD = np.sqrt(dtau/m_D)
sigma = np.broadcast_to(np.array([sigmaH, sigmaO, sigmaH, sigmaO, sigmaH])[:, None], (5, 3))
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

#m_OH = 1

omega_asym = 3815.044564/har2wave
omega_asym_D = 2235.4632530938925/har2wave

omega_sym = 2704.021674298211/har2wave
omega_sym_D = 1968.55510602268/har2wave

new_struct = np.array([
    [2.06095307, 0.05378083, 0.],
    [0., 0., 0.],
    [-0.32643038, -1.70972841, 0.52193868],
    [4.70153912, 0., 0.],
    [5.20071798, 0.80543847, 1.55595785]
])

small_grid_points = 100
Roo_grid = np.linspace(3.9, 5.8, small_grid_points)
sp_grid = np.linspace(-65, 65, small_grid_points)
wvfns = np.load('2d_h3o2_new_def_100_points_no_cutoff.npz')['wvfns']

z_ground_no_der = wvfns[:, 0].reshape((small_grid_points, small_grid_points))
z_ground_dx1 = np.load('z_ground_dx1_mod_new_derivative_method.npy')
z_ground_dy1 = np.load('z_ground_dy1_mod_new_derivative_method.npy')
z_ground_dx2 = np.load('z_ground_dx2_mod_new_derivative_method.npy')
z_ground_dy2 = np.load('z_ground_dy2_mod_new_derivative_method.npy')
z_ground_dx1_dy1 = np.load('z_ground_dx1_dy1_mod_new_derivative_method.npy')

ground_no_der = interpolate.interp2d(sp_grid, Roo_grid, z_ground_no_der.T, kind='cubic')
ground_dx1 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dx1.T, kind='cubic')
ground_dx2 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dx2.T, kind='cubic')
ground_dy1 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dy1.T, kind='cubic')
ground_dy2 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dy2.T, kind='cubic')
ground_dx1_dy1 = interpolate.interp2d(sp_grid, Roo_grid, z_ground_dx1_dy1.T, kind='cubic')

z_excite_no_der = wvfns[:, 2].reshape((small_grid_points, small_grid_points))
z_excite_dx1 = np.load('z_excite_dx1_mod_new_derivative_method.npy')
z_excite_dy1 = np.load('z_excite_dy1_mod_new_derivative_method.npy')
z_excite_dx2 = np.load('z_excite_dx2_mod_new_derivative_method.npy')
z_excite_dy2 = np.load('z_excite_dy2_mod_new_derivative_method.npy')
z_excite_dx1_dy1 = np.load('z_excite_dx1_dy1_mod_new_derivative_method.npy')

excite_no_der = interpolate.interp2d(sp_grid, Roo_grid, z_excite_no_der.T, kind='cubic')
excite_dx1 = interpolate.interp2d(sp_grid, Roo_grid, z_excite_dx1.T, kind='cubic')
excite_dx2 = interpolate.interp2d(sp_grid, Roo_grid, z_excite_dx2.T, kind='cubic')
excite_dy1 = interpolate.interp2d(sp_grid, Roo_grid, z_excite_dy1.T, kind='cubic')
excite_dy2 = interpolate.interp2d(sp_grid, Roo_grid, z_excite_dy2.T, kind='cubic')
excite_dx1_dy1 = interpolate.interp2d(sp_grid, Roo_grid, z_excite_dx1_dy1.T, kind='cubic')


def interp(x, y, poiuy):
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = poiuy(x[i], y[i])
    return out


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct, excite, initial_shifts):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([initial_struct]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.excite = excite
        self.shift = initial_shifts
        self.psit = np.zeros((walkers, 3, 5, 3))


def psi_t(coords, excite):
    psi = np.ones((len(coords), 2))
    dists = all_dists(coords)
    mw_h = omega_asym
    dead = -0.60594644269321474*dists[:, -1] + 42.200232187251913*dists[:, 0]
    dead2 = 41.561937672470521*dists[:, -1] + 1.0206303697659393*dists[:, 0]
    if excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dead ** 2))
        psi[:, 1] = interp(dead2, dists[:, -2], excite_no_der)
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dead ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dead
        psi[:, 1] = interp(dead2, dists[:, -2], ground_no_der)
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dead ** 2))
        psi[:, 1] = interp(dead2, dists[:, -2], ground_no_der)
    return psi


def dpsidasp(coords, excite, dists):
    collect = np.zeros((len(coords), 3))
    mw_h = omega_asym
    dead = -0.60594644269321474 * dists[:, -1] + 42.200232187251913 * dists[:, 0]
    dead2 = 41.561937672470521 * dists[:, -1] + 1.0206303697659393 * dists[:, 0]
    if excite == 'sp':
        collect[:, 0] = -mw_h*dead
        collect[:, 2] = interp(dead2, dists[:, -2], excite_dx1)
        collect[:, 1] = interp(dead2, dists[:, -2], excite_dy1)
    elif excite == 'a':
        collect[:, 0] = (1 - mw_h*dead**2)/dead
        collect[:, 2] = interp(dead2, dists[:, -2], ground_dx1)
        collect[:, 1] = interp(dead2, dists[:, -2], ground_dy1)
    else:
        collect[:, 0] = -mw_h*dead
        collect[:, 2] = interp(dead2, dists[:, -2], ground_dx1)
        collect[:, 1] = interp(dead2, dists[:, -2], ground_dy1)
    return collect


def d2psidasp(coords, excite, dists):
    collect = np.zeros((len(coords), 4))
    mw_h = omega_asym
    dead = -0.60594644269321474 * dists[:, -1] + 42.200232187251913 * dists[:, 0]
    dead2 = 41.561937672470521 * dists[:, -1] + 1.0206303697659393 * dists[:, 0]
    if excite == 'sp':
        collect[:, 0] = mw_h**2*dead**2 - mw_h
        collect[:, 2] = interp(dead2, dists[:, -2], excite_dx2)
        collect[:, 1] = interp(dead2, dists[:, -2], excite_dy2)
        collect[:, 3] = interp(dead2, dists[:, -2], excite_dx1_dy1)
    elif excite == 'a':
        collect[:, 0] = mw_h*(mw_h*dead**2 - 3)
        collect[:, 2] = interp(dead2, dists[:, -2], ground_dx2)
        collect[:, 1] = interp(dead2, dists[:, -2], ground_dy2)
        collect[:, 3] = interp(dead2, dists[:, -2], ground_dx1_dy1)
    else:
        collect[:, 0] = mw_h**2*dead**2 - mw_h
        collect[:, 2] = interp(dead2, dists[:, -2], ground_dx2)
        collect[:, 1] = interp(dead2, dists[:, -2], ground_dy2)
        collect[:, 3] = interp(dead2, dists[:, -2], ground_dx1_dy1)
    return collect


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    mid = (coords[:, 1] + coords[:, 3])/2
    rxh = coords[:, 0] - mid
    rxh_dist = np.linalg.norm(rxh, axis=-1)
    sp = rxh_dist*np.cos(roh_roo_angle(coords, rxh, dis[:, -2], rxh_dist))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], dis[:, -2], sp)).T


def roh_roo_angle(coords, rxh, roo_dist, rxh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (rxh)/np.broadcast_to(rxh_dist[:, None], (len(rxh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


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
        chain[:, 2*bond + 1, :, 0] = (-1)**bond*1/np.sqrt(2)*(1./dists[:, bond, None]
                                                              - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2
                                                              /dists[:, bond, None]**3)
        chain[:, 2*bond + 2, :, 0] = (-1)**bond*1/np.sqrt(2)*(1./dists[:, bond, None]
                                                              - (coords[:, 2*bond + 1]-coords[:, 2*bond + 2])**2
                                                              /dists[:, bond, None]**3)
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
    mid = (coords[:, 1] + coords[:, 3]) / 2
    rxh = coords[:, 0] - mid
    rxh_dist = np.linalg.norm(rxh, axis=1)
    sp = rxh_dist * np.cos(roh_roo_angle(coords, rxh, dis[:, -2], rxh_dist))
    return sp


def dudx(coords, dists):
    dxh = dspdx(coords)
    da = daroodx(coords, dists[:, [1, 2, -2]])[..., 0]
    return -0.60594644269321474 * dxh + 42.200232187251913 * da


def dvdx(coords, dists):
    dxh = dspdx(coords)
    da = daroodx(coords, dists[:, [1, 2, -2]])[..., 0]
    return 41.561937672470521 * dxh + 1.0206303697659393 * da


def d2dudx(coords, dists):
    d2xh = d2spdx2(coords, dists[:, -1])
    d2a = daroodx2(coords, dists[:, [1, 2, -2]])[..., 0]
    return -0.60594644269321474 * d2xh + 42.200232187251913 * d2a


def d2vdx(coords, dists):
    d2xh = d2spdx2(coords, dists[:, -1])
    d2a = daroodx2(coords, dists[:, [1, 2, -2]])[..., 0]
    return 41.561937672470521 * d2xh + 1.0206303697659393 * d2a


def dpsidx(coords, excite):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])
    droox[..., 0] = dudx(coords, dists)
    dspx = dvdx(coords, dists)
    dr = np.concatenate((droox, dspx[..., None]), axis=-1)
    collect = dpsidasp(coords, excite, dists)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def d2psidx2(coords, excite):
    dists = all_dists(coords)
    droox = daroodx(coords, dists[:, [1, 2, -2]])
    droox[..., 0] = dudx(coords, dists)
    dspx = dvdx(coords, dists)
    dr1 = np.concatenate((droox, dspx[..., None]), axis=-1)
    droox2 = daroodx2(coords, dists[:, [1, 2, -2]])
    droox2[..., 0] = d2dudx(coords, dists)
    dspx2 = d2vdx(coords, dists)
    dr2 = np.concatenate((droox2, dspx2[..., None]), axis=-1)
    first_dir = dpsidasp(coords, excite, dists)
    second_dir = d2psidasp(coords, excite, dists)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1 ** 2, second_dir[:, None, 0:3, None]).squeeze()
    part3 = dr1[..., 0] * dr1[..., 1] * np.broadcast_to(second_dir[:, -1, None, None], (len(dr1), 5, 3)).squeeze()
    part4 = np.matmul(np.broadcast_to(dr1[..., 0, None], droox.shape)*dr1[..., [1, 2]],
                      (np.broadcast_to(first_dir[:, 0, None],
                       first_dir[:, [1, 2]].shape)*first_dir[:, [1, 2]])[:, None, :, None]).squeeze()
    return part1 + part2 + 2*part3 + 2*part4


def drift_no_parr(coords, excite):
    psi = psi_t(coords, excite)
    der = 2*dpsidx(coords, excite)
    return der, psi


def metropolis(Fqx, Fqy, x, y, psi1, psi2):
    psi_1 = np.prod(psi1, axis=1)
    psi_2 = np.prod(psi2, axis=1)
    psi_ratio = (psi_2/psi_1)**2
    a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
    a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
    remove = np.argwhere(psi_2 * psi_1 <= 0)
    a[remove] = 0.
    return a


# Random walk of all the walkers
def Kinetic(Psi):
    randomwalk = np.random.normal(0.0, sigma, size=(len(Psi.coords), sigma.shape[0], sigma.shape[1]))
    x = np.array(Psi.coords) + randomwalk
    Fqx, psi_check = drift(x, Psi.excite)
    Drift = sigma**2/2.*Fqx
    y = randomwalk + Drift + np.array(Psi.coords)
    Fqy, psi = drift(y, Psi.excite)
    a = metropolis(Fqx, Fqy, Psi.coords, y, Psi.psit, psi)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    Psi.psit[accept] = psi[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqx, acceptance


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def pot(Psi):
    #start = time.time()
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    #end = time.time()
    #print(f'potential = {end-start}', flush=True)
    return Psi


def drift(coords, excite):
    #start =  time.time()
    coordz = np.array_split(coords, mp.cpu_count()-1)
    psi = pool.starmap(psi_t, zip(coordz, repeat(excite)))
    psi = np.concatenate(psi)
    der = 2*np.concatenate(pool.starmap(dpsidx, zip(coordz, repeat(excite))))
    #end = time.time()
    #print(f'drift time = {end-start}', flush=True)
    return der, psi


def local_kinetic(Psi):
    #start = time.time()
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    d2psi = pool.starmap(d2psidx2, zip(coords, repeat(Psi.excite)))
    d2psi = np.concatenate(d2psi)
    kin = -1/2 * np.sum(np.sum(sigma**2/dtau*d2psi, axis=1), axis=1)
    #end = time.time()
    #print(f'local kinetic time = {end-start}', flush=True)
    return kin



pool = mp.Pool(mp.cpu_count()-1)


def E_loc(Psi):
    d2psi = d2psidx2(Psi.coords, Psi.excite)
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psi, axis=1), axis=1)
    return kin


def E_loc(Psi):
    Psi.El = local_kinetic(Psi) + Psi.V
    return Psi


def E_ref_calc(Psi):
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
        Biggo_psit = np.array(Psi.psit[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Fqx[i[0]] = Biggo_force
        Psi.psit[i[0]] = Biggo_psit

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
        Biggo_psit = np.array(Psi.psit[i[0]])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[ind] = Biggo_pos
        Psi.V[ind] = Biggo_pot
        Psi.El[ind] = Biggo_el
        Fqx[ind] = Biggo_force
        Psi.psit[ind] = Biggo_psit
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


def run(N_0, time_steps, propagation, equilibration, wait_time, excite, initial_struct, initial_shifts, shift_rate):
    # import time
    # start = time.time()
    DW = False
    psi = Walkers(N_0, initial_struct, excite, initial_shifts)
    Fqx, psi.psit = drift(psi.coords, psi.excite)
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
            # end = time.time()
            # print(end - start, flush=True)

        if DW is False:
            prop = float(propagation)
            wait -= 1.
        else:
            prop -= 1.

        if i == 0:
            psi = pot(psi)
            psi = E_loc(psi)
            Eref = E_ref_calc(psi)

        psi, Fqx, acceptance = Kinetic(psi)
        shift[i + 1] = psi.shift
        psi = pot(psi)
        psi = E_loc(psi)

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


# def twod_pot(coords, grid1, grid2):
#     print('started making our grid')
#     mesh = np.array(np.meshgrid(grid1, grid2))
#     gridz = np.reshape(mesh, (2, len(grid1)*len(grid2)))
#     roo_coords = oo_grid(coords, gridz[1])
#     full_coords = shared_prot_grid(roo_coords, gridz[0])
#     print('finished making the grid, now to start the potential')
#     mid = (full_coords[:, 3, 0] - full_coords[:, 1, 0])/2
#     full_coords[:, :, 0] -= mid[:, None]
#     pot = get_pot(full_coords)
#     np.save('coords_for_testing', full_coords)
#     print('finished evaluating the potential')
#     import scipy.sparse as sp
#     return sp.diags([pot], [0]), pot, full_coords
#
#
# def oo_grid(coords, Roo):
#     coords = np.array([coords] * len(Roo))
#     equil_roo_roh_x = coords[0, 3, 0] - coords[0, 4, 0]
#     coords[:, 3, 0] = Roo
#     coords[:, 4, 0] = Roo - equil_roo_roh_x
#     return coords
#
#
# def shared_prot_grid(coords, sp):
#     mid = (coords[:, 3, 0] - coords[:, 1, 0])/2
#     coords[:, 0, 0] = mid-sp
#     return coords
#
#
# new_struct = np.array([
#     [0.000000000000000, 0.000000000000000, 0.000000000000000],
#     [-2.304566686034061, 0.000000000000000, 0.000000000000000],
#     [-2.740400260927908, 1.0814221449986587E-016, -1.866154718409233],
#     [2.304566686034061, 0.000000000000000, 0.000000000000000],
#     [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233]
# ])
# new_struct[:, 0] = new_struct[:, 0] + 2.304566686034061
#
# big_sp_grid = np.linspace(-0.5, 0.5, 100)
# big_Roo_grid = np.linspace(4.6, 5.2, 100)
#
# blah, V, coords = twod_pot(new_struct, big_sp_grid, big_Roo_grid)
# #
# psi = Walkers(len(coords), 0, None, [0.33, 0])
# psi.coords = coords
# psi.V = V
# psi.psit = psi_t(psi.coords, psi.excite, psi.shift)
# psi = E_loc(psi)
# psi.El = psi.El*har2wave
# psit = psi.psit[:, 1, 0, 0]
# #
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# tcc = ax.contourf(big_sp_grid, big_Roo_grid, psi.El.reshape((len(big_Roo_grid), len(big_sp_grid))))
# fig.colorbar(tcc)
# plt.show()

test_structure = np.array([
        [ 2.75704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.07502706, -1.66894299, -0.69579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.84391277,  0.14767547,  1.4669121 ],
])

test_structure2 = np.array([
        [ 2.48704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.09502706, -1.86894299, -0.69579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.24391277,  0.14767547,  1.4669121 ],
])

# wow_im_tired = True
# trials = [4, 6, 8, 9]
# for i in range(5):
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, None, test_structure, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'ground_state_full_h3o2_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
#
# coords = np.load('coords_for_testing.npy')
# print(coords.shape)
#
# psi = Walkers(10000, test_structure2, 'a', [0, 2.5721982410729867])
# psi.coords = coords
# psi.coords[:, -1, 0] += 0.02
# psi.psit = psi_t(psi.coords, psi.excite, psi.shift)
# psi = pot(psi)
# psi = E_loc(psi)
#
# import matplotlib.pyplot as plt
# roo = np.linspace(3.9, 5.8, 100)
# xh = np.linspace(-1.5, 1.5, 100)
#
# X, Y = np.meshgrid(xh, roo)
#
# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, psi.El.reshape((100, 100)))
# fig.colorbar(tcc)
# plt.xlabel('XH')
# plt.ylabel('Roo')
# plt.show()
#
# psi = Walkers(10000, test_structure2, 'sp', [0, 2.5721982410729867])
# psi.coords = coords
# psi.psit = psi_t(psi.coords, psi.excite, psi.shift)
# psi = pot(psi)
# psi = E_loc(psi)
#
# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, psi.El.reshape((100, 100)))
# fig.colorbar(tcc)
# plt.xlabel('XH')
# plt.ylabel('Roo')
# plt.show()
#
# psi = Walkers(10000, test_structure2, 'a', [0, 2.5721982410729867])
# psi.coords = coords
# psi.psit = psi_t(psi.coords, psi.excite, psi.shift)
# psi = pot(psi)
# psi = E_loc(psi)
#
# fig, ax = plt.subplots()
# tcc = ax.contourf(X, Y, psi.El.reshape((100, 100)))
# fig.colorbar(tcc)
# plt.xlabel('XH')
# plt.ylabel('Roo')
# plt.show()
# xh_left = [0, 1, 2, 3, 4]
# for i in xh_left:
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, 'sp', test_structure, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'XH_excite_state_full_h3o2_left_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)

# for i in range(5):
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, None, test_structure2, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'ground_state_full_h3o2_{i+6}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
#
# xh_right = [0]
# for i in xh_right:
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, 'sp', test_structure2, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'XH_excite_state_full_h3o2_right_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
# asym_left = [0, 1, 2, 3, 4]
# for i in asym_left:
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, 'a', test_structure, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'Asym_excite_state_full_h3o2_left_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
#
# asym_right = [0, 1, 2, 3, 4]
#
# for i in asym_right:
#     coords, weights, time, Eref_array, sum_weights, accept, des = run(
#         5000, 20000, 250, 500, 500, 'a', test_structure2, [0, 2.5721982410729867], [0, 0]
#     )
#     np.savez(f'Asym_excite_state_full_h3o2_right_{i+1}', coords=coords, weights=weights, time=time, Eref=Eref_array,
#              sum_weights=sum_weights, accept=accept, d=des)
