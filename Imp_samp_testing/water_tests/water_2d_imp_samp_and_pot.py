from Potential.Water_monomer_pot_fns import PatrickShinglePotential
import numpy as np
import sys
path = sys.path.insert(0, '../../')
from Imp_samp_testing import Derivatives
from scipy import interpolate


wvfns = np.load('2d_anti_sym_stretch_water_wvfns.npz')
gridz = wvfns['grid']
ground = wvfns['excite_anti'].reshape((len(gridz[0]), len(gridz[1])))

ground_ders = Derivatives(ground, grid1=gridz[0], grid2=gridz[1])
z_ground_dx1 = ground_ders.compute_derivative(dx=1)/ground
z_ground_dx2 = ground_ders.compute_derivative(dx=2)/ground
z_ground_dx1_dy1 = ground_ders.compute_derivative(dx=1, dy=1)/ground
z_ground_dy1 = ground_ders.compute_derivative(dy=1)/ground
z_ground_dy2 = ground_ders.compute_derivative(dy=2)/ground

ground_no_der = interpolate.interp2d(gridz[0], gridz[1], ground.T, kind='cubic')
ground_dx1 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dx1.T, kind='cubic')
ground_dx2 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dx2.T, kind='cubic')
ground_dy1 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dy1.T, kind='cubic')
ground_dy2 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dy2.T, kind='cubic')
ground_dx1_dy1 = interpolate.interp2d(gridz[0], gridz[1], z_ground_dx1_dy1.T, kind='cubic')


def interp(x, y, poiuy):
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = poiuy(x[i], y[i])
    return out


def psi_t(coords):
    psi = np.ones((len(coords), 1))
    dists = oh_dists(coords)
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    psi[:, 0] = interp(anti, sym, ground_no_der)
    return np.prod(psi, axis=1)


def dpsidx(coords):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)
    collect = dpsidrtheta(coords, dists)
    return np.matmul(drx, collect[:, None, :, None]).squeeze()


def duvdx(drx):
    return 1/np.sqrt(2)*np.concatenate(((drx[..., 1, None] - drx[..., 0, None]),
                                        (drx[..., 0, None] + drx[..., 1, None])), axis=-1)


def d2psidx2(coords):
    dists = oh_dists(coords)
    drx = drdx(coords, dists)
    drx = duvdx(drx)
    drx2 = drdx2(coords, dists)
    drx2 = duvdx(drx2)
    first_dir = dpsidrtheta(coords, dists)
    second_dir = d2psidrtheta(coords, dists)
    part1 = np.matmul(drx2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul((drx**2), second_dir[:, None, :-1, None]).squeeze()
    part3 = np.matmul((drx[..., 0]*drx[..., 1])[..., None], second_dir[:, None, -1, None, None]).squeeze()
    return part1 + part2 + 2*part3


def dpsidrtheta(coords, dists):
    collect = np.zeros((len(coords), 2))
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    collect[:, 0] = interp(anti, sym, ground_dx1)
    collect[:, 1] = interp(anti, sym, ground_dy1)
    return collect


def d2psidrtheta(coords, dists):
    collect = np.zeros((len(coords), 3))
    anti = 1/np.sqrt(2)*(dists[:, 1] - dists[:, 0])
    sym = 1/np.sqrt(2)*(dists[:, 1] + dists[:, 0])
    collect[:, 0] = interp(anti, sym, ground_dx2)
    collect[:, 1] = interp(anti, sym, ground_dy2)
    collect[:, 2] = interp(anti, sym, ground_dx1_dy1)
    return collect


def oh_dists(coords):
    bonds = [[1, 2], [1, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def drdx(coords, dists):
    chain = np.zeros((len(coords), 3, 3, 2))
    for bond in range(2):
        chain[:, 0, :, bond] += ((coords[:, 0]-coords[:, bond+1])/dists[:, bond, None])
        chain[:, bond+1, :, bond] += ((coords[:, bond+1]-coords[:, 0])/dists[:, bond, None])
    return chain


def drdx2(coords, dists):
    chain = np.zeros((len(coords), 3, 3, 2))
    for bond in range(2):
        chain[:, 0, :, bond] = (1./dists[:, bond, None] - (coords[:, 0]-coords[:, bond+1])**2/dists[:, bond, None]**3)
        chain[:, bond + 1, :, bond] = (1./dists[:, bond, None] - (coords[:, bond + 1] - coords[:, 0])**2 / dists[:, bond, None]**3)
    return chain


def full_dpsi_dx(coords):
    first_derivative = dpsidx(coords)
    second_derivative = d2psidx2(coords)
    return first_derivative, second_derivative


def get_pot(coords):
    V = PatrickShinglePotential(coords)
    return V