import numpy as np
from itertools import repeat
from scipy import interpolate
import multiprocessing as mp


bonds = 5
dx = 1.e-3


def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, bonds))
    # This method of calculating these is slightly faster than the linalg.norm method for small sample sizes
    for i in range(bonds):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch


def hh_dist(carts, rch):
    N = len(carts)
    coords = np.array(carts)
    # shift the carbon to the origin and everything else along with it
    coords -= np.broadcast_to(coords[:, None, 0], (N, bonds + 1, 3))
    # Normalize each of the bond lengths to 1
    coords[:, 1:] /= np.broadcast_to(rch[:, :, None], (N, bonds, 3))
    hh = np.zeros((N, 5, 5))
    # create a mask because I don't want the diagonals of this guy
    little_mask = np.full((5, 5), True)
    np.fill_diagonal(little_mask, False)
    mask = np.broadcast_to(little_mask, (N, 5, 5))
    # filling in the upper right triangle of hh distances for each walker
    for i in range(4):
        for j in np.arange(i + 1, 5):
            hh[:, i, j] = np.sqrt((coords[:, j + 1, 0] - coords[:, i + 1, 0]) ** 2 +
                                  (coords[:, j + 1, 1] - coords[:, i + 1, 1]) ** 2 +
                                  (coords[:, j + 1, 2] - coords[:, i + 1, 2]) ** 2)
    hh += np.transpose(hh, (0, 2, 1))
    # getting the actual standard deviations that I care about
    hh_std = np.std(hh[mask].reshape(N, 5, 4), axis=2)
    return hh_std


def all_da_psi(coords, rch, interp, imp_samp_type, interp_exp=None):
    # split up my walkers across the cores to calculate all this finite difference stuff faster
    coords = np.array_split(coords, mp.cpu_count() - 1)
    rch = np.array_split(rch, mp.cpu_count() - 1)
    psi = pool.starmap(get_da_psi, zip(coords, rch, repeat(interp), repeat(imp_samp_type), repeat(interp_exp)))
    psi = np.concatenate(psi)
    return psi


def get_da_psi(coords, rch, interp, imp_samp_type, interp_exp=None):
    much_psi = np.zeros((len(coords), 3, 6, 3))
    psi = psi_t_extra(coords, interp, imp_samp_type, interp_exp, rch=rch)  # calculate psi for this configuration
    much_psi[:, 1] = np.broadcast_to(np.prod(psi, axis=1)[:, None, None], (len(coords), 6, 3))  # throw that along the middle axis
    for atoms in range(6):
        for xyz in range(3):
            # move each cartesian coord back and forth and calculate psi for each movement
            coords[:, atoms, xyz] -= dx
            much_psi[:, 0, atoms, xyz] = np.prod(psi_t_extra(coords, interp, imp_samp_type, interp_exp), axis=1)
            coords[:, atoms, xyz] += 2.*dx
            much_psi[:, 2, atoms, xyz] = np.prod(psi_t_extra(coords, interp, imp_samp_type, interp_exp), axis=1)
            coords[:, atoms, xyz] -= dx
    return much_psi


def psi_t_extra(coords, interp, imp_samp_type, interp_exp=None, rch=None):
    if rch is None:
        rch = ch_dist(coords)
    if imp_samp_type == 'dev_dep':  # artifact of my general code that specifies the type of wavefunction I'm using for imp sampling
        hh = hh_dist(coords, rch)
    shift = np.zeros((len(coords), bonds))
    psi = np.zeros((len(coords), bonds))
    for i in range(bonds):
        if imp_samp_type == 'dev_dep':
            shift[:, i] = interpolate.splev(hh[:, i], interp_exp, der=0)  # shifts the wavefunction to be centered where the correlation says it should be
        psi[:, i] = interpolate.splev(rch[:, i] - shift[:, i], interp[i], der=0)
    return psi


def local_kinetic(Psi, sigmaCH, dtau):
    # an example of how I use that psit property of my Walkers object
    d2psidx2 = ((Psi.psit[:, 0] - 2. * Psi.psit[:, 1] + Psi.psit[:, 2]) / dx ** 2) / Psi.psit[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigmaCH ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin


pool = mp.Pool(mp.cpu_count()-1)


