import copy
import CH5pot
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp

# import Timing_p3 as tm

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num * me * 1000)
m_H = 1.007825 / (Avo_num * me * 1000)
m_CH = (m_C * m_H) / (m_H + m_C)
har2wave = 219474.6
ang2bohr = 1.e-10 / 5.291772106712e-11

# Starting orientation of walkers
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                           [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                           [1.786540362044548, -1.386051328559878, 0.000000000000000],
                           [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                           [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                           [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
bonds = 5
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]

Psi_t = np.load('params/min_wvfns/GSW_min_CH_1.npy')
x = np.linspace(0.4, 6., 5000)
if np.max(Psi_t) < 0.02:
    shift = x[np.argmin(Psi_t)]
else:
    shift = x[np.argmax(Psi_t)]
# shift = np.dot(Psi_t**2, x)
# shift = 2.2611644678388316
shift = 2.0930991957283878
x = x - shift
exp = np.load('params/sigma_hh_to_rch_cub_relationship.npy')
interp_exp = interpolate.splrep(exp[0, :], exp[1, :], s=0)
interp = interpolate.splrep(x, Psi_t, s=0)
dx = 1.e-6


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial] * walkers)
        rand_idx = np.random.rand(walkers, 5).argsort(axis=1) + 1
        b = self.coords[np.arange(walkers)[:, None], rand_idx]
        self.coords[:, 1:6, :] = b
        self.weights = np.ones(walkers)
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.psit = np.zeros((walkers, 3, 6, 3))


def hh_dist(carts, rch):
    N = len(carts)
    coords = np.array(carts)
    coords -= np.broadcast_to(coords[:, None, 0], (N, bonds + 1, 3))
    coords[:, 1:] /= np.broadcast_to(rch[:, :, None], (N, bonds, 3))
    hh = np.zeros((N, 5, 5))
    a = np.full((5, 5), True)
    np.fill_diagonal(a, False)
    mask = np.broadcast_to(a, (N, 5, 5))
    for i in range(4):
        for j in np.arange(i + 1, 5):
            hh[:, i, j] = np.sqrt((coords[:, j + 1, 0] - coords[:, i + 1, 0]) ** 2 +
                                  (coords[:, j + 1, 1] - coords[:, i + 1, 1]) ** 2 +
                                  (coords[:, j + 1, 2] - coords[:, i + 1, 2]) ** 2)
    hh += np.transpose(hh, (0, 2, 1))
    blah = hh[mask].reshape(N, 5, 4)
    hh_std = np.std(blah, axis=2)
    return hh_std


def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, bonds))
    for i in range(bonds):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch


def one_ch_dist(coords, CH):
    rch = np.sqrt((coords[:, CH, 0] - coords[:, 0, 0]) ** 2 +
                  (coords[:, CH, 1] - coords[:, 0, 1]) ** 2 +
                  (coords[:, CH, 2] - coords[:, 0, 2]) ** 2)
    return rch


def psi_t(coords):
    rch = ch_dist(coords)
    hh = hh_dist(coords, rch)
    psi = np.zeros((len(coords), bonds))
    shift = np.zeros((len(coords), bonds))
    for i in range(bonds):
        shift[:, i] = interpolate.splev(hh[:, i], interp_exp, der=0)
        psi[:, i] = interpolate.splev(rch[:, i] - shift[:, i], interp, der=0)
    return psi


def get_da_psi(coords):
    much_psi = np.zeros((len(coords), 3, 6, 3))
    psi = psi_t(coords)
    asdf = np.broadcast_to(np.prod(psi, axis=1)[:, None, None], (len(coords), 6, 3))
    much_psi[:, 1] += asdf
    for atoms in range(6):
        for xyz in range(3):
            coords[:, atoms, xyz] -= dx
            much_psi[:, 0, atoms, xyz] = np.prod(psi_t(coords), axis=1)
            coords[:, atoms, xyz] += 2. * dx
            much_psi[:, 2, atoms, xyz] = np.prod(psi_t(coords), axis=1)
            coords[:, atoms, xyz] -= dx
    return much_psi


def all_da_psi(coords):
    coords = np.array_split(coords, mp.cpu_count() - 1)
    psi = pool.map(get_da_psi, coords)
    psi = np.concatenate(psi)
    return psi


# Function for the potential for the mp to use
def get_pot(coords):
    V = CH5pot.mycalcpot(coords, len(coords))
    return V


pool = mp.Pool(mp.cpu_count()-1)
import Timing_p3 as tm
psi = Walkers(10000)
rch, rch_time = tm.time_me(ch_dist, psi.coords)
tm.print_time_list(ch_dist, rch_time)
hh, hh_time = tm.time_me(hh_dist, psi.coords, rch)
tm.print_time_list(hh_dist, hh_time)
psi_trial, trial_time = tm.time_me(psi_t, psi.coords)
tm.print_time_list(psi_t, trial_time)
psit, psit_time = tm.time_me(get_da_psi, psi.coords)
tm.print_time_list(get_da_psi, psit_time)
psi, v_time = tm.time_me(get_pot, psi.coords)
tm.print_time_list(get_pot, v_time)