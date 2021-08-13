import numpy as np
import matplotlib.pyplot as plt
from ProtWaterPES import Dipole
import multiprocessing as mp
from Eckart_turny_turn import EckartsSpinz
from PAF_spinz import MomentOfSpinz
from Coordinerds.CoordinateSystems import *
from scipy import interpolate


har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
au_to_Debye = 1/0.3934303
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_OH = (m_H*m_O)/(m_H+m_O)
omega_asym = 3815.044564/har2wave
mw = m_OH*omega_asym
conv_fac = 4.702e-7
km_mol = 5.33e6

ref = np.array([
  [0.000000000000000, 0.000000000000000, 0.000000000000000],
  [-2.304566686034061, 0.000000000000000, 0.000000000000000],
  [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
  [2.304566686034061, 0.000000000000000, 0.000000000000000],
  [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233],
])

mass = np.array([m_H, m_O, m_H, m_O, m_H])

MOM = MomentOfSpinz(ref, mass)
ref = MOM.coord_spinz()

big_Roo_grid = np.linspace(4, 5.4, 1000)
big_sp_grid = np.linspace(-1.2, 1.2, 1000)
big_sp_grid = np.linspace(-50, 50, 1000)
X, Y = np.meshgrid(big_sp_grid, big_Roo_grid)


z_ground_no_der = np.load('z_ground_no_der_new_def.npy')

ground_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
                                                       z_ground_no_der.flatten())

z_excite_xh_no_der = np.load('z_excite_xh_no_der_new_def.npy')

excite_xh_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
                                                          z_excite_xh_no_der.flatten())


def get_da_psi(coords, excite):
    psi = np.ones((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    # m = mw_h
    m = 1*omega_asym
    dead = -0.60594644269321474*dists[:, -4] + 42.200232187251913*dists[:, 0]
    dead2 = 41.561937672470521*dists[:, -4] + 1.0206303697659393*dists[:, 0]
    # dead = dists[:, 0]
    if excite == 'sp':
        psi[:, 0] = (m / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * m * dead ** 2))
        psi[:, 1] = excite_xh_no_der(dead2, dists[:, -2])
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'mod':
        psi[:, 0] = (m / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * m * dead ** 2))
        # psi[:, 1] = ground_no_der(dead2, dists[:, -2])
    elif excite == 'mod1':
        psi[:, 0] = (m / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * m * dead ** 2)) * \
                    (2 * m) ** (1 / 2) * dead
        # psi[:, 1] = ground_no_der(dead2, dists[:, -2])
    elif excite == 'a_only':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
    elif excite == 'ground':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
    else:
        psi[:, 0] = (m / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * m * dead ** 2))
        psi[:, 1] = ground_no_der(dead2, dists[:, -2])
    return np.prod(psi, axis=-1)


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0], [3, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    s_oh = 1/np.sqrt(2)*(dis[:, 0]+dis[:, 1])
    z = 1/2*(dis[:, -2] - dis[:, -1])
    mid = dis[:, 2]/2
    sp = mid - dis[:, -2]*np.cos(roh_roo_angle(coords, dis[:, -3], dis[:, -2]))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], s_oh, dis[:, -2], z, dis[:, -1], dis[:, -3], sp)).T


def roh_roo_angle(coords, roo_dist, roh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (coords[:, 1]-coords[:, 0])/np.broadcast_to(roh_dist[:, None], (len(roh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


class DipHolder:
    dip = None
    @classmethod
    def get_dip(cls, coords):
        if cls.dip is None:
            cls.dip = Dipole(coords.shape[1])
        return cls.dip.get_dipole(coords)


get_dip = DipHolder.get_dip


def dip(coords):
    coords = np.array_split(coords, mp.cpu_count()-1)
    V = pool.map(get_dip, coords)
    dips = np.concatenate(V)
    return dips

pool = mp.Pool(mp.cpu_count()-1)
walkers = 20000

xh_excite_neg_coords = np.zeros((5, walkers, 5, 3))
xh_excite_neg_erefs = np.zeros((4000))
xh_excite_neg_weights = np.zeros((5, walkers))
# for i in range(5):
blah = np.load(f'XH_left_excite_state_drifty_full_test_h3o2_{2}.npz')
coords = blah['coords']
eref = blah['Eref']
weights = blah['weights']
xh_excite_neg_coords = coords
xh_excite_neg_erefs = eref
xh_excite_neg_weights = weights

xh_excite_pos_coords = np.zeros((5, walkers, 5, 3))
xh_excite_pos_erefs = np.zeros((4000))
xh_excite_pos_weights = np.zeros((5, walkers))
# for i in range(5):
blah = np.load(f'XH_right_excite_state_drifty_full_test_h3o2_{2}.npz')
coords = blah['coords']
eref = blah['Eref']
weights = blah['weights']
xh_excite_pos_coords = coords
xh_excite_pos_erefs = eref
xh_excite_pos_weights = weights


freq1 = np.mean(xh_excite_pos_erefs[500:]*har2wave) - 6610
freq0 = np.mean(xh_excite_neg_erefs[500:]*har2wave) - 6610

freq2 = np.average((freq1, freq0))
print(freq2)


xh_excite_neg_coords = np.reshape(xh_excite_neg_coords, (5*walkers, 5, 3))
xh_excite_neg_coords = np.vstack((xh_excite_neg_coords, xh_excite_neg_coords[:, [0, 3, 4, 1, 2]]))
xh_excite_neg_weights = np.reshape(xh_excite_neg_weights, (5*walkers))
xh_excite_neg_weights = np.hstack((xh_excite_neg_weights, xh_excite_neg_weights))
xh_excite_neg_dips = np.zeros((5*walkers*2, 3))
# for i in range(5):
eck = EckartsSpinz(ref, xh_excite_neg_coords, mass, planar=True)
xh_excite_neg_coords = eck.get_rotated_coords()
xh_excite_neg_dips = dip(xh_excite_neg_coords)

xh_excite_pos_coords = np.reshape(xh_excite_pos_coords, (5*walkers, 5, 3))
xh_excite_pos_coords = np.vstack((xh_excite_pos_coords, xh_excite_pos_coords[:, [0, 3, 4, 1, 2]]))
xh_excite_pos_weights = np.reshape(xh_excite_pos_weights, (5*walkers))
xh_excite_pos_weights = np.hstack((xh_excite_pos_weights, xh_excite_pos_weights))
xh_excite_pos_dips = np.zeros((5*walkers*2, 3))
# for i in range(5):
eck = EckartsSpinz(ref, xh_excite_pos_coords, mass, planar=True)
xh_excite_pos_coords = eck.get_rotated_coords()
xh_excite_pos_dips = dip(xh_excite_pos_coords)

xh_term2 = np.zeros((3))
xh_term2_vec = np.zeros((3))
xh_term2_ov = np.zeros(1)
xh_term2_dis = np.zeros(1)
combine_dips = np.zeros((5*walkers*4, 3))
combine_weights = np.zeros((5*walkers*4))
xh_combine_dips = np.zeros((5*walkers*4, 3))
xh_combine_weights = np.zeros((5*walkers*4))
xh_combine_coords = np.vstack((xh_excite_neg_coords, xh_excite_pos_coords))
xh_combine_weights = np.hstack((xh_excite_neg_weights, xh_excite_pos_weights))
xh_combine_dips = np.vstack((xh_excite_neg_dips, xh_excite_pos_dips))
H0x = get_da_psi(xh_combine_coords, None)
H2 = get_da_psi(xh_combine_coords, 'sp')
frac2 = H0x/H2
xh_term2_ov = np.dot(xh_combine_weights, frac2)/np.sum(xh_combine_weights)
dists = all_dists(xh_combine_coords)
xh_term2_dis = np.dot(xh_combine_weights, frac2*dists[:, -1])/np.sum(xh_combine_weights)
for j in range(3):
    xh_term2[j] = np.dot(xh_combine_weights, frac2*xh_combine_dips[:, j] * au_to_Debye)\
                     /np.sum(xh_combine_weights)
    mid = (xh_combine_coords[:, 3, j] - xh_combine_coords[:, 1, j])/2
    xh_term2_vec[j] = np.dot(xh_combine_weights, frac2 * (mid - xh_combine_coords[:, 0, j])) \
                  / np.sum(xh_combine_weights)


# avg_xh_term2_v = np.average(xh_term2_vec, axis=0)
# std_xh_term2_v = np.std(xh_term2_vec, axis=0)
# avg_xh_term2_o = np.average(xh_term2_ov)
# std_xh_term2_o = np.std(xh_term2_ov)
# avg_xh_term2_d = np.average(xh_term2_dis)
# std_xh_term2_d = np.std(xh_term2_dis)

print(np.average(xh_term2, axis=0))
print(np.std(xh_term2, axis=0))
xh_term2 = np.linalg.norm(xh_term2, axis=-1)
# xh_std_term2 = np.std(xh_term2)
# xh_term2 = np.average(xh_term2)
print(xh_term2)
# print(xh_std_term2)

# std_term2_xh_sq = xh_term2**2*np.sqrt((xh_std_term2/xh_term2)**2 + (xh_std_term2/xh_term2)**2)
# xh_std_term2_sq_freq = xh_term2**2*freq2*np.sqrt((std_term2_xh_sq/xh_term2**2)**2 + (std_freq2/freq2)**2)
conversion = conv_fac*km_mol
print(conversion*xh_term2**2*freq2)

xh_term3 = 1.1154257246577732

print(conversion*xh_term3**2*freq2)
# print(xh_std_term2_sq_freq*conversion)