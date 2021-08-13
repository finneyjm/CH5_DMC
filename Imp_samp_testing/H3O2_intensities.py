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

ref = np.array([
  [0.000000000000000, 0.000000000000000, 0.000000000000000],
  [-2.304566686034061, 0.000000000000000, 0.000000000000000],
  [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
  [2.304566686034061, 0.000000000000000, 0.000000000000000],
  [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233],
])

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)

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

def a_prime(a, z):
    return -0.60594644269321474*z + 42.200232187251913*a

def z_prime(a, z):
    return 41.561937672470521*z + 1.0206303697659393*a

def get_da_psi(coords, excite):
    psi = np.ones((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    # m = mw_h
    m = 1*omega_asym
    dead = -0.60594644269321474*dists[:, -1] + 42.200232187251913*dists[:, 0]
    dead2 = 41.561937672470521*dists[:, -1] + 1.0206303697659393*dists[:, 0]
    # dead = dists[:, 0]
    # dead2 = dists[:, -1]
    if excite == 'sp':
        psi[:, 0] = (m / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * m * dead ** 2))
        psi[:, 1] = excite_xh_no_der(dead2, dists[:, -2])
    elif excite == 'a':
        psi[:, 0] = (m / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * m * dead ** 2)) * \
                    (2 * m) ** (1 / 2) * dead
        psi[:, 1] = ground_no_der(dead2, dists[:, -2])
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


dis1 = all_dists(np.array([test_structure]*1))
dis2 = all_dists(np.array([test_structure2]*1))
dead1 = -0.29339998970198611*dis1[:, -1] + 0.95598977298027321*dis1[:, 0]
dead2 = -0.29339998970198611*dis2[:, -1] + 0.95598977298027321*dis2[:, 0]

print(dis1[:, [0, -1]])
print(dis2[:, [0, -1]])
print(dead1)
print(dead2)

pool = mp.Pool(mp.cpu_count()-1)

walkers = 20000

ground_coords = np.zeros((10, 27, walkers, 5, 3))
ground_erefs = np.zeros((10, 20000))
ground_weights = np.zeros((10, 27, walkers))
for i in range(10):
    blah = np.load(f'ground_excite_state_drifty_full_h3o2_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    ground_coords[i] = coords
    ground_erefs[i] = eref
    ground_weights[i] = weights

print(np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_zpe = np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_zpe = np.std(np.mean(ground_erefs[:, 5000:]*har2wave, axis=1))
print(std_zpe)

# walkers = 5000
excite_neg_coords = np.zeros((5, 27, walkers, 5, 3))
excite_neg_erefs = np.zeros((5, 20000))
excite_neg_weights = np.zeros((5, 27, walkers))
for i in range(5):
    blah = np.load(f'asym_left_excite_state_drifty_full_h3o2_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    excite_neg_coords[i] = coords
    excite_neg_erefs[i] = eref
    excite_neg_weights[i] = weights

print(np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_excite_neg_energy = np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_excite_neg_energy = np.std(np.mean(excite_neg_erefs[:, 5000:]*har2wave, axis=1))

excite_pos_coords = np.zeros((5, 27, walkers, 5, 3))
excite_pos_erefs = np.zeros((5, 20000))
excite_pos_weights = np.zeros((5, 27, walkers))
for i in range(5):
    blah = np.load(f'asym_right_excite_state_drifty_full_h3o2_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    excite_pos_coords[i] = coords
    excite_pos_erefs[i] = eref
    excite_pos_weights[i] = weights

print(np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_excite_pos_energy = np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_excite_pos_energy = np.std(np.mean(excite_pos_erefs[:, 5000:]*har2wave, axis=1))

print(average_excite_neg_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_neg_energy**2))
print(average_excite_pos_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_pos_energy**2))

average_excite_energy = np.average(np.array([average_excite_pos_energy, average_excite_neg_energy]))
std_excite_energy = np.sqrt(std_excite_pos_energy**2 + std_excite_neg_energy**2)
print(average_excite_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_energy**2))

xh_excite_neg_coords = np.zeros((5, 27, walkers, 5, 3))
xh_excite_neg_erefs = np.zeros((5, 20000))
xh_excite_neg_weights = np.zeros((5, 27, walkers))
for i in range(5):
    blah = np.load(f'XH_left_excite_state_drifty_full_h3o2_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    xh_excite_neg_coords[i] = coords
    xh_excite_neg_erefs[i] = eref
    xh_excite_neg_weights[i] = weights

print(np.mean(np.mean(xh_excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_xh_excite_neg_energy = np.mean(np.mean(xh_excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_xh_excite_neg_energy = np.std(np.mean(xh_excite_neg_erefs[:, 5000:]*har2wave, axis=1))

xh_excite_pos_coords = np.zeros((5, 27, walkers, 5, 3))
xh_excite_pos_erefs = np.zeros((5, 20000))
xh_excite_pos_weights = np.zeros((5, 27, walkers))
for i in range(5):
    blah = np.load(f'XH_right_excite_state_drifty_full_h3o2_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    xh_excite_pos_coords[i] = coords
    xh_excite_pos_erefs[i] = eref
    xh_excite_pos_weights[i] = weights

print(np.mean(np.mean(xh_excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_xh_excite_pos_energy = np.mean(np.mean(xh_excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_xh_excite_pos_energy = np.std(np.mean(xh_excite_pos_erefs[:, 5000:]*har2wave, axis=1))

print(average_xh_excite_neg_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_xh_excite_neg_energy**2))
print(average_xh_excite_pos_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_xh_excite_pos_energy**2))

# fig, ax = plt.subplots(2, 2, figsize=(15, 15))
# d = np.zeros((27, 20000, 9))
# for i in range(27):
#     d[i] = all_dists(excite_neg_coords[0, i])
# a = d[:, :, 0]
# xh = d[:, :, -1]
# a_p = 42.200232187251913*a - 0.60594644269321474*xh
# z_p = 41.561937672470521*xh + 1.0206303697659393*a
#
# binz=75
# for i in np.arange(0, 27, 3):
#     amp, xx = np.histogram(a[i], weights=excite_neg_weights[0, i], bins=binz, range=(-0.7, 0), density=True)
#     xx = (xx[1:] + xx[:-1])/2
#     ax[0, 0].plot(xx, amp, label=f'{i}th wvfn')
#     amp, xx = np.histogram(xh[i], weights=excite_neg_weights[0, i], bins=binz, range=(-1, 1), density=True)
#     xx = (xx[1:] + xx[:-1]) / 2
#     ax[1, 0].plot(xx, amp, label=f'{i}th wvfn')
#     amp, xx = np.histogram(a_p[i], weights=excite_neg_weights[0, i], bins=binz, range=(-20, 0), density=True)
#     xx = (xx[1:] + xx[:-1]) / 2
#     ax[0, 1].plot(xx, amp, label=f'{i}th wvfn')
#     amp, xx = np.histogram(z_p[i], weights=excite_neg_weights[0, i], bins=binz, range=(-40, 40), density=True)
#     xx = (xx[1:] + xx[:-1]) / 2
#     ax[1, 1].plot(xx, amp, label=f'{i}th wvfn')
# ax[0, 0].set_xlabel('a Bohr')
# ax[1, 0].set_xlabel('xh Bohr')
# ax[0, 1].set_xlabel("a' ")
# ax[1, 1].set_xlabel("z' ")
# ax[1, 0].set_ylim(0, 1.2)
# ax[1, 1].set_ylim(0, 0.03)
# plt.legend()
# plt.tight_layout()
# plt.savefig('demonstration_of_bad_data3')
# plt.show()
#
# fig, ax = plt.subplots(2, 2, figsize=(15, 15))
# d = np.zeros((27, 20000, 9))
# for i in range(27):
#     d[i] = all_dists(xh_excite_neg_coords[1, i])
# a = d[:, :, 0]
# xh = d[:, :, -1]
# a_p = 42.200232187251913*a - 0.60594644269321474*xh
# z_p = 41.561937672470521*xh + 1.0206303697659393*a
#
# binz=75
# for i in np.arange(0, 27, 3):
#     amp, xx = np.histogram(a[i], weights=xh_excite_neg_weights[1, i], bins=binz, range=(-0.7, 0.7), density=True)
#     xx = (xx[1:] + xx[:-1])/2
#     ax[0, 0].plot(xx, amp, label=f'{i}th wvfn')
#     amp, xx = np.histogram(xh[i], weights=xh_excite_neg_weights[1, i], bins=binz, range=(-0.05, 1), density=True)
#     xx = (xx[1:] + xx[:-1]) / 2
#     ax[1, 0].plot(xx, amp, label=f'{i}th wvfn')
#     amp, xx = np.histogram(a_p[i], weights=xh_excite_neg_weights[1, i], bins=binz, range=(-20, 20), density=True)
#     xx = (xx[1:] + xx[:-1]) / 2
#     ax[0, 1].plot(xx, amp, label=f'{i}th wvfn')
#     amp, xx = np.histogram(z_p[i], weights=xh_excite_neg_weights[1, i], bins=binz, range=(-5, 40), density=True)
#     xx = (xx[1:] + xx[:-1]) / 2
#     ax[1, 1].plot(xx, amp, label=f'{i}th wvfn')
# ax[0, 0].set_xlabel('a Bohr')
# ax[1, 0].set_xlabel('xh Bohr')
# ax[0, 1].set_xlabel("a' ")
# ax[1, 1].set_xlabel("z' ")
# # ax[1, 0].set_ylim(0, 1.2)
# # ax[1, 1].set_ylim(0, 0.03)
# plt.legend()
# plt.tight_layout()
# plt.savefig('demonstration_of_bad_data4')
# plt.show()


average_xh_excite_energy = np.average(np.array([average_xh_excite_pos_energy, average_xh_excite_neg_energy]))
std_xh_excite_energy = np.sqrt(std_xh_excite_pos_energy**2 + std_xh_excite_neg_energy**2)
print(average_xh_excite_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_xh_excite_energy**2))
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


ground_coords = np.reshape(ground_coords, (10, 27*walkers, 5, 3))
ground_coords = np.hstack((ground_coords, ground_coords[:, :, [0, 3, 4, 1, 2]]))
ground_weights = np.reshape(ground_weights, (10, 27*walkers))
ground_weights = np.hstack((ground_weights, ground_weights))
ground_dips = np.zeros((10, 27*walkers*2, 3))
for i in range(10):
    eck = EckartsSpinz(ref, ground_coords[i], mass, planar=True)
    ground_coords[i] = eck.get_rotated_coords()
    ground_dips[i] = dip(ground_coords[i])

excite_neg_coords = np.reshape(excite_neg_coords, (5, 27*walkers, 5, 3))
excite_neg_coords = np.hstack((excite_neg_coords, excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
excite_neg_weights = np.reshape(excite_neg_weights, (5, 27*walkers))
excite_neg_weights = np.hstack((excite_neg_weights, excite_neg_weights))
excite_neg_dips = np.zeros((5, 27*walkers*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, excite_neg_coords[i], mass, planar=True)
    excite_neg_coords[i] = eck.get_rotated_coords()
    excite_neg_dips[i] = dip(excite_neg_coords[i])

excite_pos_coords = np.reshape(excite_pos_coords, (5, 27*walkers, 5, 3))
excite_pos_coords = np.hstack((excite_pos_coords, excite_pos_coords[:, :, [0, 3, 4, 1, 2]]))
excite_pos_weights = np.reshape(excite_pos_weights, (5, 27*walkers))
excite_pos_weights = np.hstack((excite_pos_weights, excite_pos_weights))
excite_pos_dips = np.zeros((5, 27*walkers*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, excite_pos_coords[i], mass, planar=True)
    excite_pos_coords[i] = eck.get_rotated_coords()
    excite_pos_dips[i] = dip(excite_pos_coords[i])
    # combine_weights = np.hstack((excite_pos_weights[i], excite_neg_weights[i]))


xh_excite_neg_coords = np.reshape(xh_excite_neg_coords, (5, 27*walkers, 5, 3))
xh_excite_neg_coords = np.hstack((xh_excite_neg_coords, xh_excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
xh_excite_neg_weights = np.reshape(xh_excite_neg_weights, (5, 27*walkers))
xh_excite_neg_weights = np.hstack((xh_excite_neg_weights, xh_excite_neg_weights))
xh_excite_neg_dips = np.zeros((5, 27*walkers*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, xh_excite_neg_coords[i], mass, planar=True)
    xh_excite_neg_coords[i] = eck.get_rotated_coords()
    xh_excite_neg_dips[i] = dip(xh_excite_neg_coords[i])

xh_excite_pos_coords = np.reshape(xh_excite_pos_coords, (5, 27*walkers, 5, 3))
xh_excite_pos_coords = np.hstack((xh_excite_pos_coords, xh_excite_pos_coords[:, :, [0, 3, 4, 1, 2]]))
xh_excite_pos_weights = np.reshape(xh_excite_pos_weights, (5, 27*walkers))
xh_excite_pos_weights = np.hstack((xh_excite_pos_weights, xh_excite_pos_weights))
xh_excite_pos_dips = np.zeros((5, 27*walkers*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, xh_excite_pos_coords[i], mass, planar=True)
    xh_excite_pos_coords[i] = eck.get_rotated_coords()
    xh_excite_pos_dips[i] = dip(xh_excite_pos_coords[i])
    # xh_combine_weights = np.hstack((xh_excite_pos_weights[i], xh_excite_neg_weights[i]))

binzz = 100
# amp_f0_ap_dip = np.zeros((10, 3, binzz))
# amp_f0_zp_dip = np.zeros((10, 3, binzz))
# amp_f0_ap_axyz = np.zeros((10, 3, binzz))
term1 = np.zeros((10, 3))
term1_vec = np.zeros((10, 3))
term1_ov = np.zeros(10)
term1_dis = np.zeros(10)
term1_dis_a_w_xh = np.zeros(10)
term1_dis_xh_w_a = np.zeros(10)
xh_term1 = np.zeros((10, 3))
xh_term1_vec = np.zeros((10, 3))
xh_term1_ov = np.zeros(10)
xh_term1_dis = np.zeros(10)
xyz = ['x', 'y', 'z']
for i in range(10):
    frac = get_da_psi(ground_coords[i], 'a')/get_da_psi(ground_coords[i], None)
    frac2 = get_da_psi(ground_coords[i], 'sp')/get_da_psi(ground_coords[i], None)
    term1_ov[i] = np.dot(ground_weights[i], frac)/np.sum(ground_weights[i])
    xh_term1_ov[i] = np.dot(ground_weights[i], frac2)/np.sum(ground_weights[i])
    dists = all_dists(ground_coords[i])
    term1_dis[i] = np.dot(ground_weights[i], frac*dists[:, 0])/np.sum(ground_weights[i])
    xh_term1_dis[i] = np.dot(ground_weights[i], frac2*dists[:, -1])/np.sum(ground_weights[i])
    term1_dis_a_w_xh[i] = np.dot(ground_weights[i], frac2*dists[:, 0])/np.sum(ground_weights[i])
    term1_dis_xh_w_a[i] = np.dot(ground_weights[i], frac*dists[:, -1])/np.sum(ground_weights[i])
    for j in range(3):
        plt.hist2d(ground_dips[i, :, j]*au_to_Debye, a_prime(dists[:, 0], dists[:, -1]), bins=binzz, weights=ground_weights[i])
        if j == 0:
            plt.xlabel(r'$\rm{\mu_x}$ Debye')
        elif j == 1:
            plt.xlabel(r'$\rm{\mu_y}$ Debye')
        else:
            plt.xlabel(r'$\rm{\mu_z}$ Debye')
        plt.ylabel(r"a'")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f0_ap_dipole_{xyz[j]}_simulation_{i+1}')
        plt.close()
        plt.hist2d(ground_dips[i, :, j] * au_to_Debye, z_prime(dists[:, 0], dists[:, -1]), bins=binzz,
                   weights=ground_weights[i])
        if j == 0:
            plt.xlabel(r'$\rm{\mu_x}$ Debye')
        elif j == 1:
            plt.xlabel(r'$\rm{\mu_y}$ Debye')
        else:
            plt.xlabel(r'$\rm{\mu_z}$ Debye')
        plt.ylabel(r"z'")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f0_zp_dipole_{xyz[j]}_simulation_{i + 1}')
        plt.close()
        term1[i, j] = np.dot(ground_weights[i], frac*ground_dips[i, :, j]*au_to_Debye)/np.sum(ground_weights[i])
        term1_vec[i, j] = np.dot(ground_weights[i], frac * ((ground_coords[i, :, 2, j] - ground_coords[i, :, 1, j]) +
                                                             (ground_coords[i, :, 4, j] - ground_coords[i, :, 3, j]))) \
                          / np.sum(ground_weights[i])
        xh_term1[i, j] = np.dot(ground_weights[i], frac2 * ground_dips[i, :, j] * au_to_Debye) / np.sum(ground_weights[i])
        mid = (ground_coords[i, :, 3, j] - ground_coords[i, :, 1, j]) / 2
        xh_term1_vec[i, j] = np.dot(ground_weights[i], frac2 * (mid - ground_coords[i, :, 0, j])) \
                             / np.sum(ground_weights[i])
    plt.hist2d(((ground_coords[i, :, 2, 0] - ground_coords[i, :, 1, 0]) +
                (ground_coords[i, :, 4, 0] - ground_coords[i, :, 3, 0])),
               ((ground_coords[i, :, 2, 1] - ground_coords[i, :, 1, 1]) +
                (ground_coords[i, :, 4, 1] - ground_coords[i, :, 3, 1])),
               bins=binzz, weights=ground_weights[i])

    plt.xlabel(r"$a_x$ Bohr")
    plt.ylabel(r"$a_y$ Bohr")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'f0_ap_a_comp_xy_simulation_{i + 1}')
    plt.close()
    plt.hist2d(z_prime(dists[:, 0], dists[:, -1]), dists[:, -2],
               bins=binzz, weights=ground_weights[i])

    plt.xlabel(r"z' ")
    plt.ylabel(r"$R_{OO}$ Bohr")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'f0_zp_vs_Roo_simulation_{i + 1}')
    plt.close()
    amp, xx = np.histogram(((ground_coords[i, :, 2, 0] - ground_coords[i, :, 1, 0]) +
                (ground_coords[i, :, 4, 0] - ground_coords[i, :, 3, 0])), weights=ground_weights[i],
                           bins=binzz, density=True)
    xx = (xx[1:] + xx[:-1])/2
    plt.plot(xx, amp)
    plt.xlabel(r"$a_z$ Bohr")
    plt.tight_layout()
    plt.savefig(f'f0_ap_a_comp_z_simulation_{i+1}')
    plt.close()

avg_term1_vec = np.average(term1_vec, axis=0)
std_term1_vec = np.std(term1_vec, axis=0)
avg_term1_o = np.average(term1_ov)
std_term1_o = np.std(term1_ov)
avg_term1_d = np.average(term1_dis)
std_term1_d = np.std(term1_dis)
avg_term1_d_a_w_xh = np.average(term1_dis_a_w_xh)
std_term1_d_a_w_xh = np.std(term1_dis_a_w_xh)
avg_term1_d_xh_w_a = np.average(term1_dis_xh_w_a)
std_term1_d_xh_w_a = np.std(term1_dis_xh_w_a)
avg_xh_term1_v = np.average(xh_term1_vec, axis=0)
std_xh_term1_v = np.std(xh_term1_vec, axis=0)
avg_xh_term1_o = np.average(xh_term1_ov)
std_xh_term1_o = np.std(xh_term1_ov)
avg_xh_term1_d = np.average(xh_term1_dis)
std_xh_term1_d = np.std(xh_term1_dis)
print(np.average(term1, axis=0))
print(np.std(term1, axis=0))
print(np.average(xh_term1, axis=0))
print(np.std(xh_term1, axis=0))
xh_term1 = np.linalg.norm(xh_term1, axis=-1)
term1 = np.linalg.norm(term1, axis=-1)
std_xh_term1 = np.std(xh_term1)
std_term1 = np.std(term1)
term1 = np.average(term1)
xh_term1 = np.average(xh_term1)
print(term1)
print(std_term1)
print(xh_term1)
print(std_xh_term1)

freq = average_excite_energy-average_zpe
freq2 = average_xh_excite_energy - average_zpe
std_freq2 = np.sqrt(std_zpe**2 + std_xh_excite_energy**2)
std_freq = np.sqrt(std_zpe**2 + std_excite_energy**2)
std_term1_sq = term1**2*np.sqrt((std_term1/term1)**2 + (std_term1/term1)**2)
std_term1_sq_freq = term1**2*freq*np.sqrt((std_term1_sq/term1**2)**2 + (std_freq/freq)**2)

std_term1_xh_sq = xh_term1**2*np.sqrt((std_xh_term1/xh_term1)**2 + (std_xh_term1/xh_term1)**2)
std_term1_xh_freq = xh_term1**2*freq2*np.sqrt((std_term1_xh_sq/xh_term1**2)**2 + (std_freq2/freq2)**2)
conversion = conv_fac*km_mol
print(conversion*term1**2*freq)
print(std_term1_sq_freq*conversion)

throw_it_out = [0, 2, 3, 4]
throw_it_out2 = [0, 1, 3, 4]
term2 = np.zeros((5, 3))
amp_excite = np.zeros((5, binzz))
amp_frac2 = np.zeros((5, binzz))
amp_f1 = np.zeros((5, binzz))
term2_vec = np.zeros((5, 3))
term2_ov = np.zeros(5)
xh_term2 = np.zeros((5, 3))
xh_term2_vec = np.zeros((5, 3))
xh_term2_ov = np.zeros(5)
term2_dis = np.zeros(5)
term2_dis_a_w_xh = np.zeros(5)
term2_dis_xh_w_a = np.zeros(5)
xh_term2_dis = np.zeros(5)
combine_dips = np.zeros((5, 27*walkers*4, 3))
combine_weights = np.zeros((5, 27*walkers*4))
xh_combine_dips = np.zeros((5, 27*walkers*4, 3))
xh_combine_weights = np.zeros((5, 27*walkers*4))
for i in range(5):
    combine_coords = np.vstack((excite_neg_coords[i], excite_pos_coords[i]))
    xh_combine_coords = np.vstack((xh_excite_neg_coords[i], xh_excite_pos_coords[i]))
    combine_weights[i] = np.hstack((excite_neg_weights[i], excite_pos_weights[i]))
    combine_dips[i] = np.vstack((excite_neg_dips[i], excite_pos_dips[i]))
    xh_combine_weights[i] = np.hstack((xh_excite_neg_weights[i], xh_excite_pos_weights[i]))
    xh_combine_dips[i] = np.vstack((xh_excite_neg_dips[i], xh_excite_pos_dips[i]))
    H0 = get_da_psi(combine_coords, None)
    H0x = get_da_psi(xh_combine_coords, None)
    H1 = get_da_psi(combine_coords, 'a')
    H2 = get_da_psi(xh_combine_coords, 'sp')
    frac = H0/H1
    frac2 = H0x/H2
    term2_ov[i] = np.dot(combine_weights[i], frac)/np.sum(combine_weights[i])
    xh_term2_ov[i] = np.dot(xh_combine_weights[i], frac2)/np.sum(xh_combine_weights[i])
    dists = all_dists(combine_coords)
    dists_special = dists
    term2_dis[i] = np.average(frac*dists[:, 0], weights=combine_weights[i])
    term2_dis_xh_w_a[i] = np.average(frac*dists[:, -1], weights=combine_weights[i])
    dists = all_dists(xh_combine_coords)
    term2_dis_a_w_xh[i] = np.average(frac2*dists[:, 0], weights=combine_weights[i])
    xh_term2_dis[i] = np.dot(xh_combine_weights[i], frac2*dists[:, -1])/np.sum(xh_combine_weights[i])
    for j in range(3):
        plt.hist2d(combine_dips[i, :, j] * au_to_Debye, a_prime(dists_special[:, 0], dists_special[:, -1]),
                   bins=binzz,
                   weights=combine_weights[i])
        if j == 0:
            plt.xlabel(r'$\rm{\mu_x}$ Debye')
        elif j == 1:
            plt.xlabel(r'$\rm{\mu_y}$ Debye')
        else:
            plt.xlabel(r'$\rm{\mu_z}$ Debye')
        plt.ylabel(r"a'")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f1_ap_dipole_{xyz[j]}_simulation_{i + 1}')
        plt.close()
        plt.hist2d(combine_dips[i, :, j] * au_to_Debye, z_prime(dists_special[:, 0], dists_special[:, -1]),
                   bins=binzz,
                   weights=combine_weights[i])
        if j == 0:
            plt.xlabel(r'$\rm{\mu_x}$ Debye')
        elif j == 1:
            plt.xlabel(r'$\rm{\mu_y}$ Debye')
        else:
            plt.xlabel(r'$\rm{\mu_z}$ Debye')
        plt.ylabel(r"z'")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f1_zp_dipole_{xyz[j]}_simulation_{i + 1}')
        plt.close()
        term2[i, j] = np.dot(combine_weights[i], frac * combine_dips[i, :, j] * au_to_Debye) \
                      / np.sum(combine_weights[i])
        term2_vec[i, j] = np.dot(combine_weights[i], frac * ((combine_coords[:, 2, j] - combine_coords[:, 1, j]) +
                                                             (combine_coords[:, 4, j] - combine_coords[:, 3, j]))) \
                          / np.sum(combine_weights[i])
        xh_term2[i, j] = np.dot(xh_combine_weights[i], frac2*xh_combine_dips[i, :, j] * au_to_Debye) \
                         /np.sum(xh_combine_weights[i])
        mid = (xh_combine_coords[:, 3, j] - xh_combine_coords[:, 1, j])/2
        xh_term2_vec[i, j] = np.dot(xh_combine_weights[i], frac2 * (mid - xh_combine_coords[:, 0, j])) \
                             / np.sum(xh_combine_weights[i])
    plt.hist2d(((combine_coords[:, 2, 0] - combine_coords[:, 1, 0]) +
                                                     (combine_coords[:, 4, 0] - combine_coords[:, 3, 0])),
               ((combine_coords[:, 2, 1] - combine_coords[:, 1, 1]) +
                                                     (combine_coords[:, 4, 1] - combine_coords[:, 3, 1])),
               bins=binzz, weights=combine_weights[i])

    plt.xlabel(r"$a_x$ Bohr")
    plt.ylabel(r"$a_y$ Bohr")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'f1_ap_a_comp_xy_simulation_{i + 1}')
    plt.close()
    amp, xx = np.histogram(((combine_coords[:, 2, 0] - combine_coords[:, 1, 0]) +
                            (combine_coords[:, 4, 0] - combine_coords[:, 3, 0])), weights=combine_weights[i],
                           bins=binzz, density=True)
    xx = (xx[1:] + xx[:-1]) / 2
    plt.plot(xx, amp)
    plt.xlabel(r"$a_z$ Bohr")
    plt.tight_layout()
    plt.savefig(f'f1_ap_a_comp_z_simulation_{i + 1}')
    plt.close()
    plt.hist2d(z_prime(dists_special[:, 0], dists_special[:, -1]), dists_special[:, -2],
               bins=binzz, weights=combine_weights[i])

    plt.xlabel(r"z' ")
    plt.ylabel(r"$R_{OO}$ Bohr")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'f1_zp_vs_Roo_in_ap_simulation_{i + 1}')
    plt.close()
    plt.hist2d(z_prime(dists[:, 0], dists[:, -1]), dists[:, -2],
               bins=binzz, weights=xh_combine_weights[i])

    plt.xlabel(r"z' ")
    plt.ylabel(r"$R_{OO}$ Bohr")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'f1_zp_vs_Roo_in_zp_simulation_{i + 1}')
    plt.close()

term2_vec = term2_vec[throw_it_out]
term2_ov = term2_ov[throw_it_out]
term2_dis = term2_dis[throw_it_out]
term2 = term2[throw_it_out]
xh_term2_vec = xh_term2_vec[throw_it_out2]
xh_term2_ov = xh_term2_ov[throw_it_out2]
xh_term2_dis = xh_term2_dis[throw_it_out2]
xh_term2 = xh_term2[throw_it_out2]
avg_term2_vec = np.average(term2_vec, axis=0)
std_term2_vec = np.std(term2_vec, axis=0)
avg_term2_o = np.average(term2_ov)
std_term2_o = np.std(term2_ov)
avg_term2_d = np.average(term2_dis)
std_term2_d = np.std(term2_dis)
avg_term2_d_a_w_xh = np.average(term2_dis_a_w_xh)
std_term2_d_a_w_xh = np.std(term2_dis_a_w_xh)
avg_term2_d_xh_w_a = np.average(term2_dis_xh_w_a)
std_term2_d_xh_w_a = np.std(term2_dis_xh_w_a)
avg_xh_term2_v = np.average(xh_term2_vec, axis=0)
std_xh_term2_v = np.std(xh_term2_vec, axis=0)
avg_xh_term2_o = np.average(xh_term2_ov)
std_xh_term2_o = np.std(xh_term2_ov)
avg_xh_term2_d = np.average(xh_term2_dis)
std_xh_term2_d = np.std(xh_term2_dis)
print(np.average(term2, axis=0))
print(np.std(term2, axis=0))
print(np.average(xh_term2, axis=0))
print(np.std(xh_term2, axis=0))

# x = (xx[1:] + xx[:-1]) / 2.
#
#
#
# def asym_grid(coords, a):
#     re = np.linalg.norm(coords[2]-coords[1])
#     coords = np.array([coords]*1)
#     coords = coords[:, (1, 3, 0, 2, 4)]
#     zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
#                                                                         ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
#                                                                                    [2, 0, 1, 0], [3, 0, 1, 2],
#                                                                                    [4, 1, 0, 2]])).coords
#     N = len(a)
#     zmat = np.array([zmat]*N).reshape((N, 4, 6))
#     zmat[:, 2, 1] = re + np.sqrt(2)/2*a
#     zmat[:, 3, 1] = re - np.sqrt(2)/2*a
#     new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     coords = new_coords[:, (2, 0, 3, 1, 4)]
#     coords[:, :, 1] = coords[:, :, 2]
#     coords[:, :, 2] = np.zeros(coords[:, :, 2].shape)
#     return coords
#
#
# linear_struct = np.array([
#     [0.000000000000000, 0.000000000000000, 0.000000000000000],
#     [-2.304566686034061, 0.000000000000001, 0.000000000000000],
#     [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
#     [2.304566686034061, 0.000000000000001, 0.000000000000000],
#     [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233]
# ])
# linear_struct[:, 0] = linear_struct[:, 0] + 2.304566686034061
#
#
# truth_coords = asym_grid(linear_struct, x)
#
# truth = get_da_psi(truth_coords, 'a')*get_da_psi(truth_coords, None)
# frac1 = get_da_psi(truth_coords, 'a')/get_da_psi(truth_coords, None)
# frac2 = get_da_psi(truth_coords, None)/get_da_psi(truth_coords, 'a')
#
# avg_ground = np.mean(amp_ground, axis=0)
# avg_excite = np.mean(amp_excite, axis=0)
# avg_frac1 = np.mean(amp_frac1, axis=0)
# avg_frac2 = np.mean(amp_frac2, axis=0)
# avg_f0 = np.mean(amp_f0, axis=0)
# avg_f1 = np.mean(amp_f1, axis=0)
# truth = truth/np.max(truth)*np.max(amp_f1)
# truth1 = frac1/np.max(frac1)*np.max(amp_f1)
# truth2 = frac2/np.max(frac2)*np.

term2 = np.linalg.norm(term2, axis=-1)
xh_term2 = np.linalg.norm(xh_term2, axis=-1)

std_term2 = np.std(term2)
xh_std_term2 = np.std(xh_term2)
term2 = np.average(term2)
xh_term2 = np.average(xh_term2)
print(term2)
print(std_term2)
print(xh_term2)
print(xh_std_term2)


std_term2_xh_sq = xh_term2**2*np.sqrt((xh_std_term2/xh_term2)**2 + (xh_std_term2/xh_term2)**2)
std_term2_sq = term2**2*np.sqrt((std_term2/term2)**2 + (std_term2/term2)**2)
std_term2_sq_freq = term2**2*freq*np.sqrt((std_term2_sq/term2**2)**2 + (std_freq/freq)**2)
xh_std_term2_sq_freq = xh_term2**2*freq2*np.sqrt((std_term2_xh_sq/xh_term2**2)**2 + (std_freq2/freq2)**2)
print(conversion*term2**2*freq)
print(std_term2_sq_freq*conversion)

# import matplotlib.pyplot as plt
# plt.plot(x/ang2bohr, avg_f0*frac1, label=r'$\rm{f_0\Psi_1/\Psi_0}$')
# plt.plot(x/ang2bohr, avg_f1*frac2, label=r'$\rm{f_1\Psi_0/\Psi_1}$')
# plt.plot(x/ang2bohr, truth, label=r'$\rm{\Psi_1\Psi_0}$')
# # plt.plot(x/ang2bohr, avg_f0, label=r'$\rm{f_0}$')
# # plt.plot(x/ang2bohr, avg_f1, label=r'$\rm{f_1}$')
# # plt.plot(x/ang2bohr, avg_frac1, label=r'$\rm{\Psi_1/\Psi_0}$')
# # plt.plot(x/ang2bohr, avg_frac2, label=r'$\rm{\Psi_0/\Psi_1}$')
# plt.xlabel(r'a $\rm{\AA}$')
# # plt.ylim(0, 3.2)
# plt.legend()
# plt.tight_layout()
# plt.savefig('Checking_asymmetric_stretch')
# plt.show()
#
# plt.plot(x/ang2bohr, avg_f0, label=r'$\rm{f_0}$')
# plt.plot(x/ang2bohr, avg_f1, label=r'$\rm{f_1}$')
# plt.xlabel(r'a $\rm{\AA}$')
# plt.legend()
# plt.tight_layout()
# plt.savefig('Checking_asymmetric_stretch_f0_f1')
# plt.show()
#
# plt.plot(x/ang2bohr, avg_frac1, label=r'$\rm{\Psi_1/\Psi_0}$')
# plt.plot(x/ang2bohr, avg_frac2, label=r'$\rm{\Psi_0/\Psi_1}$')
#
# plt.xlabel(r'a $\rm{\AA}$')
# plt.legend()
# plt.tight_layout()
# plt.savefig('Checking_asymmetric_stretch_fracs')
# plt.show()

# plt.plot(x/ang2bohr, frac1, label=r'$\')

# term3 = 0.14379963852224506
term3 = 0.0424328790886425
term3_mod = 0.057139489481379306
xh_term3 = 1.1154257246577732
xh_term3 = 1.0723059556104413
std_term3 = 0.0
std_term3_sq = 0.0
std_term3_sq_freq = term3**2*freq*np.sqrt((std_term3_sq/term3**2)**2 + (std_freq/freq)**2)
xh_std_term3_sq_freq = xh_term3**2*freq2*np.sqrt((std_freq2/freq2)**2)
std_term3_mod_sq_freq = term3**2*freq*np.sqrt((std_freq/freq)**2)
print(term3**2*freq*conversion)
print(std_term3_sq_freq*conversion)
print(term3_mod**2*freq*conversion)
print(std_term3_mod_sq_freq*conversion)

full_error = np.sqrt(std_term1**2 + std_term2**2 + std_term3**2)
full_error_xh = np.sqrt(std_xh_term1**2 + xh_std_term2**2)
dipole_xh = xh_term1 + xh_term2 - xh_term3
dipole = term1 + term2 - term3_mod
full_error_sq = dipole**2*np.sqrt((full_error/dipole)**2 + (full_error/dipole)**2)
full_error_sq_xh = dipole_xh**2*np.sqrt((full_error_xh/dipole_xh)**2 + (full_error_xh/dipole_xh)**2)
full_error = dipole**2*freq*np.sqrt((full_error_sq/dipole**2)**2 + (std_freq/freq)**2)
full_error_xh = dipole_xh**2*freq2*np.sqrt((full_error_sq_xh/dipole_xh**2)**2 + (std_freq2/freq2)**2)
print(dipole**2*freq*conversion)
print(full_error*conversion)


print(conversion*xh_term1**2*freq2)
print(std_term1_xh_freq*conversion)
print(conversion*xh_term2**2*freq2)
print(xh_std_term2_sq_freq*conversion)
print(xh_term3**2*freq2*conversion)
print(xh_std_term3_sq_freq*conversion)
print(dipole_xh**2*freq2*conversion)
print(full_error_xh*conversion)


print(f'term1 a overlap = {avg_term1_o} {std_term1_o}')
print(f'term2 a overlap = {avg_term2_o} {std_term2_o}')

print(f'term1 xh overlap = {avg_xh_term1_o} {std_xh_term1_o}')
print(f'term2 xh overlap = {avg_xh_term2_o} {std_xh_term2_o}')

print(f'term1 a dis = {avg_term1_d} {std_term1_d}')
print(f'term2 a dis = {avg_term2_d} {std_term2_d}')

print(f'term1 xh dis = {avg_xh_term1_d} {std_xh_term1_d}')
print(f'term2 xh dis = {avg_xh_term2_d} {std_xh_term2_d}')

print(f'term1 a vec = {avg_term1_vec} {std_term1_vec}')
print(f'term2 a vec = {avg_term2_vec} {std_term2_vec}')

print(f'term1 xh vec = {avg_xh_term1_v} {std_xh_term1_v}')
print(f'term2 xh vec = {avg_xh_term2_v} {std_xh_term2_v}')

print(f'term1 a dis with xh excite = {avg_term1_d_a_w_xh} {std_term1_d_a_w_xh}')
print(f'term2 a dis with xh excite = {avg_term2_d_a_w_xh} {std_term2_d_a_w_xh}')

print(f'term1 xh dis with a excite = {avg_term1_d_xh_w_a} {std_term1_d_xh_w_a}')
print(f'term2 xh dis with a excite = {avg_term2_d_xh_w_a} {std_term2_d_xh_w_a}')





