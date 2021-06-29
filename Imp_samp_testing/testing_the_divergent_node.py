import numpy as np
import matplotlib.pyplot as plt
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
m_OH = (m_H*m_O)/(m_H+m_O)
omega_asym = 3070.648654929466/har2wave


big_Roo_grid = np.linspace(4, 5.4, 1000)
big_sp_grid = np.linspace(-1.2, 1.2, 1000)
X, Y = np.meshgrid(big_sp_grid, big_Roo_grid)


z_ground_no_der = np.load('z_ground_no_der.npy')

ground_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
                                                       z_ground_no_der.flatten())

z_excite_xh_no_der = np.load('z_excite_xh_no_der.npy')

excite_xh_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
                                                          z_excite_xh_no_der.flatten())


def get_da_psi(coords, excite):
    psi = np.zeros((len(coords), 2))
    dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    if excite == 'sp':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = excite_xh_no_der(dists[:, -1], dists[:, -2])
    elif excite == 'a':
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    else:
        psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    return np.prod(psi, axis=-1)


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    s_oh = 1/np.sqrt(2)*(dis[:, 0]+dis[:, 1])
    mid = dis[:, 2]/2
    sp = mid - dis[:, -1]*np.cos(roh_roo_angle(coords, dis[:, -2], dis[:, -1]))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], s_oh, dis[:, -2], sp)).T


def roh_roo_angle(coords, roo_dist, roh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (coords[:, 1]-coords[:, 0])/np.broadcast_to(roh_dist[:, None], (len(roh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


ground_coords = np.zeros((10, 27, 5000, 5, 3))
ground_erefs = np.zeros((10, 20000))
ground_weights = np.zeros((10, 27, 5000))
for i in range(10):
    blah = np.load(f'ground_state_full_h3o2_{i+1}.npz')
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

xh_excite_neg_coords = np.zeros((5, 27, 5000, 5, 3))
xh_excite_neg_erefs = np.zeros((5, 20000))
xh_excite_neg_weights = np.zeros((5, 27, 5000))
for i in range(5):
    blah = np.load(f'XH_excite_state_full_h3o2_left_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    xh_excite_neg_coords[i] = coords
    xh_excite_neg_erefs[i] = eref
    xh_excite_neg_weights[i] = weights

print(np.mean(np.mean(xh_excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_xh_excite_neg_energy = np.mean(np.mean(xh_excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_xh_excite_neg_energy = np.std(np.mean(xh_excite_neg_erefs[:, 5000:]*har2wave, axis=1))

xh_excite_pos_coords = np.zeros((5, 27, 5000, 5, 3))
xh_excite_pos_erefs = np.zeros((5, 20000))
xh_excite_pos_weights = np.zeros((5, 27, 5000))
for i in range(5):
    blah = np.load(f'XH_excite_state_full_h3o2_right_{i+1}.npz')
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

average_xh_excite_energy = np.average(np.array([average_xh_excite_pos_energy, average_xh_excite_neg_energy]))
std_xh_excite_energy = np.sqrt(std_xh_excite_pos_energy**2 + std_xh_excite_neg_energy**2)
print(average_xh_excite_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_xh_excite_energy**2))


ground_coords = np.reshape(ground_coords, (10, 135000, 5, 3))
ground_coords = np.hstack((ground_coords, ground_coords[:, :, [0, 3, 4, 1, 2]]))
ground_weights = np.reshape(ground_weights, (10, 135000))
ground_weights = np.hstack((ground_weights, ground_weights))

xh_excite_neg_coords = np.reshape(xh_excite_neg_coords, (5, 135000, 5, 3))
xh_excite_neg_coords = np.hstack((xh_excite_neg_coords, xh_excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
xh_excite_neg_weights = np.reshape(xh_excite_neg_weights, (5, 135000))
xh_excite_neg_weights = np.hstack((xh_excite_neg_weights, xh_excite_neg_weights))

xh_excite_pos_coords = np.reshape(xh_excite_pos_coords, (5, 135000, 5, 3))
xh_excite_pos_coords = np.hstack((xh_excite_pos_coords, xh_excite_pos_coords[:, :, [0, 3, 4, 1, 2]]))
xh_excite_pos_weights = np.reshape(xh_excite_pos_weights, (5, 135000))
xh_excite_pos_weights = np.hstack((xh_excite_pos_weights, xh_excite_pos_weights))

xh_term1_dis = np.zeros(10)
for i in range(10):
    frac2 = get_da_psi(ground_coords[i], 'sp')/get_da_psi(ground_coords[i], None)
    dists = all_dists(ground_coords[i])
    xh_term1_dis[i] = np.dot(ground_weights[i], frac2*dists[:, -1])/np.sum(ground_weights[i])

avg_xh_term1_d = np.average(xh_term1_dis)
std_xh_term1_d = np.std(xh_term1_dis)

xh_term2_dis = np.zeros(5)
xh_combine_weights = np.zeros((5, 135000*4))
for i in range(5):
    xh_combine_coords = np.vstack((xh_excite_neg_coords[i], xh_excite_pos_coords[i]))
    xh_combine_weights[i] = np.hstack((xh_excite_neg_weights[i], xh_excite_pos_weights[i]))
    H0 = get_da_psi(xh_combine_coords, None)
    H1 = get_da_psi(xh_combine_coords, 'sp')
    frac2 = H0/H1
    dists = all_dists(xh_combine_coords)
    xh_term2_dis[i] = np.dot(xh_combine_weights[i], frac2*dists[:, -1])/np.sum(xh_combine_weights[i])

avg_xh_term2_d = np.average(xh_term2_dis)
std_xh_term2_d = np.std(xh_term2_dis)

print(f'term1 xh dis = {avg_xh_term1_d} {std_xh_term1_d}')
print(f'term2 xh dis = {avg_xh_term2_d} {std_xh_term2_d}')




