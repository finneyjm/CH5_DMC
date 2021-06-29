import numpy as np
import matplotlib.pyplot as plt
from ProtWaterPES import Dipole
import multiprocessing as mp
from Imp_samp_testing import EckartsSpinz
from Imp_samp_testing import MomentOfSpinz
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
        [ 2.55704662,  0.05115356, -0.2381117 ],
        [ 0.24088235, -0.09677082,  0.09615192],
        [-0.09502706, -1.86894299, -0.69579001],
        [ 5.02836896, -0.06798562, -0.30434529],
        [ 5.24391277,  0.14767547,  1.4669121 ],
])


dis1 = all_dists(np.array([test_structure]*1))
dis2 = all_dists(np.array([test_structure2]*1))

print(dis1[:, [0, -1]])
print(dis2[:, [0, -1]])

pool = mp.Pool(mp.cpu_count()-1)


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

excite_neg_coords = np.zeros((5, 27, 5000, 5, 3))
excite_neg_erefs = np.zeros((5, 20000))
excite_neg_weights = np.zeros((5, 27, 5000))
for i in range(5):
    blah = np.load(f'Asym_excite_state_full_h3o2_left_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    excite_neg_coords[i] = coords
    excite_neg_erefs[i] = eref
    excite_neg_weights[i] = weights

print(np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_excite_neg_energy = np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_excite_neg_energy = np.std(np.mean(excite_neg_erefs[:, 5000:]*har2wave, axis=1))

excite_pos_coords = np.zeros((5, 27, 5000, 5, 3))
excite_pos_erefs = np.zeros((5, 20000))
excite_pos_weights = np.zeros((5, 27, 5000))
for i in range(5):
    blah = np.load(f'Asym_excite_state_full_h3o2_right_{i+1}.npz')
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
au_to_Debye = 1/0.3934303
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_OH = (m_H*m_O)/(m_H+m_O)
omega_asym = 3070.648654929466/har2wave
mw = m_OH*omega_asym
conv_fac = 4.702e-7
km_mol = 5.33e6


ground_coords = np.reshape(ground_coords, (10, 135000, 5, 3))
ground_coords = np.hstack((ground_coords, ground_coords[:, :, [0, 3, 4, 1, 2]]))
ground_weights = np.reshape(ground_weights, (10, 135000))
ground_weights = np.hstack((ground_weights, ground_weights))
ground_dips = np.zeros((10, 135000*2, 3))
for i in range(10):
    eck = EckartsSpinz(ref, ground_coords[i], mass, planar=True)
    ground_coords[i] = eck.get_rotated_coords()
    ground_dips[i] = dip(ground_coords[i])

excite_neg_coords = np.reshape(excite_neg_coords, (5, 135000, 5, 3))
excite_neg_coords = np.hstack((excite_neg_coords, excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
excite_neg_weights = np.reshape(excite_neg_weights, (5, 135000))
excite_neg_weights = np.hstack((excite_neg_weights, excite_neg_weights))
excite_neg_dips = np.zeros((5, 135000*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, excite_neg_coords[i], mass, planar=True)
    excite_neg_coords[i] = eck.get_rotated_coords()
    excite_neg_dips[i] = dip(excite_neg_coords[i])

excite_pos_coords = np.reshape(excite_pos_coords, (5, 135000, 5, 3))
excite_pos_coords = np.hstack((excite_pos_coords, excite_pos_coords[:, :, [0, 3, 4, 1, 2]]))
excite_pos_weights = np.reshape(excite_pos_weights, (5, 135000))
excite_pos_weights = np.hstack((excite_pos_weights, excite_pos_weights))
excite_pos_dips = np.zeros((5, 135000*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, excite_pos_coords[i], mass, planar=True)
    excite_pos_coords[i] = eck.get_rotated_coords()
    excite_pos_dips[i] = dip(excite_pos_coords[i])
    # combine_weights = np.hstack((excite_pos_weights[i], excite_neg_weights[i]))


xh_excite_neg_coords = np.reshape(xh_excite_neg_coords, (5, 135000, 5, 3))
xh_excite_neg_coords = np.hstack((xh_excite_neg_coords, xh_excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
xh_excite_neg_weights = np.reshape(xh_excite_neg_weights, (5, 135000))
xh_excite_neg_weights = np.hstack((xh_excite_neg_weights, xh_excite_neg_weights))
xh_excite_neg_dips = np.zeros((5, 135000*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, xh_excite_neg_coords[i], mass, planar=True)
    xh_excite_neg_coords[i] = eck.get_rotated_coords()
    xh_excite_neg_dips[i] = dip(xh_excite_neg_coords[i])

xh_excite_pos_coords = np.reshape(xh_excite_pos_coords, (5, 135000, 5, 3))
xh_excite_pos_coords = np.hstack((xh_excite_pos_coords, xh_excite_pos_coords[:, :, [0, 3, 4, 1, 2]]))
xh_excite_pos_weights = np.reshape(xh_excite_pos_weights, (5, 135000))
xh_excite_pos_weights = np.hstack((xh_excite_pos_weights, xh_excite_pos_weights))
xh_excite_pos_dips = np.zeros((5, 135000*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, xh_excite_pos_coords[i], mass, planar=True)
    xh_excite_pos_coords[i] = eck.get_rotated_coords()
    xh_excite_pos_dips[i] = dip(xh_excite_pos_coords[i])
    # xh_combine_weights = np.hstack((xh_excite_pos_weights[i], xh_excite_neg_weights[i]))

term1 = np.zeros((10, 3))
term1_vec = np.zeros((10, 3))
term1_ov = np.zeros(10)
term1_dis = np.zeros(10)
xh_term1 = np.zeros((10, 3))
xh_term1_vec = np.zeros((10, 3))
xh_term1_ov = np.zeros(10)
xh_term1_dis = np.zeros(10)
for i in range(10):
    frac = get_da_psi(ground_coords[i], 'a')/get_da_psi(ground_coords[i], None)
    frac2 = get_da_psi(ground_coords[i], 'sp')/get_da_psi(ground_coords[i], None)
    term1_ov[i] = np.dot(ground_weights[i], frac)/np.sum(ground_weights[i])
    xh_term1_ov[i] = np.dot(ground_weights[i], frac2)/np.sum(ground_weights[i])
    dists = all_dists(ground_coords[i])
    term1_dis[i] = np.dot(ground_weights[i], frac*dists[:, 0])/np.sum(ground_weights[i])
    xh_term1_dis[i] = np.dot(ground_weights[i], frac2*dists[:, -1])/np.sum(ground_weights[i])
    for j in range(3):
        term1[i, j] = np.dot(ground_weights[i], frac*ground_dips[i, :, j]*au_to_Debye)/np.sum(ground_weights[i])
        term1_vec[i, j] = np.dot(ground_weights[i], frac * ((ground_coords[i, :, 2, j] - ground_coords[i, :, 1, j]) +
                                                             (ground_coords[i, :, 4, j] - ground_coords[i, :, 3, j]))) \
                          / np.sum(ground_weights[i])
        xh_term1[i, j] = np.dot(ground_weights[i], frac2 * ground_dips[i, :, j] * au_to_Debye) / np.sum(ground_weights[i])
        mid = (ground_coords[i, :, 3, j] - ground_coords[i, :, 1, j]) / 2
        xh_term1_vec[i, j] = np.dot(ground_weights[i], frac2 * (mid - ground_coords[i, :, 0, j])) \
                             / np.sum(ground_weights[i])

avg_term1_vec = np.average(term1_vec, axis=0)
std_term1_vec = np.std(term1_vec, axis=0)
avg_term1_o = np.average(term1_ov)
std_term1_o = np.std(term1_ov)
avg_term1_d = np.average(term1_dis)
std_term1_d = np.std(term1_dis)
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


term2 = np.zeros((5, 3))
term2_vec = np.zeros((5, 3))
term2_ov = np.zeros(5)
xh_term2 = np.zeros((5, 3))
xh_term2_vec = np.zeros((5, 3))
xh_term2_ov = np.zeros(5)
term2_dis = np.zeros(5)
xh_term2_dis = np.zeros(5)
combine_dips = np.zeros((5, 135000*4, 3))
combine_weights = np.zeros((5, 135000*4))
xh_combine_dips = np.zeros((5, 135000*4, 3))
xh_combine_weights = np.zeros((5, 135000*4))
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
    term2_dis[i] = np.dot(combine_weights[i], frac*dists[:, 0])/np.sum(combine_weights[i])
    dists = all_dists(xh_combine_coords)
    xh_term2_dis[i] = np.dot(xh_combine_weights[i], frac2*dists[:, -1])/np.sum(xh_combine_weights[i])
    for j in range(3):
        term2[i, j] = np.dot(combine_weights[i], frac * combine_dips[i, :, j] * au_to_Debye) \
                      / np.sum(combine_weights[i])
        term2_vec[i, j] = np.dot(combine_weights[i], frac * ((combine_coords[:, 2, j] - combine_coords[:, 1, j]) +
                                                         (combine_coords[:, 4, j] - combine_coords[:, 3, j]))) \
                      / np.sum(combine_weights[i])
        xh_term2[i, j] = np.dot(xh_combine_weights[i], frac2*xh_combine_dips[i, :, j] * au_to_Debye)\
                         /np.sum(xh_combine_weights[i])
        mid = (combine_coords[:, 3, j] - combine_coords[:, 1, j])/2
        xh_term2_vec[i, j] = np.dot(xh_combine_weights[i], frac2 * (mid - xh_combine_coords[:, 0, j])) \
                      / np.sum(xh_combine_weights[i])

avg_term2_vec = np.average(term2_vec, axis=0)
std_term2_vec = np.std(term2_vec, axis=0)
avg_term2_o = np.average(term2_ov)
std_term2_o = np.std(term2_ov)
avg_term2_d = np.average(term2_dis)
std_term2_d = np.std(term2_dis)
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


# term3 = 0.14379963852224506
term3 = 0.046455953675400584
xh_term3 = 1.1154257246577732
std_term3 = 0.0
std_term3_sq = 0.0
std_term3_sq_freq = term3**2*freq*np.sqrt((std_term3_sq/term3**2)**2 + (std_freq/freq)**2)
xh_std_term3_sq_freq = xh_term3**2*freq2*np.sqrt((std_freq2/freq2)**2)
print(term3**2*freq*conversion)
print(std_term3_sq_freq*conversion)

full_error = np.sqrt(std_term1**2 + std_term2**2 + std_term3**2)
full_error_xh = np.sqrt(std_xh_term1**2 + xh_std_term2**2)
dipole_xh = xh_term1 + xh_term2 - xh_term3
dipole = term1 + term2 - term3
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









