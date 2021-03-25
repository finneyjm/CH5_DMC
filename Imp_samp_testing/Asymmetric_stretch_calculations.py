import numpy as np
import matplotlib.pyplot as plt
from ProtWaterPES import Dipole
import multiprocessing as mp
from Imp_samp_testing import EckartsSpinz
from Imp_samp_testing import MomentOfSpinz


har2wave = 219474.6

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


pool = mp.Pool(mp.cpu_count()-1)


ground_coords = np.zeros((10, 27, 5000, 5, 3))
ground_erefs = np.zeros((10, 20000))
ground_weights = np.zeros((10, 27, 5000))
for i in range(10):
    blah = np.load(f'ground_state_h3o2_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    ground_coords[i] = coords
    ground_erefs[i] = eref
    ground_weights[i] = weights

print(np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_zpe = np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_zpe = np.std(np.mean(ground_erefs[:, 5000:]*har2wave, axis=1))

excite_neg_coords = np.zeros((5, 27, 5000, 5, 3))
excite_neg_erefs = np.zeros((5, 20000))
excite_neg_weights = np.zeros((5, 27, 5000))
for i in range(5):
    blah = np.load(f'asym_excite_state_h3o2_{i+1}.npz')
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
    blah = np.load(f'asym_excite_state_h3o2_right_{i+1}.npz')
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


ground_coords = np.reshape(ground_coords, (10, 135000, 5, 3))
ground_coords = np.hstack((ground_coords, ground_coords[:, :, [0, 3, 4, 1, 2]]))
ground_weights = np.reshape(ground_weights, (10, 135000))
ground_weights = np.hstack((ground_weights, ground_weights))
ground_dists = np.zeros((10, 135000*2))
ground_dips = np.zeros((10, 135000*2, 3))
print(ground_coords[0, 0])
for i in range(10):
    eck = EckartsSpinz(ref, ground_coords[i], mass, planar=True)
    ground_coords[i] = eck.get_rotated_coords()
    ground_dists[i] = all_dists(ground_coords[i])[:, 0]
    ground_dips[i] = dip(ground_coords[i])
    # Mom = MomentOfSpinz(ground_coords[i], mass)
    # eigvals = 1/(2*Mom.gimme_dat_eigval())
    # amp, xx = np.histogram(eigvals[:, 0], weights=ground_weights[i], bins=75)
    # amp, xx = np.histogram(ground_dists[i], weights=ground_weights[i], bins=75, range=(-0.7, 0.7), density=True)
    # bin = (xx[1:] + xx[:-1]) / 2.
    # plt.plot(bin, amp)
    # amp, xx = np.histogram(eigvals[:, 1], weights=ground_weights[i], bins=75)
    # bin = (xx[1:] + xx[:-1]) / 2.
    # plt.plot(bin, amp)
    # amp, xx = np.histogram(eigvals[:, 2], weights=ground_weights[i], bins=75)
    # bin = (xx[1:] + xx[:-1]) / 2.
    # plt.plot(bin, amp)
print(ground_coords[0, 0])
print(ref)
excite_neg_coords = np.reshape(excite_neg_coords, (5, 135000, 5, 3))
excite_neg_coords = np.hstack((excite_neg_coords, excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
excite_neg_weights = np.reshape(excite_neg_weights, (5, 135000))
excite_neg_weights = np.hstack((excite_neg_weights, excite_neg_weights))
excite_neg_dists = np.zeros((5, 135000*2))
excite_neg_dips = np.zeros((5, 135000*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, excite_neg_coords[i], mass, planar=True)
    excite_neg_coords[i] = eck.get_rotated_coords()
    excite_neg_dists[i] = all_dists(excite_neg_coords[i])[:, 0]
    excite_neg_dips[i] = dip(excite_neg_coords[i])
    # amp, xx = np.histogram(excite_neg_dists[i], weights=excite_neg_weights[i], bins=75, range=(-0.5, 0.5), density=True)
    # bin = (xx[1:] + xx[:-1])/2
    # plt.plot(bin, amp)

excite_pos_coords = np.reshape(excite_pos_coords, (5, 135000, 5, 3))
excite_pos_coords = np.hstack((excite_pos_coords, excite_pos_coords[:, :, [0, 3, 4, 1, 2]]))
excite_pos_weights = np.reshape(excite_pos_weights, (5, 135000))
excite_pos_weights = np.hstack((excite_pos_weights, excite_pos_weights))
excite_pos_dists = np.zeros((5, 135000*2))
excite_pos_dips = np.zeros((5, 135000*2, 3))
for i in range(5):
    eck = EckartsSpinz(ref, excite_pos_coords[i], mass, planar=True)
    excite_pos_coords[i] = eck.get_rotated_coords()
    excite_pos_dists[i] = all_dists(excite_pos_coords[i])[:, 0]
    excite_pos_dips[i] = dip(excite_pos_coords[i])
    # Mom2 = MomentOfSpinz(np.concatenate((excite_pos_coords[i], excite_neg_coords[i]), axis=0), mass)
    # eigvals = 1/(2*Mom2.gimme_dat_eigval())
    # amp, xx = np.histogram(eigvals[:, 0], weights=np.hstack((excite_pos_weights[i], excite_neg_weights[i])), bins=75)
    # amp, xx = np.histogram(ground_dists[i], weights=ground_weights[i], bins=75, range=(-0.7, 0.7), density=True)
    # bin = (xx[1:] + xx[:-1]) / 2.
    # plt.plot(bin, amp)
    # amp, xx = np.histogram(eigvals[:, 1], weights=np.hstack((excite_pos_weights[i], excite_neg_weights[i])), bins=75)
    # bin = (xx[1:] + xx[:-1]) / 2.
    # plt.plot(bin, amp)
    # amp, xx = np.histogram(eigvals[:, 2], weights=np.hstack((excite_pos_weights[i], excite_neg_weights[i])), bins=75)
    # bin = (xx[1:] + xx[:-1]) / 2.
    # plt.plot(bin, amp)
    # amp, xx = np.histogram(np.hstack((excite_pos_dists[i], excite_neg_dists[i])), weights=np.hstack((excite_pos_weights[i], excite_neg_weights[i])), bins=75, range=(-0.7, 0.7), density=True)
    # bin = (xx[1:] + xx[:-1])/2
    # plt.plot(bin, amp)
# plt.xlabel('a (Bohr)')
# plt.show()

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_OH = (m_H*m_O)/(m_H+m_O)
omega_asym = 3070.648654929466/har2wave
mw = m_OH*omega_asym
conv_fac = 4.702e-7
km_mol = 5.33e6


def Harmonic_wvfn(x, state):
    if state == 1:
        return ((mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (x) ** 2)) * (2 * mw) ** (1 / 2) * (x))/20
    else:
        return ((mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (x) ** 2)))/20


au_to_Debye = 1/0.3934303
term1 = np.zeros((10, 3))
for i in range(10):
    frac = Harmonic_wvfn(ground_dists[i], 1)/Harmonic_wvfn(ground_dists[i], 0)
    for j in range(3):
        term1[i, j] = np.dot(ground_weights[i], frac*ground_dips[i, :, j]*au_to_Debye)/np.sum(ground_weights[i])
    # term1[i, 0] = np.dot(ground_weights[i], frac)/np.sum(ground_weights[i])
print(term1)
term1 = np.linalg.norm(term1, axis=-1)
# term1 = np.sum(term1, axis=-1)
std_term1 = np.std(term1)
term1 = np.average(term1)
# print(term1)
# print(std_term1)

freq = average_excite_energy-average_zpe
std_freq = np.sqrt(std_zpe**2 + std_excite_energy**2)
std_term1_sq = term1**2*np.sqrt((std_term1/term1)**2 + (std_term1/term1)**2)
std_term1_sq_freq = term1**2*freq*np.sqrt((std_term1_sq/term1**2)**2 + (std_freq/freq)**2)
conversion = conv_fac*km_mol
print(conversion*term1**2*freq)
print(std_term1_sq_freq*conversion)


term2 = np.zeros((5, 3))
combine_dists = np.zeros((5, 135000*4))
combine_dips = np.zeros((5, 135000*4, 3))
combine_weights = np.zeros(combine_dists.shape)
for i in range(5):
    combine_dists[i] = np.hstack((excite_neg_dists[i], excite_pos_dists[i]))
    combine_weights[i] = np.hstack((excite_neg_weights[i], excite_pos_weights[i]))
    combine_dips[i] = np.vstack((excite_neg_dips[i], excite_pos_dips[i]))
    frac = Harmonic_wvfn(combine_dists[i], 0)/Harmonic_wvfn(combine_dists[i], 1)
    for j in range(3):
        term2[i, j] = np.dot(combine_weights[i], frac*combine_dips[i, :, j]*au_to_Debye)/np.sum(combine_weights[i])
    # term2[i, 0] = np.dot(combine_weights[i], frac)/np.sum(combine_weights[i])
print(term2)
term2 = np.linalg.norm(term2, axis=-1)
# term2 = np.sum(term2, axis=-1)


std_term2 = np.std(term2)
term2 = np.average(term2)
# print(term2)
# print(std_term2)

std_term2_sq = term2**2*np.sqrt((std_term2/term2)**2 + (std_term2/term2)**2)
std_term2_sq_freq = term2**2*freq*np.sqrt((std_term2_sq/term2**2)**2 + (std_freq/freq)**2)
print(conversion*term2**2*freq)
print(std_term2_sq_freq*conversion)


# term3 = 0.14379963852224506
# term3 = 0.014661873657093441
# term3 = 0.008699279761412408
term3 = 0.022111361939871958
std_term3 = 0.0
std_term3_sq = 0.0
std_term3_sq_freq = term3**2*freq*np.sqrt((std_term3_sq/term3**2)**2 + (std_freq/freq)**2)
print(term3**2*freq*conversion)
print(std_term3_sq_freq*conversion)

full_error = np.sqrt(std_term1**2 + std_term2**2 + std_term3**2)
dipole = term1 + term2 - term3
full_error_sq = dipole**2*np.sqrt((full_error/dipole)**2 + (full_error/dipole)**2)
full_error = dipole**2*freq*np.sqrt((full_error_sq/dipole**2)**2 + (std_freq/freq)**2)
print(dipole**2*freq*conversion)
print(full_error*conversion)


