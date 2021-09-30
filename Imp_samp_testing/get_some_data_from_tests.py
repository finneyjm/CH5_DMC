import numpy as np
import matplotlib.pyplot as plt
from ProtWaterPES import Dipole
import multiprocessing as mp
from Imp_samp_testing import EckartsSpinz
from Imp_samp_testing import MomentOfSpinz
from scipy import interpolate

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_OH = (m_H*m_O)/(m_H+m_O)
omega_asym = 3815.044564/har2wave
big_Roo_grid = np.linspace(4, 5.4, 1000)
big_sp_grid = np.linspace(-1.2, 1.2, 1000)
X, Y = np.meshgrid(big_sp_grid, big_Roo_grid)


# z_ground_no_der = np.load('z_ground_no_der.npy')
#
# ground_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                        z_ground_no_der.flatten())
#
# z_excite_xh_no_der = np.load('z_excite_xh_no_der.npy')
#
# excite_xh_no_der = interpolate.CloughTocher2DInterpolator(list(zip(X.flatten(), Y.flatten())),
#                                                           z_excite_xh_no_der.flatten())


def get_da_psi(coords, stretch, excite):
    # psi = np.zeros((len(coords)))
    # dists = all_dists(coords)
    mw_h = m_OH * omega_asym
    if stretch == 'sp' and excite is True:
        # psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 0] ** 2))
        psi = excite_xh_no_der(coords[:, 0], coords[:, 1])
    elif stretch == 'a' and excite is True:
        psi = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * coords ** 2)) * \
                    (2 * mw_h) ** (1 / 2) * coords
        # psi[:, 1] = ground_no_der(dists[:, -1], dists[:, -2])
    elif stretch == 'sp' and excite is False:
        psi = ground_no_der(coords[:, 0], coords[:, 1])
    else:
        psi = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * coords ** 2))
    return psi

binzz = 100
ground_coords = np.zeros((10, 27, 500))

ground_erefs = np.zeros((10, 20000))
ground_weights = np.zeros((10, 27, 500))
for i in range(10):
    blah = np.load(f'ground_state_1d_a_h3o2_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    ground_coords[i] = coords.squeeze()
    ground_erefs[i] = eref
    ground_weights[i] = weights

print(np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_zpe = np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave

excite_neg_coords = np.zeros((5, 27, 500))
excite_neg_erefs = np.zeros((5, 20000))
excite_neg_weights = np.zeros((5, 27, 500))
for i in range(5):
    blah = np.load(f'Asym_excite_state_1d_a_h3o2_left_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    excite_neg_coords[i] = coords.squeeze()
    excite_neg_erefs[i] = eref
    excite_neg_weights[i] = weights

print(np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_neg_a_energy = np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave

excite_pos_coords = np.zeros((5, 27, 500))
excite_pos_erefs = np.zeros((5, 20000))
excite_pos_weights = np.zeros((5, 27, 500))
for i in range(5):
    blah = np.load(f'Asym_excite_state_1d_a_h3o2_right_{i+1}.npz')
    coords = blah['coords']
    eref = blah['Eref']
    weights = blah['weights']
    excite_pos_coords[i] = coords.squeeze()
    excite_pos_erefs[i] = eref
    excite_pos_weights[i] = weights

print(np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_pos_a_energy = np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave

combine_coords = np.hstack((excite_neg_coords, excite_pos_coords))
combine_weights = np.hstack((excite_neg_weights, excite_pos_weights))

H0_ground = np.zeros((10, 27*500))
H1_ground = np.zeros((10, 27*500))
H0_excite = np.zeros((5, 54*500))
H1_excite = np.zeros((5, 54*500))
for i in range(10):
    H0_ground[i] = get_da_psi(ground_coords[i].reshape(27*500), 'a', False)
    H1_ground[i] = get_da_psi(ground_coords[i].reshape(27*500), 'a', True)

for i in range(5):
    H0_excite[i] = get_da_psi(combine_coords[i].reshape(27*500*2), 'a', False)
    H1_excite[i] = get_da_psi(combine_coords[i].reshape(27*500*2), 'a', True)

term1 = np.zeros(10)
amp_ground = np.zeros((10, binzz))
amp_excite = np.zeros((5, binzz))
term2 = np.zeros(5)
for i in range(10):
    amp_ground[i], xx = np.histogram(ground_coords[i].reshape(27*500), weights=ground_weights[i].reshape(27*500),
                                     bins=binzz, range=(-0.5, 0.5), density=True)
    term1[i] = np.dot(ground_weights[i].reshape(27*500), H1_ground[i]/H0_ground[i]*ground_coords[i].reshape(27*500))\
        /np.sum(ground_weights[i].reshape(27*500))

for i in range(5):
    amp_excite[i], xx = np.histogram(combine_coords[i].reshape(27*500*2), weights=combine_weights[i].reshape(27*500*2),
                                     bins=binzz, range=(-0.5, 0.5), density=True)
    term2[i] = np.dot(combine_weights[i].reshape(27*500*2), H0_excite[i]/H1_excite[i]*combine_coords[i].reshape(27*500*2))\
        /np.sum(combine_weights[i].reshape(27*500*2))

print(np.average(term1))
print(np.std(term1))
print(np.average(term2))
print(np.std(term2))
x = (xx[1:] + xx[:-1]) / 2.

p0 = get_da_psi(x, 'a', False)
p1 = get_da_psi(x, 'a', True)
p0 = p0/np.linalg.norm(p0)
p1 = p1/np.linalg.norm(p1)





# amp_ground = np.mean(amp_ground, a)
# amp_excite = np.mean(amp_excite, axis=0)

# amp_ground = amp_ground/np.linalg.norm(amp_ground, axis=1)
# amp_excite = amp_excite/np.linalg.norm(amp_excite, axis=1)
truth = p0*p1
import matplotlib.pyplot as plt
# for i in range(5):
    # print(np.dot(amp_ground[i], x*p1/p0)/np.sum(amp_ground[i]))
    # print(np.dot(amp_excite[i], x*p0/p1)/np.sum(amp_excite[i]))
    # mod1 = amp_ground[i]*(p1/p0)
    # mod2 = amp_excite[i]*(p0/p1)
    # plt.plot(x/ang2bohr, mod1/np.max(mod1)*np.max(truth), label=r'$\rm{f_0\Psi_1/\Psi_0}$')
plt.plot(x/ang2bohr, x*p0/p1, label=r'$\rm{f_1\Psi_0/\Psi_1}$')
# plt.plot(x/ang2bohr, truth, label=r'$\rm{\Psi_1\Psi_0}$')
# plt.plot(x/ang2bohr, mod1+mod2-truth, label='blah')
plt.xlabel(r'a $\rm{\AA}$')
# plt.ylim(0, 3.2)
plt.legend()
plt.tight_layout()
plt.savefig('Checking_asymmetric_stretch_1d')
plt.show()


# ground_coords = np.zeros((10, 27, 500, 2))
# ground_erefs = np.zeros((10, 20000))
# ground_weights = np.zeros((10, 27, 500))
# for i in range(10):
#     blah = np.load(f'ground_state_2d_sp_h3o2_{i+1}.npz')
#     coords = blah['coords']
#     eref = blah['Eref']
#     weights = blah['weights']
#     ground_coords[i] = coords
#     ground_erefs[i] = eref
#     ground_weights[i] = weights
#
# print(np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave)
# average_zpe = np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave
#
# excite_neg_coords = np.zeros((5, 27, 500, 2))
# excite_neg_erefs = np.zeros((5, 20000))
# excite_neg_weights = np.zeros((5, 27, 500))
# for i in range(5):
#     blah = np.load(f'XH_excite_state_2d_sp_h3o2_left_{i+1}.npz')
#     coords = blah['coords']
#     eref = blah['Eref']
#     weights = blah['weights']
#     excite_neg_coords[i] = coords
#     excite_neg_erefs[i] = eref
#     excite_neg_weights[i] = weights
#
# print(np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave)
# average_neg_a_energy = np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave
#
# excite_pos_coords = np.zeros((5, 27, 500, 2))
# excite_pos_erefs = np.zeros((5, 20000))
# excite_pos_weights = np.zeros((5, 27, 500))
# for i in range(5):
#     blah = np.load(f'XH_excite_state_2d_sp_h3o2_right_{i+1}.npz')
#     coords = blah['coords']
#     eref = blah['Eref']
#     weights = blah['weights']
#     excite_pos_coords[i] = coords
#     excite_pos_erefs[i] = eref
#     excite_pos_weights[i] = weights
#
# print(np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave)
# average_pos_a_energy = np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave
#
# combine_coords = np.hstack((excite_neg_coords, excite_pos_coords))
# combine_weights = np.hstack((excite_neg_weights, excite_pos_weights))
#
# H0_ground = np.zeros((10, 27*500))
# H1_ground = np.zeros((10, 27*500))
# H0_excite = np.zeros((5, 54*500))
# H1_excite = np.zeros((5, 54*500))
# for i in range(10):
#     H0_ground[i] = get_da_psi(ground_coords[i].reshape(27*500, 2), 'sp', False)
#     H1_ground[i] = get_da_psi(ground_coords[i].reshape(27*500, 2), 'sp', True)
#
# for i in range(5):
#     H0_excite[i] = get_da_psi(combine_coords[i].reshape(27*500*2, 2), 'sp', False)
#     H1_excite[i] = get_da_psi(combine_coords[i].reshape(27*500*2, 2), 'sp', True)
#
# term1 = np.zeros(10)
# term2 = np.zeros(5)
# for i in range(10):
#     term1[i] = np.dot(ground_weights[i].reshape(27*500), H1_ground[i]/H0_ground[i]*ground_coords[i].reshape(27*500, 2)[:, 0])\
#         /np.sum(ground_weights[i].reshape(27*500))
#
# for i in range(5):
#     term2[i] = np.dot(combine_weights[i].reshape(27*500*2), H0_excite[i]/H1_excite[i]*combine_coords[i].reshape(27*500*2, 2)[:, 0])\
#         /np.sum(combine_weights[i].reshape(27*500*2))
#
# print(np.average(term1))
# print(np.std(term1))
# print(np.average(term2))
# print(np.std(term2))



