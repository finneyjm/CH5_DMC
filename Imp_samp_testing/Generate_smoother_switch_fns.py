import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
ang2bohr = 1.e-10/5.291772106712e-11


def uni_norm(x, *args):
    m, s, k = args
    return k*scipy.stats.norm.pdf(x, loc=m, scale=s)


def bi_norm(x, *args):
    m1, m2, s1, s2, k1, k2 = args
    ret = k1 * scipy.stats.norm.pdf(x, loc=m1, scale=s1)
    ret += k2 * scipy.stats.norm.pdf(x, loc=m2, scale=s2)
    return ret


def tri_norm(x, *args):
    m1, m2, m3, s1, s2, s3, k1, k2, k3 = args
    ret = k1*scipy.stats.norm.pdf(x, loc=m1, scale=s1)
    ret += k2*scipy.stats.norm.pdf(x, loc=m2, scale=s2)
    ret += k3*scipy.stats.norm.pdf(x, loc=m3, scale=s3)
    return ret


def quad_norm(x, *args):
    m1, m2, m3, m4, s1, s2, s3, s4, k1, k2, k3, k4 = args
    ret = k1*scipy.stats.norm.pdf(x, loc=m1, scale=s1)
    ret += k2*scipy.stats.norm.pdf(x, loc=m2, scale=s2)
    ret += k3*scipy.stats.norm.pdf(x, loc=m3, scale=s3)
    ret += k4*scipy.stats.norm.pdf(x, loc=m4, scale=s4)
    return ret


def cat_norm(x, *args):
    m1, m2, m3, m4, m5, s1, s2, s3, s4, s5, k1, k2, k3, k4, k5 = args
    ret = k1*scipy.stats.norm.pdf(x, loc=m1, scale=s1)
    ret += k2*scipy.stats.norm.pdf(x, loc=m2, scale=s2)
    ret += k3*scipy.stats.norm.pdf(x, loc=m3, scale=s3)
    ret += k4*scipy.stats.norm.pdf(x, loc=m4, scale=s4)
    ret += k5*scipy.stats.norm.pdf(x, loc=m5, scale=s5)
    return ret


def sex_norm(x, *args):
    m1, m2, m3, m4, m5, m6, s1, s2, s3, s4, s5, s6, k1, k2, k3, k4, k5, k6 = args
    ret = k1*scipy.stats.norm.pdf(x, loc=m1, scale=s1)
    ret += k2*scipy.stats.norm.pdf(x, loc=m2, scale=s2)
    ret += k3*scipy.stats.norm.pdf(x, loc=m3, scale=s3)
    ret += k4*scipy.stats.norm.pdf(x, loc=m4, scale=s4)
    ret += k5*scipy.stats.norm.pdf(x, loc=m5, scale=s5)
    ret += k6*scipy.stats.norm.pdf(x, loc=m6, scale=s6)
    return ret


psi = np.load(f'Switch_wvfns/Switch_min_wvfn_speed_1.0.npy')
psi[0, :] /= ang2bohr

params = [1.16255402,
          0.15132058,
          0.04107842]
params2 = [1.04606652, 1.22983871,
           0.07755864, 0.11993431,
           0.01272802, 0.02722187]
params3 = [1.07620131, 1.29043464, 1.24081179,
           0.09054991, 0.10427892, 0.04978319,
           0.0216147,  0.01515032, 0.00332449]
params4 = [1.35281857e+00, 1.23710693e+00, 1.10441447e+00, 1.15000767e+00, # alpha 1
           0.1, 0.1, 0.1, 0.1,
           0.001, 0.001, 0.001, 0.01]
# params4 = [1.21466852e+00, 1.27331772e+00, 1.08774732e+00, 1.32837319e+00,  # alpha 61
#            3.07846157e-02, 5.45377168e-02, 9.55975280e-02, 9.83518356e-02,
#            6.08981341e-04, 2.02412967e-03, 1.05996284e-02, 4.11202869e-03]
params5 = [1.14118446e+00, 1.22986460e+00, 1.30170998e+00, 9.85271568e-01, 1.06123029e+00,   # alpha 1
           9.61942128e-02, 1.13054080e-01, 1.30585264e-01, 7.55606468e-02, 8.43913715e-02,
           5.12341232e-03, 5.79436741e-03, 1.98733049e-03, 6.25321635e-04, 3.25785573e-03]
# params5 = [1.01161854e+00, 1.21556278e+00, 1.21949825e+00, 1.06617219e+00, 1.11192790e+00,   # alpha 61
#            6.90652381e-02, 6.09564545e-02, 1.29818055e-01, 4.08518207e-02, 2.30681854e-02,
#            2.96559107e-03, 7.92807909e-04, 1.21065770e-02, 1.13998715e-03, 3.42547510e-04]
params6 = [1.25613356e+00, 9.55986225e-01, 1.32108354e+00, 1.17716270e+00, 1.02710545e+00, 1.09823286e+00,   #alpha 1
           1.15149634e-01, 7.12697401e-02, 1.32119476e-01, 1.00268652e-01, 7.87531556e-02, 8.77497523e-02,
           4.55979221e-03, 2.40454046e-04, 1.09824679e-03, 4.96764073e-03, 1.77758343e-03, 4.14456678e-03]
# params6 = [1.25974714e+00, 6.42542264e-01, 1.30298760e+00, 1.20720087e+00, 1.10624566e+00, 1.07258601e+00,    # alpha 61
#            6.12624409e-02, 2.27948025e-02, 1.08228964e-01, 3.26585959e-02, 3.05799159e-02, 8.99801279e-02,
#            2.28421107e-03, -3.46397657e-08, 5.22218916e-03, 5.36755415e-04, 3.06070649e-04, 8.99633494e-03]


x = np.linspace(0.4, 6, 5000)/ang2bohr
# fitted_params, _ = scipy.optimize.curve_fit(uni_norm, psi[0, :], psi[1, :], p0=params)
# fitted_params2, _ = scipy.optimize.curve_fit(bi_norm, psi[0, :], psi[1, :], p0=params2)
# fitted_params3, _ = scipy.optimize.curve_fit(tri_norm, psi[0, :], psi[1, :], p0=params3)
# fitted_params4, _ = scipy.optimize.curve_fit(quad_norm, psi[0, :], psi[1, :], p0=params4)
# fitted_params5, _ = scipy.optimize.curve_fit(cat_norm, psi[0, :], psi[1, :], p0=params5)
# fitted_params6, _ = scipy.optimize.curve_fit(sex_norm, psi[0, :], psi[1, :], p0=params6)
# print(fitted_params6)
# broad = 1.00
# fitted_params2 = np.array(fitted_params2)
# fitted_params2[2:6] *= broad
# plt.plot(psi[0, :], psi[1, :])
# # plt.plot(x, uni_norm(x, *fitted_params), label='1 Gaussian')
# # plt.plot(x, bi_norm(x, *fitted_params2), label=f'2 Gaussian fit with {broad}x broadening')
# # plt.plot(x, tri_norm(x, *fitted_params3), label='3 Gaussians')
# # plt.plot(x, quad_norm(x, *fitted_params4), label='4 Gaussians')
# # plt.plot(x, cat_norm(x, *fitted_params5), label='5 Gaussians')
# # plt.plot(x, sex_norm(x, *fitted_params6), label='6 Gaussians')
# plt.xlim(0.7, 1.0)
# # plt.ylim(0.08, 0.13)
# plt.legend()
# print(fitted_params2)
# plt.savefig('example_fitted_wvfn.png')
# plt.close()

# alpha = [1, 5, 11, 21, 31, 41, 51, 61]
# for i in range(len(alpha)):
#     interp = interpolate.splrep(psi[0, :], psi[1, :], s=0)
#     plt.plot(x, interpolate.splev(x, interp, der=0)/np.linalg.norm(interpolate.splev(x, interp, der=0)), label='wvfn')
#     plt.plot(x, interpolate.splev(x, interp, der=1)/np.linalg.norm(interpolate.splev(x, interp, der=1)), label='first derivative')
#     plt.plot(x, interpolate.splev(x, interp, der=2)/np.linalg.norm(interpolate.splev(x, interp, der=2)), label='second derivative')
#     plt.legend()
#     plt.savefig(f'looking_at_dem_derivatives_{alpha[i]}')
#     plt.close()

# Rad_shift = fitted_params[0]-2.11596/ang2bohr
# for i in range(11):
#     broad = round(1.00 + i*0.1, 5)
#     bro_params = np.array(fitted_params5)
#     bro_params[0:5] -= Rad_shift
#     bro_params[5:10] *= broad
#     y = cat_norm(x, *bro_params)
#     # y /= np.linalg.norm(y)
#     plt.plot(psi[0, :], psi[1, :])
#     plt.plot(x, y, label=f'5 Gaussian fit with {broad}x broadening')
#     plt.legend()
#     # plt.ylim(0.1, 0.13)
#     plt.savefig(f'Gauss_wvfn_fits/Gaussian_5_alpha_1_Radau_broadening_{broad}.png')
#     plt.close()
#     x_new = x*ang2bohr
#     np.save(f'Switch_wvfns/5_Gauss_Switch_wvfn_min_alpha_1_Radau_{broad}x_broadening', np.vstack((x_new, y)))

# y = tri_norm(x, *fitted_params3)
# x_new = x*ang2bohr
# np.save(f'Switch_wvfn_min_alpha_1_{1.0}x_broadening', np.vstack((x_new, y)))
# diff = psi[1, :] - y
# plt.plot(x, diff)
# # plt.plot(x, y)
# # plt.plot(psi[0, :], psi[1, :])
# plt.xlabel('rCH (Angstrom)')
# plt.title(r'Difference Between $\psi$ and the Fit')
# plt.savefig('Difference_between_psi_and_fit.png')
# plt.close()
#
# interp = interpolate.splrep(psi[0, :], psi[1, :], s=0)
# interp_fit = interpolate.splrep(x, y, s=0)
# der1 = interpolate.splev(psi[0, :], interp, der=1)
# der1_fit = interpolate.splev(x, interp_fit, der=1)
# der2 = interpolate.splev(psi[0, :], interp, der=2)
# der2_fit = interpolate.splev(x, interp_fit, der=2)
#
# diff_der1 = der1_fit-der1
# diff_der2 = der2_fit-der2
#
#
# plt.plot(x, diff_der1)
# # plt.plot(x, der1_fit)
# # plt.plot(psi[0, :], der1)
# plt.xlabel('rCH (Angstrom)')
# plt.title(f'Difference Between d$\psi$/dr and the Fit')
# plt.savefig('Difference_between_dpsidr_and_fit.png')
# plt.close()
#
# plt.plot(x, diff_der2)
# # plt.plot(x, der2_fit)
# # plt.plot(psi[0, :], der2)
# plt.xlabel('rCH (Angstrom)')
# plt.title(f'Difference Between d2$\psi$/dr2 and the Fit')
# plt.savefig('Difference_between_dpsidr2_and_fit.png')
# plt.close()
bings = 50
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
amp_all = np.zeros((5, 5, bings))
for i in range(5):
    coords = np.load(f'Non_imp_sampled/DMC_CH5_coords_20000_walkers_{i+1}.npy')
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    weights = np.load(f'Non_imp_sampled/DMC_CH5_weights_20000_walkers_{i+1}.npy')
    for CH in range(5):
        amp, xx = np.histogram(zmat[:, CH, 1]/ang2bohr, weights=weights[0, :], bins=bings, range=(0.6, 1.8), density=True)
        bins = (xx[1:] + xx[:-1])/2.
        amp_all[i, CH, :] += amp

amp = np.mean(np.mean(amp_all, axis=0), axis=0)

x = np.linspace(0.4, 6., 5000)/ang2bohr
psi = np.zeros((5, 5000))
type = 'min'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')

min_psi = np.mean(psi, axis=0)
type = 'cs'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')

cs_psi = np.mean(psi, axis=0)

type = 'c2v'
for i in range(5):
    psi[i, :] += np.load(f'{type}_wvfns/GSW_{type}_CH_{i+1}.npy')

c2v_psi = np.mean(psi, axis= 0)

new_psi = np.mean(np.vstack((min_psi, cs_psi, c2v_psi)), axis=0)
# fitted_params, _ = scipy.optimize.curve_fit(uni_norm, bins, amp, p0=params)
# fitted_params2, _ = scipy.optimize.curve_fit(bi_norm, bins, amp, p0=params2)
# # fitted_params3, _ = scipy.optimize.curve_fit(tri_norm, bins, amp, p0=params3)
# fitted_params4, _ = scipy.optimize.curve_fit(quad_norm, bins, amp, p0=params4)
# # fitted_params5, _ = scipy.optimize.curve_fit(cat_norm, bins, amp, p0=params5)
# # fitted_params6, _ = scipy.optimize.curve_fit(sex_norm, bins, amp, p0=params6)
#
plt.plot(bins, amp, label='averaged wvfn')

# plt.plot(x, min_psi*112., label='averaged min psi')
# plt.plot(x, cs_psi*42., label='averaged cs psi')
# plt.plot(x, c2v_psi*25., label='averaged c2v psi')
# # plt.plot(x, uni_norm(x, *fitted_params), label='1 Gaussian fit')
# # plt.plot(x, bi_norm(x, *fitted_params2), label='2 Gaussian fit')
# # plt.plot(x, tri_norm(x, *fitted_params3), label='3 Gaussian fit')
# plt.plot(x, quad_norm(x, *fitted_params4), label='4 Gaussian fit')
# asdf = quad_norm(x, *fitted_params4)
# interp = interpolate.splrep(x, new_psi, s=0)
# plt.plot(x, interpolate.splev(x, interp, der=1))
# plt.plot(x, interpolate.splev(x, interp, der=2))
# # plt.plot(x, cat_norm(x, *fitted_params5), label='5 Gaussian fit')
# # plt.plot(x, sex_norm(x, *fitted_params6), label='6 Gaussian fit')
plt.xlim(0.8, 1.8)
# # plt.ylim(2.0, 3.2)
plt.legend()
plt.savefig('testing_these_fits.png')
#
# np.save('Fits_CH_stretch_wvfns/4_Guass_fit', np.vstack((x*ang2bohr, quad_norm(x, *fitted_params4))))
# np.save('Fits_CH_stretch_wvfns/1_Guass_fit', np.vstack((x*ang2bohr, uni_norm(x, *fitted_params))))
# np.save('Fits_CH_stretch_wvfns/2_Guass_fit', np.vstack((x*ang2bohr, bi_norm(x, *fitted_params2))))
np.save(f'Fits_CH_stretch_wvfns/Average_no_fit', np.vstack((x*ang2bohr, new_psi)))