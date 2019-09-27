import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
from scipy import interpolate

ang2bohr = 1.e-10/5.291772106712e-11

switch_1 = np.load('Switch_wvfns/Switch_min_wvfn_speed_1.0.npy')
switch_61 = np.load('Switch_wvfns/Switch_min_wvfn_speed_61.0.npy')

diff = switch_1[1, :] - switch_61[1, :]


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


params2 = [1.88877633e+00, 2.51971525e+00,
           8.43433842e-02, 1.01236299e-01,
           -8.80282312e-04, -1.14388138e-03]
params3 = [2.51970920e+00, 1.86666079e+00, 1.86604700e+00,
           1.01233007e-01, 6.87064603e-02, 6.76311871e-02,
           -1.14386889e-03, -2.48801819e-02, 2.39392335e-02]
params4 = [1.60814687e+00, 1.93639591e+00, 2.51970941e+00, 2.53669481e+00,
           8.27882793e-02, 4.23745227e-02, 1.01245190e-01, 4.33202947e-02,
           -4.73918371e-04, 8.04262118e-03, -1.14392682e-03, -8.54469025e-03]
params5 = [2.10828978e+00, 1.93651052e+00, 2.56791887e+00, 2.43679819e+00, 2.09999942e-00,
           8.27965595e-02, 4.23093410e-02, 3.26922669e+03, 4.32236699e-02, 1.00032945e-02,
           -4.74749341e-04, 8.31477826e-03, 1.10949028e-04, -8.81592120e-03, -1.63499727e-14]
params6 = [2.18803128e+00, 6.40750209e-01, 2.46499111e+00, 1.85609002e+00, 2.31284780e+00, 1.85770915e+00,
           5.80633799e-02, 4.27488717e-02, 1.76867519e-01, 8.64363297e-02, 4.58593707e-02, 8.79752242e-02,
           1.53888246e-03, 1.63763662e-17, -1.81186585e-03, 1.65727420e-02, 2.26411116e-04, -1.75956746e-02]

fitted_params2, _ = scipy.optimize.curve_fit(bi_norm, switch_1[0, :], diff, p0=params2)
# print(fitted_params2)
fitted_params3, _ = scipy.optimize.curve_fit(tri_norm, switch_1[0, :], diff, p0=params3)
# print(fitted_params3)
fitted_params4, _ = scipy.optimize.curve_fit(quad_norm, switch_1[0, :], diff, p0=params4)
# print(fitted_params4)
fitted_params5, _ = scipy.optimize.curve_fit(cat_norm, switch_1[0, :], diff, p0=params5)

fitted_params6, _ = scipy.optimize.curve_fit(sex_norm, switch_1[0, :], diff, p0=params6)
# print(fitted_params6)

# plt.plot(switch_1[0, :], diff, label='difference')
# plt.plot(switch_1[0, :], switch_1[1, :], label='switch alpha 1')
plt.plot(switch_1[0, :], switch_61[1, :], label='switch alpha 61')
# plt.plot(switch_1[0, :], bi_norm(switch_1[0, :], *fitted_params2), label='2 Gaussians')
# plt.plot(switch_1[0, :], switch_1[1, :] - tri_norm(switch_1[0, :], *fitted_params3), label='3 Gaussians')
plt.plot(switch_1[0, :], switch_1[1, :] - quad_norm(switch_1[0, :], *fitted_params4), label='4 Gaussians')
plt.plot(switch_1[0, :], switch_1[1, :] - cat_norm(switch_1[0, :], *fitted_params5), label='5 Gaussians')
plt.plot(switch_1[0, :], switch_1[1, :] - sex_norm(switch_1[0, :], *fitted_params6), label='6 Gaussians')
plt.xlim(1.5, 3.2)
plt.legend()
plt.show()








