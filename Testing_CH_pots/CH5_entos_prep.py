import numpy as np
ang2bohr = (1.e-10)/(5.291772106712e-11)


def hh_dist(carts, rch):
    N = len(carts)
    coords = np.array(carts)
    coords -= np.broadcast_to(coords[:, None, 0], (N, 5 + 1, 3))
    coords[:, 1:] /= np.broadcast_to(rch[:, :, None], (N, 5, 3))
    hh = np.zeros((N, 5, 4))
    for i in range(4):
        for j in np.arange(i, 4):
            hh[:, i, j] = np.sqrt((coords[:, j + 2, 0] - coords[:, i + 1, 0]) ** 2 +
                                  (coords[:, j + 2, 1] - coords[:, i + 1, 1]) ** 2 +
                                  (coords[:, j + 2, 2] - coords[:, i + 1, 2]) ** 2)
            hh[:, j+1, i] = hh[:, i, j]
    hh_std = np.std(hh, axis=2)
    return hh_std


def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, 5))
    for i in range(5):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch



struct1 = np.array([[6.88559508e-05, 4.93690950e-04, 3.88319822e-05],
                    [1.31680501e-01, 2.08915388e+00, 4.28382015e-05],
                    [1.78720780e+00, -1.38497098e+00, -1.28395552e-04],
                    [2.23379787e+00, 3.55251899e-01, 1.72815454e-05],
                    [-8.24914405e-01, -6.29767500e-01, -1.77491284e+00],
                    [-8.25032413e-01, -6.29623798e-01, 1.77494229e+00]])


struct2 = np.array([[-5.64755991e-04, -3.29039113e-05, -2.66708386e-04],
                    [ 1.93148809e+00, 8.71012566e-05, -6.82345138e-01],
                    [ 1.04457635e-03, 8.91878228e-01, 2.08382460e+00],
                    [-8.33860550e-05, -8.91577028e-01, 2.08409317e+00],
                    [-1.14560685e+00, -1.65958631e+00, -4.97775064e-01],
                    [-1.14586541e+00, 1.65923086e+00, -4.97181938e-01]])


struct3 = np.array([[-2.08775053e-04, 3.73145321e-05, 3.87178045e-01],
                    [6.09550059e-05, 7.72349965e-05, -1.81025423e+00],
                    [1.79697955e+00, -9.59735207e-05, 1.38214361e+00],
                    [-1.79711477e+00, -3.78814499e-05, 1.38245520e+00],
                    [1.69049230e-04, -1.89516539e+00, -6.42150667e-01],
                    [1.13988280e-04, 1.89518470e+00, -6.42321119e-01]])

print(struct1/ang2bohr)

lets_go = np.array([[struct1], [struct2], [struct3]]).reshape((3, 6, 3))

r = ch_dist(lets_go)
std = hh_dist(lets_go, r)

from scipy import interpolate
import scipy.optimize

newhh = std.flatten()
newr = r.flatten()

ind = np.argsort(newhh)
newhh = newhh[ind]
newr = newr[ind]


def exp_fit(x, *args):
    a, b, c = args
    return a*np.exp(b*x) + c


params = [0.02375625, 6.29972956, 2.01513908]
fitted_params, _ = scipy.optimize.curve_fit(exp_fit, newhh, newr, p0=params)
print(fitted_params)
np.save('CH5_entos_exp_params', fitted_params)

g = np.linspace(0, 1, num=5000)

import matplotlib.pyplot as plt

plt.plot(g, exp_fit(g, *fitted_params), label='fit')
plt.scatter(std[0], r[0], label='min')
plt.scatter(std[1], r[1], label='cs')
plt.scatter(std[2], r[2], label='c2v')
plt.legend()
plt.show()