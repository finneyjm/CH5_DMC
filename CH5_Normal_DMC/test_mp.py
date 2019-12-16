from Coordinerds.CoordinateSystems import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import Timing_p3 as tm
import scipy.optimize
ang2bohr = 1.e-10/5.291772106712e-11


def hh_dist(coords):
    N = len(coords[:, 0, 0])
    hh = np.zeros((N, 5, 5))
    a = np.full((5, 5), True)
    np.fill_diagonal(a, False)
    mask = np.broadcast_to(a, (N, 5, 5))
    for i in range(4):
        for j in np.arange(i, 5):
            hh[:, i, j] += np.linalg.norm(coords[:, j+1, :] - coords[:, i+1, :], axis=1)
    hh += np.transpose(hh, (0, 2, 1))
    blah = hh[mask].reshape(N, 5, 4)
    return blah

wvfn = np.load('Non_imp_sampled/DMC_CH5_coords_20000_walkers_1.npy')
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
print(np.sqrt(coords_initial[1, 0]**2 + coords_initial[1, 1]**2))
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
wvfn_zmat = CoordinateSet(wvfn, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
blah = np.array(wvfn_zmat)
blah[:, :, 1] = np.ones((20000, 5))
wvfn_coords = CoordinateSet(blah, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
wvfn_hh = hh_dist(wvfn_coords)
# plt.scatter(np.std(wvfn_hh[5], axis=1), wvfn_zmat[5, :, 1], label='random walker 1')
# plt.scatter(np.std(wvfn_hh[6], axis=1), wvfn_zmat[6, :, 1], label='random walker 2')
# plt.scatter(np.std(wvfn_hh[7], axis=1), wvfn_zmat[7, :, 1], label='random walker 3')
# plt.scatter(np.std(wvfn_hh[8], axis=1), wvfn_zmat[8, :, 1], label='random walker 4')
# plt.scatter(np.std(wvfn_hh[9], axis=1), wvfn_zmat[9, :, 1], label='random walker 5')

coords = np.array([coords_initial]*10000)
zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
blah = np.array(zmat)
blah[:, :, 1] = np.ones((10000, 5))
coords = CoordinateSet(blah, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
hh = hh_dist(coords)


coords_initial_cs = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                       [1.931652478009080, -4.5126502395556294E-008, -0.6830921182334913],
                       [5.4640011799588715E-017, 0.8923685824271653, 2.083855680290835],
                       [-5.4640011799588715E-017, -0.8923685824271653, 2.083855680290835],
                       [-1.145620108130841, -1.659539840225091, -0.4971351597887673],
                       [-1.145620108130841, 1.659539840225091, -0.4971351597887673]])

coords_cs = np.array([coords_initial_cs]*10000)
zmat_cs = CoordinateSet(coords_cs, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
blah = np.array(zmat_cs)
blah[:, :, 1] = np.ones((10000, 5))
coords_cs = CoordinateSet(blah, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
hh_cs = hh_dist(coords_cs)

coords_initial_c2v = np.array([[0.000000000000000, 0.000000000000000, 0.386992362158741],
                       [0.000000000000000, 0.000000000000000, -1.810066283748844],
                       [1.797239666982623, 0.000000000000000, 1.381637275550612],
                       [-1.797239666982623, 0.000000000000000, 1.381637275550612],
                       [0.000000000000000, -1.895858229423645, -0.6415748897955779],
                       [0.000000000000000, 1.895858229423645, -0.6415748897955779]])

coords_c2v = np.array([coords_initial_c2v]*10000)
zmat_c2v = CoordinateSet(coords_c2v, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
blah = np.array(zmat_c2v)
blah[:, :, 1] = np.ones((10000, 5))
coords_c2v = CoordinateSet(blah, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
hh_c2v = hh_dist(coords_c2v)


rch = np.array([zmat[0, :, 1], zmat_cs[0, :, 1], zmat_c2v[0, :, 1]]).flatten()
std_hh = np.array([np.std(hh[0], axis=1), np.std(hh_cs[0], axis=1), np.std(hh_c2v[0], axis=1)]).flatten()
ind = np.argsort(std_hh)
new_hh = std_hh[ind]
new_rch = rch[ind]


def exp_fit(x, *args):
    a, b, c = args
    return a*np.exp(b*x) + c


def quad_fit(x, *args):
    a, b, c, d = args
    return a*x**3 + b*x**2 + c*x + d


params = [0.02388915, 6.29098812, 2.0149631]
fitted_params, _ = scipy.optimize.curve_fit(exp_fit, new_hh, new_rch, p0=params)
print(fitted_params)
params2 = [1, 1, 1, 1]
fitted_params2, _ = scipy.optimize.curve_fit(quad_fit, new_hh, new_rch, p0=params2)
print(fitted_params2)
g = np.linspace(0, 1., num=5000)
# np.save('sigma_hh_to_rch_exp_relationship', np.vstack((g, exp_fit(g, *fitted_params))))
# np.save('sigma_hh_to_rch_cub_relationship', np.vstack((g, quad_fit(g, *fitted_params2))))
plt.scatter(np.std(hh[0], axis=1), zmat[0, :, 1], label='min geo')
plt.scatter(np.std(hh_cs[0], axis=1), zmat_cs[0, :, 1], label='cs saddle')
plt.scatter(np.std(hh_c2v[0], axis=1), zmat_c2v[0, :, 1], label='c2v saddle')
plt.plot(new_hh, new_rch)
plt.plot(g, exp_fit(g, *fitted_params), label='exp fit')
plt.plot(g, quad_fit(g, *fitted_params2), label='quadratic fit')
plt.legend()
plt.xlabel(r'$\sigma_{HH}$')
plt.ylabel('rCH (Bohr)')
plt.show()
plt.close()
from scipy import interpolate
interp = interpolate.splrep(g, quad_fit(g, *fitted_params2), s=0)
import time
blah = np.random.uniform(0, 0.5, 10000)
print(blah)
start = time.time()
y = interpolate.splev(blah, interp, der=0)
print(y)
end = time.time()
print(f'interpolate took : {end-start}')

start = time.time()
y = quad_fit(blah, *fitted_params2)
print(y)
end = time.time()
print(f'function took : {end-start}')

start = time.time()
y = exp_fit(blah, *fitted_params)
print(y)
end = time.time()
print(f'exp function took : {end-start}')


