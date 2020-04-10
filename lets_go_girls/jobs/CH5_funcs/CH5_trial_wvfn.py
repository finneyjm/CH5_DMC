import numpy as np
from scipy import interpolate

hh_relate = np.load('../params/sigma_hh_to_rch_exp_relationship_params.npy')
ch_stretch = np.load('../params/min_wvfns/GSW_min_CH_2.npy')
x = np.linspace(0.4, 6., 5000)
shift = x[np.argmax(ch_stretch)]
x -= shift
interp = interpolate.splrep(x, ch_stretch, s=0)


def hh_relate_fit(x, *args):
    a, b, c = args
    return a * np.exp(b * x) + c


def hh_dist(carts, rch):
    N = len(carts)
    coords = np.array(carts)
    coords -= np.broadcast_to(coords[:, None, 0], (N, 6, 3))
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


def trial_wvfn(coords):
    rch = ch_dist(coords)
    hh = hh_dist(coords, rch)
    shift = np.zeros((len(coords), 5))
    psi = np.zeros((len(coords), 5))
    for i in range(5):
        shift[:, i] = hh_relate_fit(hh[:, i], *hh_relate)
        psi[:, i] = interpolate.splev(rch[:, i] - shift[:, i], interp, der=0)
    return np.prod(psi, axis=1)




coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])

crds = np.array([coords_initial]*100)
print(trial_wvfn(crds))