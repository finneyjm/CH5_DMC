import numpy as np
from scipy import interpolate

ch_stretch = np.load('Min_CH_average.npy')
interp = interpolate.splrep(ch_stretch[0], ch_stretch[1], s=0)


def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, 5))
    for i in range(5):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch


def trial_wavefunction(coords, atoms, extra_args):

    bools, ints, floats = extra_args
    atomic_units = bools[-1]

    if not atomic_units:
        from RynLib.RynUtils import Constants
        coords = Constants.convert(coords, "angstroms", in_AU=True)
    main_shape = coords.shape[:-2]
    n_walkers = max(np.prod(main_shape), 1)
    coords = np.reshape(coords, (n_walkers,) + coords.shape[-2:])

    rch = ch_dist(coords)
    shift = np.zeros((len(coords), 5))
    psi = np.zeros((len(coords), 5))
    for i in range(5):
        psi[:, i] = interpolate.splev(rch[:, i] - shift[:, i], interp, der=0)
    return np.prod(psi, axis=1)

