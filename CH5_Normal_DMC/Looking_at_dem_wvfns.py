from Coordinerds.CoordinateSystems import *
import matplotlib.pyplot as plt

ang2bohr = 1.e-10/5.291772106712e-11
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]


def regular_wvfns(walkers):
    coords = np.load(f'DMC_CH5_coords.npy')
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    weights = np.load(f'DMC_CH5_weights.npy')
    fig, axes = plt.subplots(1, 5, figsize=(25, 8))
    for CH in range(5):
        amp, xx = np.histogram(zmat[:, CH, 1]/ang2bohr, weights=weights[0, :], bins=25, range=(0.6, 1.8), density=True)
        bins = (xx[1:] + xx[:-1])/2.
        axes[CH].plot(bins, amp)
        axes[CH].set_title(f'CH stretch {CH+1}')
        axes[CH].set_xlabel('rCH (Angstrom)')
        axes[CH].set_ylabel('Probability Density')
    plt.tight_layout()
    fig.savefig('Normal_DMC_wvfns.png')
    plt.close(fig)


def imp_samp__wvfns(walkers):
    coords = np.load(f'Imp_samp_DMC_CH5_coords_{walkers}_walkers_{1}.npy')
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    weights = np.load(f'Imp_samp_DMC_CH5_weights_{walkers}_walkers_{1}.npy')
    fig, axes = plt.subplots(1, 5, figsize=(25, 8))
    for CH in range(5):
        amp, xx = np.histogram(zmat[:, CH, 1]/ang2bohr, weights=weights[0, :], bins=25, range=(0.6, 1.8), density=True)
        bins = (xx[1:] + xx[:-1])/2.
        axes[CH].plot(bins, amp)
        axes[CH].set_title(f'CH stretch {CH+1}')
        axes[CH].set_xlabel('rCH (Angstrom)')
        axes[CH].set_ylabel('Probability Density')
    plt.tight_layout()
    fig.savefig('Imp_samp_DMC_wvfns.png')
    plt.close(fig)


def biased_samp__wvfns(walkers):
    coords = np.load(f'Biased_DMC_CH5_coords_{walkers}_walkers_{1}.npy')
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    weights = np.load(f'Biased_DMC_CH5_weights_{walkers}_walkers_{1}.npy')
    fig, axes = plt.subplots(1, 5, figsize=(25, 8))
    for CH in range(5):
        amp, xx = np.histogram(zmat[:, CH, 1]/ang2bohr, weights=weights[0, :], bins=25, range=(0.6, 1.8), density=True)
        bins = (xx[1:] + xx[:-1])/2.
        axes[CH].plot(bins, amp)
        axes[CH].set_title(f'CH stretch {CH+1}')
        axes[CH].set_xlabel('rCH (Angstrom)')
        axes[CH].set_ylabel('Probability Density')
    plt.tight_layout()
    fig.savefig('Biased_DMC_wvfns.png')
    plt.close(fig)


imp_samp__wvfns(5000)
biased_samp__wvfns(5000)
regular_wvfns(0)



