from Coordinerds.CoordinateSystems import *
import matplotlib.pyplot as plt
from scipy import interpolate

ang2bohr = 1.e-10/5.291772106712e-11
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
psi_t = np.load('Switch_min_wvfn_speed_61.0.npy')
interp = interpolate.splrep(psi_t[0, :]/ang2bohr, psi_t[1, :], s=0)


def regular_wvfns(walkers):
    fig, axes = plt.subplots(1, 5, figsize=(25, 8))
    amp_all = np.zeros((2, 5, 5, 25))
    for i in range(5):
        coords = np.load(f'Imp_samp_DMC_CH5_randomly_sampled_coords_alpha_61_{walkers}_walkers_{i+1}.npy')
        zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
        weights = np.load(f'Imp_samp_DMC_CH5_randomly_sampled_weights_alpha_61_{walkers}_walkers_{i+1}.npy')

        for CH in range(5):
            amp, xx = np.histogram(zmat[:, CH, 1] / ang2bohr, weights=weights[1, :], bins=25, range=(0.6, 1.8),
                                   density=True)
            bins = (xx[1:] + xx[:-1]) / 2.
            psi = interpolate.splev(bins, interp, der=0)
            amp_all[0, i, CH, :] += amp/psi
        # coords = np.load(f'Biased_DMC_CH5_coords_{walkers}_walkers_{1}.npy')
        # zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
        # weights = np.load(f'Biased_DMC_CH5_weights_{walkers}_walkers_{1}.npy')
        # for CH in range(5):
        #     amp, xx = np.histogram(zmat[:, CH, 1]/ang2bohr, weights=weights[0, :], bins=25, range=(0.6, 1.8), density=True)
        #     bins = (xx[1:] + xx[:-1])/2.
        #     if CH is not 0:
        #         amp_all[1, i, CH, :] += amp
        #     else:
        #         psi = interpolate.splev(bins, interp, der=0)
        #         amp_all[1, i, CH, :] += amp/psi
        coords = np.load(f'DMC_CH5_randomly_sampled_coords_{walkers}_walkers_{i+1}.npy')
        zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
        weights = np.load(f'DMC_CH5_randomly_sampled_weights_{walkers}_walkers_{i+1}.npy')
        for CH in range(5):
            amp, xx = np.histogram(zmat[:, CH, 1]/ang2bohr, weights=weights[1, :], bins=25, range=(0.6, 1.8), density=True)
            bins = (xx[1:] + xx[:-1])/2.
            amp_all[1, i, CH, :] += amp

    for CH in range(5):
        axes[CH].plot(bins, np.mean(amp_all[1, :, CH, :]/np.linalg.norm(np.mean(amp_all[1, :, CH, :], axis=0)), axis=0), color='blue', label=r'$\phi$(x)^2 w/o imp samp')
        axes[CH].set_title(f'CH stretch {CH+1}')
        axes[CH].set_xlabel('rCH (Angstrom)')
        axes[CH].set_ylabel('Probability Density')
        axes[CH].plot(bins, np.mean(amp_all[0, :, CH, :]/np.linalg.norm(np.mean(amp_all[0, :, CH, :], axis=0)), axis=0), color='red', label=r'$\phi$(x)^2 w/ imp samp')
        axes[CH].set_title(f'CH stretch {CH + 1}')
        axes[CH].set_xlabel('rCH (Angstrom)')
        axes[CH].set_ylabel('Probability Density')

        # axes[CH].plot(bins, np.mean(amp_all[1, :, CH, :]/np.linalg.norm(np.mean(amp_all[1, :, CH, :], axis=0)), axis=0), color='red', label=r'$\phi$(x) biased')
        # axes[CH].set_title(f'CH stretch {CH + 1}')
        # axes[CH].set_xlabel('rCH (Angstrom)')
        # axes[CH].set_ylabel('Probability Density')
        axes[CH].legend()

    plt.tight_layout()
    fig.savefig('Comparing_f_different_DMC_alpha_61.png')
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
        if CH is not 0:
            psi = interpolate.splev(bins, interp, der=0)
            axes[CH].plot(bins, amp * psi)
        else:
            axes[CH].plot(bins, amp)
        axes[CH].set_title(f'CH stretch {CH+1}')
        axes[CH].set_xlabel('rCH (Angstrom)')
        axes[CH].set_ylabel('Probability Density')
    plt.tight_layout()
    fig.savefig('Biased_DMC_wvfns.png')
    plt.close(fig)



regular_wvfns(5000)



