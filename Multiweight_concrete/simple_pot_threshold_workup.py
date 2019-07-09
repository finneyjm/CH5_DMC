import numpy as np
import matplotlib.pyplot as plt


har2wave = 219474.6


def load_psi(which_one):
    Psi_coords = np.load("DMC_HO_Psi_pos_threshold_%s.npy" %which_one)
    Psi_d = np.load("DMC_HO_descendants_threshold_%s.npy" %which_one)
    Psi_weights = np.load("DMC_HO_Psi_weights_threshold_%s.npy" %which_one)
    return Psi_coords, Psi_d, Psi_weights


def load_observables(which_one):
    Energy = np.load("DMC_HO_Energy_threshold_%s.npy" %which_one)
    return Energy


class Walkers(object):

    def __init__(self, coords, d, weights):
        self.walkers = np.linspace(0, len(weights)-1, num=len(weights))
        self.weights = weights
        self.coords = coords
        self.d = d


def graphing():
    for i in range(5):
        energies = np.zeros(6)
        for j in range(6):
            Energy = load_observables('%s' % (0.1**(i+1)) + '_job_%s' % (j+1))
            energies[j] += np.mean(Energy[500:]*har2wave)
            fig, axes = plt.subplots()
            coords, d, weights = load_psi('%s' % (0.1**(i+1)))
            Psi = Walkers(coords, d, weights)
            amp, xx = np.histogram(Psi.coords, weights=Psi.weights, bins=50, range=(-2.5, 2.5), density=True)
            bins = (xx[1:] + xx[:-1])/2.
            axes.plot(bins, amp)
            axes.set_ylabel('Probability Amplitude')
            fig.savefig('Psi_for_threshold_{0}{1}.png'.format((0.1**(i+1)), j+1))
            plt.close(fig)
        std = np.std(energies)
        print(str(np.mean(energies)) + ' +/- ' + str(std))
    # Energy = load_observables('%s' % (1./10000.))
    # print(np.mean(Energy[500:]*har2wave))


graphing()















