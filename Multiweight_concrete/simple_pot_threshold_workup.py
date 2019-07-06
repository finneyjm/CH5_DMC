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
        self.walkers = np.linspace(0, len(weights[0, 0, :])-1, num=len(weights[0, 0, :]))
        self.weights = weights
        self.coords = coords
        self.d = d


def graphing():
    for i in range(5):
        Energy = load_observables('%s' % (0.1**(i+1)))
        print(np.mean(Energy[500:]*har2wave))
    # Energy = load_observables('%s' % (1./10000.))
    # print(np.mean(Energy[500:]*har2wave))


graphing()















