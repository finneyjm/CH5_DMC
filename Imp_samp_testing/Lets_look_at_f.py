import numpy as np
import matplotlib.pyplot as plt

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_CH = (m_C*m_H)/(m_H+m_C)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
grid = np.linspace(1, 4, num=500)/ang2bohr


class Walkers(object):
    wvfn = 0

    def __init__(self, wvfn):
        self.dist = wvfn[0, :]
        self.weights = wvfn[1, :]
        self.d = wvfn[2, :]


DMC_psi_1 = Walkers(np.load('Wavefunction_CH_stretch_1.npy'))
DMC_psi_3 = Walkers(np.load('Wavefunction_CH_stretch_3.npy'))
DMC_psi_5 = Walkers(np.load('Wavefunction_CH_stretch_5.npy'))
DMC_psi_coords = np.array([DMC_psi_1.dist, DMC_psi_3.dist, DMC_psi_5.dist])
DMC_psi_weights = np.array([DMC_psi_1.weights, DMC_psi_3.weights, DMC_psi_5.weights])
DMC_psi_d = np.array([DMC_psi_1.d, DMC_psi_3.d, DMC_psi_5.d])

DVR_psi = np.zeros((3, 500))
DVR_psi[0, :] = np.load('GSW_min_CH_1.npy')
DVR_psi[1, :] = np.load('GSW_min_CH_3.npy')
DVR_psi[2, :] = np.load('GSW_min_CH_5.npy')

Switch_fn = np.load('Switch_min_wvfn_speed_5.0.npy')

CH = [1, 3, 5]


def lets_see_this_f():
    for i in range(3):
        fig, axes = plt.subplots()
        DVR_f = DVR_psi[i, :] * Switch_fn[1, :]
        amp, xx = np.histogram(DMC_psi_coords[i, :]/ang2bohr, weights=DMC_psi_weights[i, :], bins=75, range=(grid[0], grid[-1]), density=True)
        bins = (xx[1:] + xx[:-1])/2.
        axes.plot(grid, DVR_f*300, label='f(x) from DVR')
        axes.plot(bins, amp, label='f(x) from DMC')
        axes.legend()
        axes.set_xlabel('rCH (Angstrom)')
        axes.set_ylabel('Probability Density')
        fig.savefig(f'Looking_at_f_CH_{CH[i]}.png')
        plt.close(fig)


lets_see_this_f()

