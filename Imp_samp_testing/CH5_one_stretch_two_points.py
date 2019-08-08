import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

N_0 = 10000
# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_CH = (m_C*m_H)/(m_H+m_C)
m_CH5 = ((m_C + m_H*4)*m_H)/(m_H*5 + m_C)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# Starting orientation of walkers
coords_initial = np.array([4.000000000000000, 6.00000000000000])

ch_stretch = 4
Psi_t = np.load(f'GSW_min_CH_{ch_stretch+1}.npy')
interp = interpolate.splrep(np.linspace(1, 4, num=500), Psi_t, s=0)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.array([coords_initial]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


def psi_t(dist):
    return interpolate.splev(dist, interp, der=0)


def potential(Psi, CH):
    return interpolate.splev((Psi.coords[:, 1]-Psi.coords[:, 0]), CH, der=0)


def drdx(coords, dist):
    chain = np.zeros((N_0, 2))
    chain[:, 0] += ((coords[:, 0]-coords[:, 1])/dist)
    chain[:, 1] += ((coords[:, 1]-coords[:, 0])/dist)
    return chain


def drdx2(coords, dist):
    chain = np.zeros((N_0, 2))
    chain[:, 0] += (1./dist - (coords[:, 0]-coords[:, 1])**2/dist**3)
    chain[:, 1] += (1./dist - (coords[:, 1]-coords[:, 0])**2/dist**3)
    return chain


def E_loc(Psi):
    dist = Psi.coords[:, 1] - Psi.coords[:, 0]
    psi = psi_t(dist)
    dr1 = drdx(Psi.coords, dist)
    dr2 = drdx2(Psi.coords, dist)

    der1 = interpolate.splev(dist, interp, der=1)/psi
    der2 = interpolate.splev(dist, interp, der=2)/psi

    masses = np.array([1./m_C, 1./m_H])

    kin1 = np.tensordot(masses, dr1**2, axes=([0], [1]))
    kin1 = der2*kin1
    kin2 = np.tensordot(masses, dr2, axes=([0], [1]))
    kin2 = der1*kin2
    kin = kin1 + kin2

    return -1. / 2. * kin


Psi = Walkers(N_0)
Psi.coords[:, 1] = np.linspace(0.8, 1.4, num=N_0)*ang2bohr + 4.
pot = interpolate.splrep(np.linspace(1, 4, num=500), np.load('Potential_CH_stretch5.npy'), s=0)
Psi.V = potential(Psi, pot)
Psi.El = E_loc(Psi)
plt.plot((Psi.coords[:, 1]-Psi.coords[:, 0])/ang2bohr, Psi.V*har2wave, label='Potential')
plt.plot((Psi.coords[:, 1]-Psi.coords[:, 0])/ang2bohr, Psi.El*har2wave + Psi.V*har2wave, label='Local Energy')
plt.legend()
# plt.ylim(0, 22000)
plt.savefig('Testing_two_point_imp_samp.png')