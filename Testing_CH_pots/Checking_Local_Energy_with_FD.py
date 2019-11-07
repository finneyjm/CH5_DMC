import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


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
coords_initial = np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])

ch_stretch = 4
Psi_t = np.load(f'GSW_min_CH_{ch_stretch+1}.npy')
interp = interpolate.splrep(np.linspace(0.4, 6, num=5000), Psi_t, s=0)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, acc):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.array([coords_initial]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.psi_t = np.zeros((walkers, 6, acc))


def psi_t(dist):
    return interpolate.splev(dist, interp, der=0)


def distance(coords):
    dist = np.zeros((len(coords[:, 0, 0]), 3))
    for i in range(3):
        dist[:, i] += coords[:, 1, i] - coords[:, 0, i]
    return np.linalg.norm(dist, axis=1)


def potential(Psi, CH):
    return interpolate.splev(distance(Psi.coords), CH, der=0)


def sec_dir_matrix(acc):
    if acc is 3:
        sec_der = np.diag(np.zeros(N_0) - 2.0) + np.diag(np.zeros(N_0-1) + 1., 1) + np.diag(np.zeros(N_0-1) + 1., -1)
    elif acc is 5:
        sec_der = np.diag(np.zeros(N_0) - 5./2.) + np.diag(np.zeros(N_0-1) + 4./3., 1) + \
                  np.diag(np.zeros(N_0-2) - 1./12., 2)
        sec_der = sec_der + sec_der.T - np.diag(np.zeros(N_0) - 5./2.)
    elif acc is 7:
        sec_der = np.diag(np.zeros(N_0) - 49./18.) + np.diag(np.zeros(N_0-1) + 3./2., 1) + \
                  np.diag(np.zeros(N_0-2) - 3./20., 2) + np.diag(np.zeros(N_0-3) + 1./90., 3)
        sec_der = sec_der + sec_der.T - np.diag(np.zeros(N_0) - 49./18.)
    elif acc is 9:
        sec_der = np.diag(np.zeros(N_0) - 205./72.) + np.diag(np.zeros(N_0-1) + 8./5., 1) + \
                  np.diag(np.zeros(N_0-2) - 1./5., 2) + np.diag(np.zeros(N_0-3) + 8./315., 3) + \
                  np.diag(np.zeros(N_0-4) - 1./560., 4)
        sec_der = sec_der + sec_der.T - np.diag(np.zeros(N_0) - 205./72.)
    else:
        print('you crazy')
        sec_der = 0
    return sec_der


def sec_der_indices(acc):
    if acc is 3:
        index = [1., -2., 1.]
    elif acc is 5:
        index = [-1./12, 4./3., -5./2., 4./3., -1./12]
    elif acc is 7:
        index = [1./90., 3./20., 3./2., -49./18., 3./2., 3./20.,  1./90.]
    elif acc is 9:
        index = [-1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.]
    else:
        print('you crazy')
        index = 0
    return index


def local_kinetic(Psi, acc, dx):
    index = sec_der_indices(acc)
    coords = np.array(Psi.coords)
    for i in range(6):
        for j in range(acc):
            if i < 3:
                if j < acc//2:
                    coords[:, 0, i] -= (dx * float(acc//2 - j))
                    Psi.psi_t[:, i, j] += (index[j]*psi_t(distance(coords)))
                    coords = np.array(Psi.coords)
                elif j > acc//2:
                    coords[:, 0, i] += (dx * float(j - acc//2))
                    Psi.psi_t[:, i, j] += (index[j]*psi_t(distance(coords)))
                    coords = np.array(Psi.coords)
                else:
                    Psi.psi_t[:, i, j] += (index[j] * psi_t(distance(coords)))
            else:
                if j < acc//2:
                    coords[:, 1, i-3] -= (dx * float(acc//2 - j))
                    Psi.psi_t[:, i, j] += (index[j]*psi_t(distance(coords)))
                    coords = np.array(Psi.coords)
                elif j > acc//2:
                    coords[:, 1, i-3] += (dx * float(j - acc//2))
                    Psi.psi_t[:, i, j] += (index[j]*psi_t(distance(coords)))
                    coords = np.array(Psi.coords)
                else:
                    Psi.psi_t[:, i, j] += (index[j] * psi_t(distance(coords)))

    masses = np.array([[1./m_C]*3, [1./m_H]*3]).flatten()

    psi = psi_t(distance(Psi.coords))
    sec_dir = np.sum(Psi.psi_t, axis=2)
    sec_dir *= (dx**2)**-1
    kin = np.tensordot(masses, sec_dir, axes=(0, 1))/psi

    return -1./2.*kin


Psi = Walkers(N_0, 9)
Psi.coords[:, 1, :] = np.array([np.linspace(0.46188, 0.80829, num=N_0)*ang2bohr + 4.]*3).T
pot = interpolate.splrep(np.linspace(1, 4, num=500), np.load('Potential_CH_stretch5.npy'), s=0)
Psi.V = potential(Psi, pot)
loc_kin = local_kinetic(Psi, 9, 0.0001)
dist = distance(Psi.coords)/ang2bohr
plt.plot(dist, Psi.V*har2wave, label='Potential')
plt.plot(dist, loc_kin*har2wave + Psi.V*har2wave, label='Local Energy')
plt.plot(dist, loc_kin*har2wave, label='Kinetic')
plt.xlabel('rCH (Angstrom)')
plt.ylabel('Energy (cm^-1)')
plt.ylim(-2500, 2500)
plt.legend()
plt.show()
plt.close()
# plt.savefig('Testing_imp_samp_FD.png')
