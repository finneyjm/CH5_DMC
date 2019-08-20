from Coordinerds.CoordinateSystems import *
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
coords_initial = np.array([[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]])

ch_stretch = 4
Psi_t = np.load(f'GSW_min_CH_{ch_stretch+1}.npy')
interp = interpolate.splrep(np.linspace(1, 4, num=500), Psi_t, s=0)
order = [[0, 0, 0, 0], [1, 0, 0, 0]]


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


def distance(coords):
    dist = np.zeros((len(coords[:, 0, 0]), 3))
    for i in range(3):
        dist[:, i] += coords[:, 1, i] - coords[:, 0, i]
    return np.linalg.norm(dist, axis=1)


def potential(Psi, CH):
    return interpolate.splev(distance(Psi.coords), CH, der=0)


def drdx(coords, dist):
    chain = np.zeros((N_0, 2, 3))
    for xyz in range(3):
        chain[:, 0, xyz] += ((coords[:, 0, xyz]-coords[:, 1, xyz])/dist)
        chain[:, 1, xyz] += ((coords[:, 1, xyz]-coords[:, 0, xyz])/dist)
    return chain


def drdx2(coords, dist):
    chain = np.zeros((N_0, 2, 3))
    for xyz in range(3):
        chain[:, 0, xyz] += (1./dist - (coords[:, 0, xyz]-coords[:, 1, xyz])**2/dist**3)
        chain[:, 1, xyz] += (1./dist - (coords[:, 1, xyz]-coords[:, 0, xyz])**2/dist**3)
    return chain


def E_loc(Psi):
    dist = distance(Psi.coords)
    psi = psi_t(dist)
    dr1 = drdx(Psi.coords, dist)
    dr2 = drdx2(Psi.coords, dist)

    der1 = interpolate.splev(dist, interp, der=1)/psi
    der2 = interpolate.splev(dist, interp, der=2)/psi

    masses = np.array([[1./m_C]*3, [1./m_H]*3])

    kin1 = np.tensordot(masses, dr1**2, axes=([0, 1], [1, 2]))
    kin1 = der2*kin1
    kin56468 = np.tensordot(masses, dr2, axes=([0, 1], [1, 2]))
    kin2 = der1*kin56468
    kin = kin1 + kin2

    extra_term = der1*2/dist*(1./m_C + 1./m_H)

    return -1. / 2. * kin, dist, -1./2. * kin1, -1./2. * kin2, -1./2.*extra_term


Psi = Walkers(N_0)
Psi.coords[:, 1, :] = np.array([np.linspace(0.46188, 0.80829, num=N_0)*ang2bohr + 4.]*3).T
asdf = distance(Psi.coords)/ang2bohr
pot = interpolate.splrep(np.linspace(1, 4, num=500), np.load('Potential_CH_stretch5.npy'), s=0)
Psi.V = potential(Psi, pot)
Psi.El, dist, kin1, kin2, extra = E_loc(Psi)
plt.plot(dist/ang2bohr, Psi.V*har2wave, label='Potential')
# plt.plot(dist/ang2bohr, Psi.El*har2wave + Psi.V*har2wave, label='Local Energy')
# plt.plot(dist/ang2bohr, kin1*har2wave, label='Local Energy from Kin1')
# plt.plot(dist/ang2bohr, kin2*har2wave, label='Local Energy from Kin2')
# plt.plot(dist/ang2bohr, extra*har2wave, label='Extra Term')
plt.plot(dist/ang2bohr, Psi.El*har2wave+Psi.V*har2wave, label='Local Energy')
# plt.plot(dist/ang2bohr, mass_drdx*har2wave, label='Mass weighted drdx')
# plt.plot(dist/ang2bohr, dpsidr*har2wave, label='dPsidr')
# plt.plot(dist/ang2bohr, -Psi.El*har2wave, label='Local Kinetic Energy')
plt.xlabel('rCH (Angstrom)')
plt.ylabel('Energy (cm^-1)')
plt.legend()
plt.ylim(0, 2500)
plt.savefig('Testing_6_point_imp_samp.png')

