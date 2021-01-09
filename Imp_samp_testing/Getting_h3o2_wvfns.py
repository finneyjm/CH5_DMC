import numpy as np
from ProtWaterPES import *
from Coordinerds.CoordinateSystems import *

struct = np.array([
    [2.06095307, 0.05378083, 0.],
    [0., 0., 0.],
    [-0.32643038, -1.70972841, 0.52193868],
    [4.70153912, 0., 0.],
    [5.20071798, 0.80543847, 1.55595785]
])

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
m_red_D = (m_O*m_D)/(m_O+m_D)
m_red_sp = 1/(1/m_H + 1/(2*m_O))


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def asym_grid(coords, r1, a):
    coords = np.array([coords]*1)
    coords = coords[:, (1, 3, 0, 2, 4)]
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0], [3, 0, 1, 2],
                                                                                   [4, 1, 0, 2]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).reshape((N, 4, 6))
    zmat[:, 2, 1] = r1
    zmat[:, 3, 1] = r1 - a
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    coords = new_coords[:, (2, 0, 3, 1, 4)]
    return coords


def shared_prot_grid(coords, sp):
    coords = np.array([coords] * len(sp))
    mid = (coords[:, 3, 0] - coords[:, 1, 0])/2
    coords[:, 0, 0] = mid-sp
    return coords



def pot(coords):
    pot = get_pot(coords)
    return np.diag(pot)


def Kinetic_Calc(grid, red_m):
    N = len(grid)
    a = grid[0]
    b = grid[-1]
    coeff = (1./((2.*red_m)/(((float(N)-1.)/(b-a))**2)))

    Tii = np.zeros(N)

    Tii += coeff*((np.pi**2.)/3.)
    T_initial = np.diag(Tii)
    for i in range(1, N):
        for j in range(i):
            T_initial[i, j] = coeff*((-1.)**(i-j))*(2./((i-j)**2))
    T_final = T_initial + T_initial.T - np.diag(Tii)
    return T_final


def Energy(T, V):
    H = (T + V)
    En, Eigv = np.linalg.eigh(H)
    ind = np.argsort(En)
    En = En[ind]
    Eigv = Eigv[:, ind]
    return En, Eigv


def run(mass, stretch, grid, r1=None):
    if stretch == 'asymmetric':
        if r1 is None:
            r1 = np.linspace(0.5, 3.0, len(grid))
        coords = asym_grid(struct, r1, grid)
    elif stretch == 'shared proton':
        coords = shared_prot_grid(struct, grid)
    V = pot(coords)
    T = Kinetic_Calc(grid, mass)
    En, Eig = Energy(T, V)
    print(En[0] * har2wave)
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    print((En[1] - En[0]) * har2wave)
    return En, Eig, np.diag(V)


grid1 = np.linspace(-2.5, 2.5, 1000)
grid2 = np.linspace(-1.5, 1.5, 1000)
# en1, eig1, V1 = run(m_red, 'asymmetric', grid1)
# en2, eig2, V2 = run(m_red_sp, 'shared proton', grid2)
import matplotlib.pyplot as plt
# np.save('asymmetric_energies.npy', en1)
# np.save('asymmetric_wvfns.npy', eig1)
# np.save('asymmetric_pot.npy', V1)
# np.save('shared_prot_energies.npy', en2)
# np.save('shared_prot_wvfns.npy', eig2)
# np.save('shared_prot_pot.npy', V2)
#
en1 = np.load('asymmetric_energies.npy')
eig1 = np.load('asymmetric_wvfns.npy')
V1 = np.load('asymmetric_pot.npy')
en2 = np.load('shared_prot_energies.npy')
eig2 = np.load('shared_prot_wvfns.npy')
V2 = np.load('shared_prot_pot.npy')
plt.plot(grid1, eig1[:, 0]*24000 + en1[0]*har2wave)
plt.plot(grid1, eig1[:, 1]*24000 + en1[1]*har2wave)
plt.plot(grid1, V1*har2wave)
plt.ylim(0, 10000)
plt.xlim(-0.8, 0.8)
plt.show()
plt.plot(grid2, eig2[:, 0]*5000 + en2[0]*har2wave)
plt.plot(grid2, eig2[:, 1]*5000 + en2[1]*har2wave)
plt.plot(grid2, V2*har2wave)
plt.ylim(0, 2000)
plt.xlim(-1, 1)
plt.show()
