import numpy as np
from ProtWaterPES import *
from Coordinerds.CoordinateSystems import *

oxy_pos = 4.61607485e+00
new_pos = oxy_pos

struct = np.array([
    [4.61607485e+00/2, 0., 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [-4.65570340e-01, 1.67058080e+00, -5.46666468e-01],
    [4.61607485e+00, 0.00000000e+00, 0.00000000e+00],
    [5.12936209e+00, -8.18802009e-01, -1.54030505e+00]
])

struct = np.array([
    [0.000000000000000, 0.000000000000000, 0.000000000000000],
    [-2.303263755760085, 0.000000000000000, 0.000000000000000],
    [-2.720583162407882, 1.129745554266140, -1.363735721982301],
    [2.303263755760085, 0.000000000000000, 0.000000000000000],
    [2.720583162407882, 1.129745554266140, 1.363735721982301]
])
struct[:, 0] = struct[:, 0] + 2.303263755760085

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
m_red_OO = (m_O**2)/(2*m_O)


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


# def oo_grid(coords, Roo):
#     coords = np.array([coords] * 1)
#     coords = coords[:, (1, 3, 0, 2, 4)]
#     zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
#                                                                         ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
#                                                                                    [2, 0, 1, 0], [3, 0, 1, 2],
#                                                                                    [4, 1, 0, 2]])).coords
#     N = len(Roo)
#     zmat = np.array([zmat] * N).reshape((N, 4, 6))
#     zmat[:, 0, 1] = Roo
#     new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     coords = new_coords[:, (2, 0, 3, 1, 4)]
#     return coords

def oo_grid(coords, Roo):
    coords = np.array([coords] * len(Roo))
    equil_roo_roh_x = coords[0, 3, 0] - coords[0, 4, 0]

    coords[:, 3, 0] = Roo
    coords[:, 4, 0] = Roo - equil_roo_roh_x

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
    else:
        coords = oo_grid(struct, grid)
    V = pot(coords)
    T = Kinetic_Calc(grid, mass)
    En, Eig = Energy(T, V)
    print(En[0] * har2wave)
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    print((En[1] - En[0]) * har2wave)
    return En, Eig, np.diag(V)


grid1 = np.linspace(-2.5, 2.5, 1000)
grid2 = np.linspace(-1., 1., 1000)
grid3 = np.linspace(4.2, 5.2, 1000)
# en1, eig1, V1 = run(m_red, 'asymmetric', grid1)
# en2, eig2, V2 = run(m_red_sp, 'shared proton', grid2)
# en3, eig3, V3 = run(m_red_OO, 'oo', grid3)
import matplotlib.pyplot as plt
# np.save('asymmetric_energies.npy', en1)
# np.save('asymmetric_wvfns.npy', eig1)
# np.save('asymmetric_pot.npy', V1)
# np.save('shared_prot_energies.npy', en2)
# np.save('shared_prot_wvfns.npy', eig2)
# np.save('shared_prot_pot.npy', V2)
# np.save('oo_energies.npy', en3)
# np.save('oo_wvfns.npy', eig3)
# np.save('oo_pot.npy', V3)

en1 = np.load('asymmetric_energies.npy')
eig1 = np.load('asymmetric_wvfns.npy')
V = np.load('asymmetric_pot.npy')
dx = 5.0/1000
ind = np.argmin(V)
second_der = ((1/90*V[ind-3] - 3/20*V[ind-2] + 3/2*V[ind-1] - 49/18*V[ind] + 3/2*V[ind+1] - 3/20*V[ind+2] + 1/90*V[ind+3])/dx**2)
print(np.sqrt(second_der/m_red)*har2wave)
print(np.sqrt(second_der/m_red_D)*har2wave)
omega = np.sqrt(second_der/m_red)
omegaD = np.sqrt(second_der/m_red_D)
mw = m_red*omega


def Harmonic_wvfn(x, state):
    if state == 1:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (x) ** 2)) * (2 * mw) ** (1 / 2) * (x)
    else:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (x) ** 2))

# en2 = np.load('shared_prot_energies.npy')
# eig2 = np.load('shared_prot_wvfns.npy')
# V2 = np.load('shared_prot_pot.npy')
# plt.plot(1/np.sqrt(2)*grid1, eig1[:, 0]/np.max(eig1[:, 0])*2400 + en1[0]*har2wave, label=r'$\Phi_0$')
# plt.plot(1/np.sqrt(2)*grid1, eig1[:, 1]/np.max(eig1[:, 1])*2400 + en1[1]*har2wave, label=r'$\Phi_1$')
plt.plot(1/np.sqrt(2)*grid1, 1/2*m_red*omega**2*grid1**2*har2wave, label='Harmonic Potential')
# plt.plot(1/np.sqrt(2)*grid1, V*har2wave-np.min(V)*har2wave, label='Anharmonic Potential')
plt.plot(1/np.sqrt(2)*grid1, Harmonic_wvfn(grid1, 0)/np.max(Harmonic_wvfn(grid1, 0))*1500 + en1[0]*har2wave, label=r'$\Psi_0$')
# plt.plot(1/np.sqrt(2)*grid1, -1*Harmonic_wvfn(grid1, 1)/np.max(-1*Harmonic_wvfn(grid1, 1))*1500 + en1[1]*har2wave, label=r'$\Psi_0$')
plt.ylim(0, 10000)
plt.xlim(-0.8, 0.8)
# plt.xlabel(r'Asymmetric Stretch (Bohr)', fontsize=16)
# plt.ylabel(r'Energy cm$^{-1}$', fontsize=16)
# plt.legend()
plt.tight_layout()
plt.show()
# plt.plot(1/np.sqrt(2)*grid2/ang2bohr, eig2[:, 0]*5000 + en2[0]*har2wave)
# plt.plot(1/np.sqrt(2)*grid2/ang2bohr, eig2[:, 1]*5000 + en2[1]*har2wave)
# plt.plot(1/np.sqrt(2)*grid2/ang2bohr, V2*har2wave)
# plt.xlabel(r'XH $\rm\AA$')
# plt.ylim(0, 3000)
# # plt.xlim(-1, 1)
# plt.show()
# plt.plot(grid3/ang2bohr, eig3[:, 0]*4000 + en3[0]*har2wave)
# plt.plot(grid3/ang2bohr, eig3[:, 1]*4000 + en3[1]*har2wave)
# plt.plot(grid3/ang2bohr, V3*har2wave)
# plt.xlabel(r'R$_{\rmOO}$ $\rm\AA$')
# plt.ylim(0, 3000)
# plt.xlim(4.2/ang2bohr, 5.2/ang2bohr)
# plt.show()