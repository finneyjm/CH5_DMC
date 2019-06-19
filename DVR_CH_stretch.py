import numpy as np
import matplotlib.pyplot as plt
from Coordinerds.CoordinateSystems import *
import CH5pot

coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                  [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                  [1.786540362044548, -1.386051328559878, 0.000000000000000],
                  [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                  [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                  [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0106 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_red = (m_C*m_H)/(m_C+m_H)
har2wave = 219474.6


def grid(a, b, N, CH):
    spacing = np.linspace(a, b, num=N)
    if CH == 1:
        new_coords = CoordinateSet(coords_initial, system=CartesianCoordinates3D)
        new_coords = new_coords.convert(ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
        g = np.array([new_coords]*N)
        g[:, 1, 0] += spacing
    if CH is not 1:
        sub = np.array([coords_initial[1]])
        coords_initial[1] = coords_initial[CH]
        coords_initial[CH] = sub
        new_coords = CoordinateSet(coords_initial, system=CartesianCoordinates3D)
        new_coords = new_coords.convert(ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
        g = np.array([new_coords] * N)
        g[:, 1, 0] = spacing
    return g


def Potential(grid, CH):
    V = CH5pot.mycalcpot(grid, len(grid[:, 0, 0]))
    V_final = np.diag(np.array(V))
    plt.plot(grid[:, 1, 0], np.diag(V_final)*har2wave, label='CH stretch %s' %CH)
    plt.legend()
    plt.xlabel('Bond Distance (Bohr)')
    plt.ylabel(r'Energy (cm${-1}$)')
    plt.savefig('Potential_CH_stretch%s.png' %CH)
    return V_final


def Kinetic_Calc(grid):
    a = grid[0, 1, 0]
    b = grid[-1, 1, 0]
    N = len(grid[:, 0, 0])
    coeff = (1./((2.*m_red)/(((float(N)-1.)/(b-a))**2)))

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


def run(CH):
    g = grid(1., 4., 50, CH)
    V = Potential(g, CH)
    T = Kinetic_Calc(g)
    En, Eig = Energy(T, V)
    print(En[0]*har2wave)
    np.save('CH_stretch_wavefunction%s' %CH, Eig[:, 0])
    # plt.plot(g[:, 1, 0], -Eig[:, 0]*har2wave + En[0]*har2wave, label='Ground State Wavefunction CH stretch %s' %CH)
    # plt.xlabel('Bond Distance (Bohr)')
    # plt.ylabel('Energy (Hartree)')
    # plt.legend()
    # plt.savefig('GSW%s.png' %CH)



for i in np.arange(1, 6):
    run(i)
