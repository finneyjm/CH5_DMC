import numpy as np
import Timing_p3 as tm

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6
ang2bohr = (1.e-10)/(5.291772106712e-11)


def potential(x):
    bh = 4000./har2wave
    sp = 5.
    A = bh * (8./sp**2)
    B = bh * (16/sp**4)
    V = np.diag(bh - A * x ** 2 + B * x ** 4)
    return V, np.diag(V)


def Kinetic_Calc(grid):
    a = grid[0]
    b = grid[-1]
    N = len(grid)
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


def run():
    g = np.linspace(-10, 10, num=1000)
    V, vish = potential(g)
    T = Kinetic_Calc(g)
    En, Eigv = Energy(T, V)
    for i in range(10):
        print(En[i]*har2wave)
    import matplotlib.pyplot as plt
    plt.plot(g, -Eigv[:, 0]*har2wave)
    plt.plot(g, -Eigv[:, 1]*har2wave)
    plt.plot(g, vish*har2wave)
    plt.xlim(-5, 5)
    plt.ylim(-50000, 50000)
    plt.show()
    return V


run()
