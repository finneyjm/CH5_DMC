import numpy as np
import matplotlib.pyplot as plt


ang2bohr = (1.e-10)/(5.291772106712e-11)
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6
omega = 3600./har2wave


def grid(a, b, N):
    # a = a*ang2bohr
    # b = b*ang2bohr
    x = np.linspace(a, b, num=N)
    return x


def Potential_HO(grid):
    x = grid
    pot = (1./2.)*m_red*omega**2*(x**2)
    v_final = np.diag(pot)
    return v_final


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


def Plot_wavefn(En, Eigv, V, grid):
    Eigv += En
    Eigv = Eigv*har2wave  # for scale
    Eo = np.zeros(1000)
    Eo += (En[0])*har2wave
    y = np.diag(V)*har2wave
    x = grid/ang2bohr
    plt.plot(x, y, "r-", x, Eigv[:, 0], "b-", x, Eo, "k-")
    plt.xlabel("R (Angstroms)")
    plt.ylabel("Energy (cm$^{-1}$)")
    plt.show()
    plt.close()


def fir_sec_der(grid, gswvfn):
    # dx = (grid[-1] - grid[1])/float(len(grid))
    fir_der = np.gradient(gswvfn, grid)
    sec_der = np.gradient(fir_der, grid)
    return fir_der, sec_der


def run():
    g = grid(-5, 5, 10000)
    V = Potential_HO(g)
    T = Kinetic_Calc(g)
    En, Eig = Energy(T, V)
    Eig[:, 0] = Eig[:, 0]*-31.621289561335708
    Eig0f, Eig0s = fir_sec_der(g, Eig[:, 0])
    wvfn = np.zeros((4, len(g)))
    wvfn[0, :] = g
    wvfn[1, :] = Eig[:, 0]
    wvfn[2, :] = Eig0f
    wvfn[3, :] = Eig0s
    np.save('Ground_state_wavefunction_HO', wvfn)
    # print(En[0]*har2wave)
    return wvfn


run()


