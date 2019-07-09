import numpy as np
import matplotlib.pyplot as plt

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6


def grid(a, b, N):
    return np.linspace(a, b, num=N)


def Potential(g, bh, spacing):
    bh = bh/har2wave
    A = bh * 8. / spacing ** 2
    B = bh * (4. / spacing ** 2) ** 2
    V = bh - A * g ** 2 + B * (g ** 4)
    plt.figure()
    plt.plot(g, V*har2wave)
    plt.xlabel('Arbitrary Grid')
    plt.ylabel('Energy (cm^-1)')
    plt.title('Double Well Potential')
    plt.ylim(0, 5000)
    plt.savefig('Double_well_pot_%s.png' %g[-1])
    return np.diag(V)


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
    g = grid(-2.5, 2.5, 1000)
    V = Potential(g, 1000., 2.)
    T = Kinetic_Calc(g)
    En, Eig = Energy(T, V)
    print(En[0]*har2wave)
    plt.figure()
    plt.plot(g, -Eig[:, 0])
    plt.ylabel('Probability Amplitude')
    plt.savefig('Double_well_pot_wavefn_%s.png' %g[-1])


run()








