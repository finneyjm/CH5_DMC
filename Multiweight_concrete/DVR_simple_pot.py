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


def Potential(g, bh, spacing, Ecut):
    bh = bh/har2wave
    Ecut = Ecut/har2wave
    A = bh * 8. / spacing ** 2
    B = bh * (4. / spacing ** 2) ** 2
    V0 = bh - A * g ** 2 + B * (g ** 4)
    ind = np.argwhere(V0 < Ecut)
    V = np.array(V0)
    V[ind] = Ecut
    return np.diag(V), np.diag(V0)


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


def run(Ecut):
    g = grid(-2.5, 2.5, 1000)
    V, V0 = Potential(g, 1000., 2., Ecut)
    T = Kinetic_Calc(g)
    En, Eig = Energy(T, V)
    E_correction = np.dot((Eig[:, 0]*Eig[:, 0]), np.diag(V0-V))
    print(En[0]*har2wave + E_correction * har2wave)
    return En, Eig, g


cut = 10
En0, Eig0, g = run(0)
En10, Eig10, g10 = run(cut)
overlap = np.linalg.norm(Eig0[:, 0]*Eig10[:, 0], ord=1)
print(overlap)
plt.figure()
plt.plot(g, -Eig0[:, 0], label='No cut')
plt.plot(g, -Eig10[:, 0], label='Ecut=%s cm^-1' %cut)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Overlap of Wavefunctions %s' % overlap)
plt.legend()
plt.savefig('GSW_test_small_cut_overlap%s.png' %cut)







