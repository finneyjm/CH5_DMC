import numpy as np
import matplotlib.pyplot as plt

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_C*m_H)/(m_H+m_C)
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
    g = grid(-5., 5., 1000)
    V, V0 = Potential(g, 1000., 2., Ecut)
    T = Kinetic_Calc(g)
    En, Eig = Energy(T, V)
    E_correction = np.dot((Eig[:, 0]*Eig[:, 0]), np.diag(V0-V))
    print(E_correction*har2wave)
    print(En[0]*har2wave + E_correction * har2wave)
    return En, Eig, g, np.diag(V), En[0]*har2wave + E_correction*har2wave


def plotting(Ecut, scaling):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig2, axes2 = plt.subplots()
    for j, cut in enumerate(Ecut):
        En, Eig, g, V, correct_en = run(cut)
        if Eig[500, 0] <= 0.:
            Eig[:, 0] *= -1.
        axes[0].plot(g, V*har2wave, color=f'C{j}', label=f'Ecut = {cut}')
        axes[0].plot(g, (Eig[:, 0]*scaling + En[0]*har2wave), color=f'C{j}')
        axes[1].scatter(cut, En[0]*har2wave, color=f'C{j}')
        axes2.scatter(cut, correct_en, color=f'C{j}')
        axes[0].legend()
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Energy (cm^-1)')
        axes[0].set_xlim(-2.0, 2.0)
        axes[0].set_ylim(0, 5000)
        # axes[1].legend()
        axes[1].set_xlabel('Ecut (cm^-1)')
        axes[1].set_ylabel('Energy (cm^-1)')
        # axes[1].set_xlim(-1.75, 1.75)
        axes[1].set_ylim(450, 1100)
        # axes2.legend()
        axes2.set_xlabel('Ecut (cm^-1)')
        axes2.set_ylabel('Corrected Energy (cm^-1)')
        # axes2.set_xlim(-1.75, 1.75)
        axes2.set_ylim(450, 1100)
        plt.tight_layout()
        fig.savefig(f'Simple_concrete_filling_Ecut_{cut}.png')
    fig2.savefig(f'Simple_concrete_filling_corrected.png')


# plotting([0, 100, 300, 500, 700, 1000], 5000.)


# cut = 100
# En0, Eig0, g = run(0)
# En10, Eig10, g10 = run(cut)
# overlap = np.linalg.norm(Eig0[:, 0]*Eig10[:, 0], ord=1)
# print(overlap)
# plt.figure()
# plt.plot(g, -Eig0[:, 0], label='No cut')
# plt.plot(g, -Eig10[:, 0], label='Ecut=%s cm^-1' %cut)
# plt.xlabel('x')
# plt.ylabel('Probability Density')
# plt.title('Overlap of Wavefunctions %s' % overlap)
# plt.legend()
# plt.savefig('GSW_test_small_cut_overlap%s.png' %cut)
# en, eig, g = run(500)
# print(np.dot((eig[:, 0]*eig[:, 0]), g))
run(0)






