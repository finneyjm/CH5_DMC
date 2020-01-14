import numpy as np

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# parameters for the potential and for the analytic wavefuntion
De = 0.02
omega = 3600./har2wave
mw = m_red * omega
A = np.sqrt(omega**2 * m_red/(2*De))


def Potential(grid, anharm):
    if anharm is True:
        return De*(1. - np.exp(-A*grid))**2
    else:
        return 1./2.*m_red*omega**2*grid**2


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


def run(anharm=True):
    grid = np.linspace(-1.5, 1.5, 1000)
    V = np.diag(Potential(grid, anharm))
    T = Kinetic_Calc(grid)
    En, Eig = Energy(T, V)
    print(En[0]*har2wave)
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    # np.save('Harmonic_gsw_CH_2', np.vstack((grid, Eig[:, 0])))
    return grid, En[0], Eig[:, 0], np.diag(V)


import matplotlib.pyplot as plt
g, en, wvfn, anharm_pot = run()
g, en2, wvfn2, harm_pot = run(anharm=False)
g = g/ang2bohr

# plt.plot(g, harm_pot*har2wave)
plt.plot(g, anharm_pot*har2wave, color='red')
plt.plot(g, (wvfn*50000+(en*har2wave)), color='green')
plt.ylim(0, 10000)
plt.xlabel(r'$\Delta$r ($\AA$)', fontsize=16)
plt.ylabel(r'Energy (cm$^{-1}$)', fontsize=16)
plt.xlim(-0.4, 0.8)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()
plt.close()
















