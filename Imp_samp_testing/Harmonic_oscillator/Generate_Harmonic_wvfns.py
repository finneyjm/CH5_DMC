import numpy as np

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0106 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_red = (m_C*m_H)/(m_C+m_H)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
harm_freq = 2273.2753549448880/har2wave
re = 2.2625250501002006  # Minimum bond length in Bohr


def Potential(grid):
    return np.diag(1./2. * m_red * harm_freq**2 * (grid - re)**2)
    # return np.diag(np.load('Potential_CH_stretch2.npy'))


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
    grid = np.linspace(1., 4., 500)
    V = Potential(grid)
    T = Kinetic_Calc(grid)
    En, Eig = Energy(T, V)
    print(En[0]*har2wave)
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    np.save('Harmonic_gsw_CH_2', np.vstack((grid, Eig[:, 0])))
    return grid, Eig[:, 0], np.diag(V)


import matplotlib.pyplot as plt
g, wvfn, harm_pot = run()
print(g[210])
g = g/ang2bohr
true_pot = np.load('Potential_CH_stretch2.npy')
print(np.argmin(true_pot))
true_wvfn = np.load('GSW_min_CH_2.npy')
print(np.argmax(true_wvfn))
print(np.argmax(wvfn))
print(g[210])
fig, axes = plt.subplots(2, 1)
axes[0].plot(g, true_pot*har2wave, label='True Potential')
axes[0].plot(g, harm_pot*har2wave, label='Harmonic Potential')
axes[1].plot(np.linspace(0.4, 6., 5000)/ang2bohr, true_wvfn/np.max(true_wvfn), label='True wvfn')
axes[1].plot(g, wvfn/np.max(wvfn), label='Harmonic wvfn')
axes[0].legend()
axes[1].legend()
axes[0].set_ylim(0, 20000)
axes[0].set_xlabel('rCH (Angstrom)')
axes[1].set_xlabel('rCH (Angstrom)')
axes[0].set_ylabel('Energy (cm^-1)')
axes[1].set_ylabel('Probability Amplitude')
plt.tight_layout()
# fig.savefig('Harmonic_approximation_of_CH_stretch.png')
plt.show()
plt.close()

