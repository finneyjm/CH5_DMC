import numpy as np

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0106 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_red = (m_C*m_H)/(m_C+m_H)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
# harm_freq = 2273.2753549448880/har2wave
harm_freq = 3600/har2wave
# re = 2.2625250501002006  # Minimum bond length in Bohr
re=0.
omega = 3600./har2wave

wexe = 500./har2wave
De = omega**2/4/wexe
# De = 0.1897
# De = 0.037
# De = 0.0147
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
# m_red = 918.59073
# harm_freq = 0.0200534
# re = 1.40112
# sigmaOH = np.sqrt(dtau/m_red)

mw = m_red * omega
A = np.sqrt(omega**2 * m_red/(2*De))


def Potential(grid):
    # V = np.zeros(len(grid))
    # for i in range(len(grid)):
    #     if grid[i] < .1:
    #         V[i] = 1./2. * m_red * harm_freq**2 * (grid[i] - re)**2
    #     else:
    #         V[i] = 500000000000.
    # return np.diag(V)
    return np.diag(De * (1. - np.exp(-A * grid)) ** 2)
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
    grid = np.linspace(-1, 1., 2000)
    V = Potential(grid)
    T = Kinetic_Calc(grid)
    En, Eig = Energy(T, V)
    print(En[0]*har2wave)
    print(En[1]*har2wave)
    print((En[1]-En[0])*har2wave)
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    # np.save('Harmonic_gsw_CH_2', np.vstack((grid, Eig[:, 0])))
    return grid, Eig, np.diag(V)


import matplotlib.pyplot as plt
g, wvfn, harm_pot = run()
np.save('Anharmonic_trial_wvfn_ground_150_wvnum', np.vstack((g, wvfn[:, 0])))
np.save('Anharmonic_trial_wvfn_150_wvnum', np.vstack((g, -wvfn[:, 1])))
print(np.dot(wvfn[:, 1], g**2*wvfn[:, 0]))
print(np.dot(wvfn[:, 0], g*wvfn[:, 1]))
print(np.dot(wvfn[:, 1], g**2*wvfn[:, 1]))
x = g-0.039
x=g

a = (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * x ** 2)) * (2 * mw) ** (1 / 2) * x
ma = np.max(a)
a /= (ma/np.max(wvfn[:, 1]))

# plt.plot(g/ang2bohr, 0.5*m_red*(3600/har2wave)**2*g**2*har2wave, label='Harmonic Potential Energy')

# plt.plot(g/ang2bohr, -a*50000+5400, label='Harmonic Excited State')

# plt.plot(g, wvfn[:, 0]**2)
# plt.plot(g/ang2bohr, wvfn[:, 0]*50000 + 3601, label='Anharmonic Excited State')
# plt.plot(g/ang2bohr, harm_pot*har2wave, label='Anharmonic Potential Energy')
# plt.legend(loc='upper center')
# plt.xlim(-0.5, 0.5)
# plt.ylim(0, 20000)
# plt.xlabel(r'q ($\rm\AA$)', fontsize=14)
# plt.ylabel(r'Energy cm$^{-1}$', fontsize=14)
# plt.tight_layout()
# plt.show()






# print(g[210])
# g = g/ang2bohr
# true_pot = np.load('Potential_CH_stretch2.npy')
# print(np.argmin(true_pot))
# true_wvfn = np.load('GSW_min_CH_2.npy')
# print(np.argmax(true_wvfn))
# print(np.argmax(wvfn))
# print(g[210])
# fig, axes = plt.subplots(2, 1)
# axes[0].plot(g, true_pot*har2wave, label='True Potential')
# axes[0].plot(g, harm_pot*har2wave, label='Harmonic Potential')
# axes[1].plot(np.linspace(0.4, 6., 5000)/ang2bohr, true_wvfn/np.max(true_wvfn), label='True wvfn')
# axes[1].plot(g, wvfn/np.max(wvfn), label='Harmonic wvfn')
# axes[0].legend()
# axes[1].legend()
# axes[0].set_ylim(0, 20000)
# axes[0].set_xlabel('rCH (Angstrom)')
# axes[1].set_xlabel('rCH (Angstrom)')
# axes[0].set_ylabel('Energy (cm^-1)')
# axes[1].set_ylabel('Probability Amplitude')
# plt.tight_layout()
# # fig.savefig('Harmonic_approximation_of_CH_stretch.png')
# plt.show()
# plt.close()

