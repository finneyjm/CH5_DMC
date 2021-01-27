import numpy as np


me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
omega = 3600./har2wave

wexe = 150./har2wave
De = omega**2/4/wexe
omega = 3600/har2wave
mw = m_red * 3600/har2wave
A = np.sqrt(omega**2 * m_red/(2*De))


ground = np.load('ground_state_wvfn.npy')
excite = np.load('excited_state_wvfn.npy')


def Harmonic_wvfn(x, state):
    if state == 1:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (x-0.039) ** 2)) * (2 * mw) ** (1 / 2) * (x-0.039)
    else:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (x-0.039) ** 2))

term1_ov = np.zeros(5)
for i in range(5):
    frac = Harmonic_wvfn(ground[1, i], 1)/Harmonic_wvfn(ground[1, i], 0)
    term1_ov[i] = np.dot(ground[0, i], frac)/np.sum(ground[0, i])
term1_ov = np.average(term1_ov)

term2_ov = np.zeros(5)
for i in range(5):
    frac = Harmonic_wvfn(excite[1, i], 0)/Harmonic_wvfn(excite[1, i], 1)
    term2_ov[i] = np.dot(excite[0, i], frac)/np.sum(excite[0, i])
term2_ov = np.average(term2_ov)

grid = np.linspace(-1, 1, 2000)-0.039
term3_ov = np.dot(Harmonic_wvfn(grid, 1), Harmonic_wvfn(grid, 0))

print(f'All term overlap = {term1_ov + term2_ov - term3_ov}')
print(f'Term 1 + term 2 over 2 = {(term1_ov + term2_ov)/2}')
print(f'term 1 overlap = {term1_ov}')


term1 = np.zeros(5)
for i in range(5):
    frac = Harmonic_wvfn(ground[1, i], 1)/Harmonic_wvfn(ground[1, i], 0)
    term1[i] = np.dot(ground[0, i], frac*ground[1, i])/np.sum(ground[0, i])
term1_dip = np.average(term1)

term2 = np.zeros(5)
for i in range(5):
    frac = Harmonic_wvfn(excite[1, i], 0)/Harmonic_wvfn(excite[1, i], 1)
    term2[i] = np.dot(excite[0, i], frac*excite[1, i])/np.sum(excite[0, i])
term2_dip = np.average(term2)

grid = np.linspace(-1, 1, 2000)
term3_dip = np.dot(Harmonic_wvfn(grid, 0)*grid*Harmonic_wvfn(grid, 1), np.ones(2000))

print(f'All term dipole = {term1_dip + term2_dip - term3_dip}')
print(f'Term 1 + term 2 over 2 = {(term1_dip + term2_dip)/2}')
print(f'term 1 dipole = {term1_dip}')


term1 = np.zeros(5)
for i in range(5):
    frac = Harmonic_wvfn(ground[1, i], 1)/Harmonic_wvfn(ground[1, i], 0)
    term1[i] = np.dot(ground[0, i], frac*ground[1, i]**2)/np.sum(ground[0, i])
term1_dip = np.average(term1)

term2 = np.zeros(5)
for i in range(5):
    frac = Harmonic_wvfn(excite[1, i], 0)/Harmonic_wvfn(excite[1, i], 1)
    term2[i] = np.dot(excite[0, i], frac*excite[1, i]**2)/np.sum(excite[0, i])
term2_dip = np.average(term2)

grid = np.linspace(-1, 1, 2000)
term3_dip = np.dot(Harmonic_wvfn(grid, 0)*grid**2*Harmonic_wvfn(grid, 1), np.ones(2000))

print(f'All term q^2 = {term1_dip + term2_dip - term3_dip}')
print(f'Term 1 + term 2 over 2 = {(term1_dip + term2_dip)/2}')
print(f'term 1 q^2 = {term1_dip}')

term1 = np.zeros(5)
for i in range(5):
    frac = Harmonic_wvfn(ground[1, i], 1)/Harmonic_wvfn(ground[1, i], 0)
    term1[i] = np.dot(ground[0, i], frac**2*ground[1, i]**2)/np.sum(ground[0, i])
term1_dip = np.average(term1)

term2 = np.zeros(5)
for i in range(5):
    frac = Harmonic_wvfn(excite[1, i], 0)/Harmonic_wvfn(excite[1, i], 1)
    term2[i] = np.dot(excite[0, i], frac**2*excite[1, i]**2)/np.sum(excite[0, i])
term2_dip = np.average(term2)

grid = np.linspace(-1, 1, 2000)
term3_dip = np.dot(Harmonic_wvfn(grid, 1)*grid**2*Harmonic_wvfn(grid, 1), np.ones(2000))

print(f'All term q^2 = {term1_dip + term2_dip - term3_dip}')
print(f'Term 1 + term 2 over 2 = {(term1_dip + term2_dip)/2}')
print(f'term 1 q^2 = {term1_dip}')

import matplotlib.pyplot as plt

# amp, xx = np.histogram(excite[1, 4], weights=excite[0, 4], range=(-1, 1), density=True, bins=75)
# bin = (xx[1:] + xx[:-1]) / 2.
#
# plt.plot(bin, amp)
# plt.show()
