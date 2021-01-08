import numpy as np
wvfn = np.load('getting_coords.npy')

d = np.load('excited_d_values.npy')
g = np.load('ground_d_values.npy')
excite_d = np.average(np.load('excited_d_values.npy'), axis=0)
ground_d = np.average(np.load('ground_d_values.npy'), axis=0)
std_excite = np.std(np.load('excited_d_values.npy'), axis=0)
std_ground = np.std(np.load('ground_d_values.npy'), axis=0)
coords = wvfn[0]

# find = np.argwhere(coords < 0.039)
# excite_d[find] *= -1

# amp1, xx1 = np.histogram(coords, weights=excite_d, range=(-0.5, 0.5), bins=150)
# bins1 = (xx1[1:] + xx1[:-1]) / 2.
import matplotlib.pyplot as plt
#
# plt.scatter(excite_d, std_excite/excite_d*100, label='excited')
# plt.scatter(ground_d, std_ground/ground_d*100, label='ground')
# # plt.legend()
# # plt.show()
#
# plt.plot(bins1, amp1/np.max(amp1), label='excited descendant weights')
#
# amp2, xx2 = np.histogram(coords, weights=ground_d, range=(-0.5, 0.5), bins=150)
# bins2 = (xx2[1:] + xx2[:-1]) / 2.
#
# plt.plot(bins2, amp2/np.max(amp2), label='ground descendant weights')

amp3, xx3 = np.histogram(coords, weights=wvfn[1], range=(-0.5, 0.5), bins=75)
bins3 = (xx3[1:] + xx3[:-1]) / 2.
# plt.plot(bins3, amp3/np.max(amp3), label='ground state initial weights')


plt.xlabel(r'$\Delta \rmr_{OH}$')

from scipy import interpolate

trial_wvfn = np.load('Harmonic_oscillator/Anharmonic_trial_wvfn_150_wvnum.npy')
interp1 = interpolate.splrep(trial_wvfn[0], trial_wvfn[1], s=0)
psi_1 = interpolate.splev(bins3, interp1, der=0)
trial_wvfn = np.load('Harmonic_oscillator/Anharmonic_trial_wvfn_ground_150_wvnum.npy')
interp0 = interpolate.splrep(trial_wvfn[0], trial_wvfn[1], s=0)
psi2 = interpolate.splev(bins3, interp0, der=0)
# plt.plot(bins3, psi2**2/np.linal.norm(psi2**2), label=r'$\Phi_1^2$')
#
me = 9.10938356e-31
Avo_num = 6.0221367e23
har2wave = 219474.6
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
omega = 3600./har2wave
mw = m_red * omega

a = (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (bins3-0.039) ** 2))

b = (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (bins3-0.039) ** 2)) * (2 * mw) ** (1 / 2) * (bins3-0.039)

c = amp3/np.linalg.norm(amp3)*b/a

# plt.plot(bins3, amp3/np.linalg.norm(amp3), label=r'f($\rm\Delta r_{OH}$)')

plt.plot(bins3, c/np.linalg.norm(c), label=r'f($\rm\Delta r_{OH}$)*$\Psi_1/\Psi_0$')
plt.plot(bins3, -psi2*psi_1/(np.linalg.norm(psi2*psi_1)), label=r'$\Phi_1\Phi_0$')
plt.plot(bins3, psi2*b/np.linalg.norm(psi2*b), label=r'$\Phi_0\Psi_1$')
plt.plot(bins3, a*b/np.linalg.norm(a*b), label=r'$\Psi_1\Psi_0$')
plt.plot(bins3, psi_1*psi_1/np.linalg.norm(psi_1*psi_1), label=r'$\Phi_1\Phi_1$')
d = amp3/np.linalg.norm(amp3)*(b/a)**2
plt.plot(bins3, d/np.linalg.norm(d), label=r'f$\rm\Delta r_{OH}$)*$(\Psi_1\Psi_0)^2$')


plt.xlabel(r'$\rm\Delta r_{OH}$')
plt.legend()
# plt.show()


a = (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (coords-0.039) ** 2))

b = (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (coords-0.039) ** 2)) * (2 * mw) ** (1 / 2) * (coords-0.039)
asdf = (2 * mw) ** (1 / 2) * (coords-0.039)
c = np.sum(wvfn[1]*b/a)/np.sum(wvfn[1])
print(np.sum(b/a))
print(np.sum(wvfn[1]/a))
print(c)

print(f'dipole = {np.dot(wvfn[1]/np.sum(wvfn[1]), asdf*coords)}')
print(f'dipole = {np.dot(wvfn[1], asdf*coords)/np.sum(wvfn[1])}')
print(f'q^2 = {np.dot(wvfn[1]/np.sum(wvfn[1]), asdf*coords**2)}')
print(f'q^2 = {np.dot(wvfn[1], asdf*coords**2)/np.sum(wvfn[1])}')
print(f'q^2 = {np.dot(wvfn[1]/np.sum(wvfn[1]), asdf**2*coords**2)}')



#
# plt.plot(bins3, a**2/4, label=r'$\Psi_0^2$')
# plt.plot(bins3, -interpolate.splev(bins3, interp1, der=0)*a*5, label=r'$\Phi_1\Psi_0$')
trial_wvfn = np.load('Harmonic_oscillator/Anharmonic_trial_wvfn_ground_150_wvnum.npy')
interp0 = interpolate.splrep(trial_wvfn[0], trial_wvfn[1], s=0)
a = interpolate.splev(bins3, interp0, der=0)
# plt.plot(bins3, a**2/np.linalg.norm(a**2), label=r'$\Phi_0^2$')
# plt.plot(bins3, (interpolate.splev(bins3, interp0, der=0)*a*5), label=r'$\Phi_0\Psi_0$')
# plt.plot(bins3, (interpolate.splev(bins3, interp0, der=0)*b*5), label=r'$\Phi_0\Psi_1$')
# plt.plot(bins3, -interpolate.splev(bins3, interp0, der=0)*interpolate.splev(bins3, interp1, der=0)*100, label=r'$\Phi_0\Phi_1$')


# for i in np.arange(0, 200, 1):
#     amp1, xx1 = np.histogram(coords, weights=d[i], range=(-0.5, 0.5), bins=75)
#     bins1 = (xx1[1:] + xx1[:-1]) / 2.
#     plt.plot(bins1, amp1)
plt.legend()
plt.show()


# # function that plugs in the coordinates of the walkers and gets back the values of the trial wavefunction
# def psi_t(coords, shift, DW):
#     coords = coords - shift
#     if DW:
#         return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))
#     else:
#         return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * coords ** 2)) * (2 * mw) ** (1 / 2) * coords



# a = np.sum(excite_d*coords)
# b = np.sum(ground_d)
# something = np.sum(excite_d)
# something_else = np.sum(wvfn[1])
# # find_it = np.argwhere(ground_d < 0.01)
# # ground_d[find_it] = 0.01
# # find_it = np.argwhere(excite_d < 0.01)
# # excite_d[find_it] = 0
# c = np.sum(np.nan_to_num(excite_d**2/ground_d))
# # c = np.sum(excite_d)
# d = np.sqrt(c*b)
# e = a/d
# blah = np.sum(excite_d**2/ground_d*coords**2)/np.sum(excite_d**2/ground_d)
# blah2 = np.sum(ground_d*coords**2)/np.sum(ground_d)
# blah3 = np.sum(wvfn[1]*coords**2)/np.sum(wvfn[1])
# print(blah)
# print(blah2)
# print(blah3)
# print(e)