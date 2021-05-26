import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
omega = 3600./har2wave

wexe = 75./har2wave
De = omega**2/4/wexe
omega = 3600/har2wave
mw = m_red * 3600/har2wave
A = np.sqrt(omega**2 * m_red/(2*De))


ground_coords = np.zeros((5, 27, 10000))
ground_weights = np.zeros((5, 27, 10000))
for i in range(5):
    blah = np.load(f'ground_state_morse_{i}.npz')
    coords = blah['coords']
    weights = blah['weights']
    ground_coords[i] = coords
    ground_weights[i] = weights

# excite_coords = np.zeros((5, 27, 10000))
# excite_weights = np.zeros((5, 27, 10000))
# for i in range(5):
#     blah = np.load(f'excite_state_morse_{i}.npz')
#     coords = blah['coords']
#     weights = blah['weights']
#     excite_coords[i] = coords
#     excite_weights[i] = weights


def Harmonic_wvfn(x, state):
    if state == 1:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (x-0.039) ** 2)) * (2 * mw) ** (1 / 2) * (x-0.039)
    else:
        return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * (x-0.039) ** 2))


trial_wvfn = np.load('Harmonic_oscillator/Anharmonic_trial_wvfn_150_wvnum.npy')
interp1 = interpolate.splrep(trial_wvfn[0], trial_wvfn[1], s=0)
trial_wvfn = np.load('Harmonic_oscillator/Anharmonic_trial_wvfn_ground_150_wvnum.npy')
interp0 = interpolate.splrep(trial_wvfn[0], trial_wvfn[1], s=0)

ground_weights = np.reshape(ground_weights, (5, 135000*2))
ground_coords = np.reshape(ground_coords, (5, 135000*2))
ratio = np.zeros((5, 100))
# for i in range(5):
#     ratio[i] = Harmonic_wvfn(ground_coords[i], 1)/Harmonic_wvfn(ground_coords[i], 0)
# excite_coords = np.reshape(excite_coords, (10, 135000))
# excite_weights = np.reshape(excite_weights, (10, 135000))
for i in range(1):
    amp, xx = np.histogram(ground_coords[i], weights=ground_weights[i], bins=100, range=(-1, 1), density=True)
    bin = (xx[1:] + xx[:-1]) / 2
    # plt.plot(bin, amp/np.linalg.norm(amp))
    ratio[i] = Harmonic_wvfn(bin, 1)/Harmonic_wvfn(bin, 0)
    new_amp = amp/np.linalg.norm(amp)*ratio[i]
    # plt.plot(bin, new_amp/np.linalg.norm(new_amp), label=r'f($\rm\Delta r_{OH}$)*$\Psi_1/\Psi_0$', linewidth=3)
    psi_1 = interpolate.splev(bin, interp1, der=0)
    psi_0 = interpolate.splev(bin, interp0, der=0)
    plt.plot(bin, -Harmonic_wvfn(bin, 1), label=r'$\Phi_1$', linewidth=3, color='orange')

plt.xlabel(r'$\rm\Delta r_{OH}$', fontsize=28)
plt.tick_params(axis='both', labelsize=18)
# plt.legend(fontsize=20)
# plt.xlim(-0.5, 0.5)
plt.tight_layout()
plt.show()
