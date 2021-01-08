import numpy as np
from scipy import interpolate
import Prot_water_funcs.Non_imp_sampled as pni
import Prot_water_funcs.Imp_sample as pi

tetramer_data = np.load('../../../CH5_Normal_DMC/Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/'
                        'ptetramer_full_imp_samp_patched_20000_Walkers_Test_1.npz')


wvfn = np.load("../Prot_water_params/wvfns/free_oh_wvfn.npy")
free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)

hbond_wvfn = np.load("../Prot_water_params/wvfns/shared_prot_moveable_wvfn.npy")
hbond_wvfn = interpolate.splrep(hbond_wvfn[:, 0], hbond_wvfn[:, 1], s=0)

trial_wvfn_tetramer = {
    "reg_oh": free_oh_wvfn,
    "ang": None,
    "hbond": hbond_wvfn,
    "OO_shift": np.array(np.loadtxt("../Prot_water_params/shared_prot_params/bowman_patched_h9o4_Re_Polynomials")),
    "OO_scale": np.array(np.loadtxt("../Prot_water_params/shared_prot_params/bowman_patched_h9o4_Std_Polynomials"))
}
atoms_tetramer = ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O']

psi = pni.Walkers(20000, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], np.ones((13, 3)))
psi.coords = tetramer_data['coords'][-1]
psi.weights = tetramer_data['weights'][-1]

def sigma_alpha(dtau, atoms):
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_H = 1.00782503223 / (Avo_num * me * 1000)
    m_D = 2.01410177812 / (Avo_num * me * 1000)
    m_O = 15.99491461957 / (Avo_num * me * 1000)
    alpha = 1. / (2. * dtau)
    sigmaH = np.sqrt(dtau / m_H)
    sigmaO = np.sqrt(dtau / m_O)
    sigmaD = np.sqrt(dtau / m_D)
    sigmaOH = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        if psi.atoms[i].upper() == 'H':
            sigmaOH[i] = np.array([[sigmaH] * 3])
        elif psi.atoms[i].upper() == 'D':
            sigmaOH[i] = np.array([[sigmaD] * 3])
        elif psi.atoms[i].upper() == 'O':
            sigmaOH[i] = np.array([[sigmaO] * 3])
    return sigmaOH, alpha
# sigma_dtau1, alpha1 = sigma_alpha(1, atoms_tetramer)
#
# pni.simulation_time(psi, sigma_dtau1, 1, 1, 1, 1, 1, True, .01, 20, system='tetramer')
# sigma_dtau10, _ = sigma_alpha(10, atoms_tetramer)
# pni.simulation_time(psi, sigma_dtau10, 1, 10, 1, 1, 1, True, .01, 20, system='tetramer')

# psi = pi.Walkers(20000, atoms, np.ones((13, 3)))
# psi.coords = tetramer_data['coords'][-1]
# psi.weights = tetramer_data['weights'][-1]
# psi.interp_ang = None
# psi.interp_reg_oh = trial_wvfn_tetramer['reg_oh']
# psi.interp_hbond = trial_wvfn_tetramer['hbond']
# psi.interp_OO_scale = trial_wvfn_tetramer['OO_scale']
# psi.interp_OO_shift = trial_wvfn_tetramer['OO_shift']

# f, psi.psit = pi.drift(psi.coords, psi.atoms, (len(atoms)-1)/3, psi.interp_reg_oh, psi.interp_hbond,
#                        psi.interp_OO_shift, psi.interp_OO_scale, psi.interp_ang, True)
# pi.simulation_time(psi, alpha1, sigma_dtau1, f, 1, 1, 1, 1, 1, 0.01, 20, True, system='tetramer')


trimer_data = np.load('../../../CH5_Normal_DMC/Trial_wvfn_testing/results/ptrimer_full_imp_samp/'
                        'ptrimer_full_imp_samp_20000_Walkers_Test_1.npz')


wvfn = np.load("../Prot_water_params/wvfns/free_oh_wvfn.npy")
free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)

hbond_wvfn = np.load("../Prot_water_params/wvfns/shared_prot_moveable_wvfn.npy")
hbond_wvfn = interpolate.splrep(hbond_wvfn[:, 0], hbond_wvfn[:, 1], s=0)

trial_wvfn_trimer = {
    "reg_oh": free_oh_wvfn,
    "ang": None,
    "hbond": hbond_wvfn,
    "OO_shift": np.array(np.loadtxt("../Prot_water_params/shared_prot_params/bowman_patched_h7o3_Re_Polynomials")),
    "OO_scale": np.array(np.loadtxt("../Prot_water_params/shared_prot_params/bowman_patched_h7o3_Std_Polynomials"))
}

atoms_trimer = ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O']

psi = pni.Walkers(20000, ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O'], np.ones((10, 3)))
psi.coords = trimer_data['coords'][-1]
psi.weights = trimer_data['weights'][-1]

sigma_dtau1, alpha1 = sigma_alpha(1, atoms_trimer)

# pni.simulation_time(psi, sigma_dtau1, 1, 1, 1, 1, 1, True, .01, 20, system='trimer')
# sigma_dtau10, _ = sigma_alpha(10, atoms_trimer)
# pni.simulation_time(psi, sigma_dtau10, 1, 10, 1, 1, 1, True, .01, 20, system='trimer')

# psi = pi.Walkers(20000, atoms, np.ones((10, 3)))
# psi.coords = trimer_data['coords'][-1]
# psi.weights = trimer_data['weights'][-1]
# psi.interp_ang = None
# psi.interp_reg_oh = trial_wvfn_trimer['reg_oh']
# psi.interp_hbond = trial_wvfn_trimer['hbond']
# psi.interp_OO_scale = trial_wvfn_trimer['OO_scale']
# psi.interp_OO_shift = trial_wvfn_trimer['OO_shift']

# f, psi.psit = pi.drift(psi.coords, psi.atoms, (len(atoms)-1)/3, psi.interp_reg_oh, psi.interp_hbond,
#                        psi.interp_OO_shift, psi.interp_OO_scale, psi.interp_ang, True)
# pi.simulation_time(psi, alpha1, sigma_dtau1, f, 1, 1, 1, 1, 1, 0.01, 20, True, system='trimer')



water_data = np.load('../../../CH5_Normal_DMC/Trial_wvfn_testing/results/water_imp_samp/'
                        'water_imp_samp_20000_Walkers_Test_1.npz')

wvfn = np.load("../Prot_water_params/wvfns/free_oh_wvfn.npy")
free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)
trial_wvfn_water = {
    "reg_oh": free_oh_wvfn,
    "ang": None,
}

atoms_water = ['H', "H", 'O']
psi = pni.Walkers(20000, atoms_water, np.ones((3,3)))
psi.coords = water_data['coords'][-1]
psi.weights = water_data['weights'][-1]

sigma_dtau1, alpha1 = sigma_alpha(1, atoms_water)
sigma_dtau10, alpha10 = sigma_alpha(10, atoms_water)

pni.simulation_time(psi, sigma_dtau1, 1, 1, 1, 1, 1, True, .01, 20, system='water')
pni.simulation_time(psi, sigma_dtau10, 1, 10, 1, 1, 1, True, .01, 20, system='water')

psi = pi.Walkers(20000, atoms_water, np.ones((10, 3)))
psi.coords = water_data['coords'][-1]
psi.weights = water_data['weights'][-1]
psi.interp_ang = None
psi.interp_reg_oh = trial_wvfn_water['reg_oh']

f, psi.psit = pi.drift(psi.coords, psi.atoms, int((len(atoms_water)-1)/3), psi.interp_reg_oh, psi.interp_hbond,
                       psi.interp_OO_shift, psi.interp_OO_scale, psi.interp_ang, True)
pi.simulation_time(psi, alpha1, sigma_dtau1, f, 1, 1, 1, 1, 1, 0.01, 20, True, system='water')



tetramer_weights = tetramer_data['weights'][-1]
# psi = pni.Walkers(20000, atoms_tetramer, np.ones((4, 3)))
# psi.coords = tetramer_data['coords'][-1]
# psi = pi.Parr_Potential(psi)
# print(np.average(np.load('intial_tetramer_pot.npy')))
# tetramer_pot = psi.V
# np.save('intial_tetramer_pot', psi.V)
trimer_weights = trimer_data['weights'][-1]
# psi = pni.Walkers(20000, atoms_trimer, np.ones((4, 3)))
# psi.coords = trimer_data['coords'][-1]
# psi = pi.Parr_Potential(psi)
# trimer_pot = psi.V
# np.save('intial_trimer_pot', psi.V)
psi = pni.Walkers(20000, atoms_water, np.ones((4,3)))
psi.coords = water_data['coords'][-1]
psi = pi.Parr_Potential(psi)
water_pot = psi.V
np.save('intial_water_pot', water_pot)

tetramer_pot_initial = np.load('intial_tetramer_pot.npy')

trimer_pot_initial = np.load('intial_trimer_pot.npy')

water_pot_initial = np.load('intial_water_pot.npy')

import matplotlib.pyplot as plt

tetramer_pot_dtau1 = np.load('nonny_imp_samp_dtau_1_tetramer.npy')

tetramer_pot_dtau10 = np.load('nonny_imp_samp_dtau_10_tetramer.npy')

tetramer_pot_dtau1_imp = np.load('non_imp_samp_dtau_1_tetramer.npy')

trimer_pot_dtau1 = np.load('nonny_imp_samp_dtau_1_trimer.npy')

trimer_pot_dtau10 = np.load('nonny_imp_samp_dtau_10_trimer.npy')

trimer_pot_dtau1_imp = np.load('non_imp_samp_dtau_1_trimer.npy')

water_pot_dtau1 = np.load('nonny_imp_samp_dtau_1_water.npy')

water_pot_dtau10 = np.load('nonny_imp_samp_dtau_10_water.npy')

water_pot_dtau1_imp = np.load('non_imp_samp_dtau_1_water.npy')
har2wave = 219474.6

amp1, xx = np.histogram((tetramer_pot_dtau1-tetramer_pot_initial)*har2wave/4.5, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp1/np.linalg.norm(amp1), label=r'No imp. samp. $\Delta\tau$=1 H$_9$O$_4^+$', color='blue')

amp1, xx = np.histogram((water_pot_dtau1-water_pot_initial)*har2wave, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp1/np.linalg.norm(amp1), label=r'No imp. samp. $\Delta\tau$=1 H$_2$O', color='orange')

plt.legend()
plt.xlabel(r'V($\tau + \Delta\tau$) - V($\tau$)')
plt.show()

amp2, xx = np.histogram((tetramer_pot_dtau10-tetramer_pot_initial)*har2wave/4.5, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp2/np.linalg.norm(amp2), label=r'No imp. samp. $\Delta\tau$=10 H$_9$O$_4^+$', color='green')

amp2, xx = np.histogram((water_pot_dtau10-water_pot_initial)*har2wave, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp2/np.linalg.norm(amp2), label=r'No imp. samp. $\Delta\tau$=10 H$_2$O', color='red')

plt.legend()
plt.xlabel(r'V($\tau + \Delta\tau$) - V($\tau$)')
plt.show()

amp3, xx = np.histogram((tetramer_pot_dtau1_imp-tetramer_pot_initial)*har2wave/4.5, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp3/np.linalg.norm(amp3), label=r'Imp. samp. $\Delta\tau$=1 H$_9$O$_4^+$', color='purple')

amp3, xx = np.histogram((water_pot_dtau1_imp-water_pot_initial)*har2wave, range=(-5000, 5000), bins=75)

plt.plot(bin, amp3/np.linalg.norm(amp3), label=r'Imp. samp. $\Delta\tau$=1 H$_2$O', color='magenta')
bin = (xx[1:] + xx[:-1])/2
plt.legend()
plt.xlabel(r'V($\tau + \Delta\tau$) - V($\tau$)')
plt.show()

amp1, xx = np.histogram((trimer_pot_dtau1-trimer_pot_initial)*har2wave/3.5, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp1/np.linalg.norm(amp1), label=r'No imp. samp. $\Delta\tau$=1 H$_7$O$_3^+$', color='blue')

amp1, xx = np.histogram((water_pot_dtau1-water_pot_initial)*har2wave, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp1/np.linalg.norm(amp1), label=r'No imp. samp. $\Delta\tau$=1 H$_2$O', color='orange')

plt.legend()
plt.xlabel(r'V($\tau + \Delta\tau$) - V($\tau$)')
plt.show()

amp2, xx = np.histogram((trimer_pot_dtau10-trimer_pot_initial)*har2wave/3.5, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp2/np.linalg.norm(amp2), label=r'No imp. samp. $\Delta\tau$=10 H$_7$O$_3^+$', color='green')

amp2, xx = np.histogram((water_pot_dtau10-water_pot_initial)*har2wave, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp2/np.linalg.norm(amp2), label=r'No imp. samp. $\Delta\tau$=10 H$_2$O', color='red')

plt.legend()
plt.xlabel(r'V($\tau + \Delta\tau$) - V($\tau$)')
plt.show()

amp3, xx = np.histogram((trimer_pot_dtau1_imp-trimer_pot_initial)*har2wave/3.5, range=(-5000, 5000), bins=75)
bin = (xx[1:] + xx[:-1])/2

plt.plot(bin, amp3/np.linalg.norm(amp3), label=r'Imp. samp. $\Delta\tau$=1 H$_7$O$_3^+$', color='purple')

amp3, xx = np.histogram((water_pot_dtau1_imp-water_pot_initial)*har2wave, range=(-5000, 5000), bins=75)

plt.plot(bin, amp3/np.linalg.norm(amp3), label=r'Imp. samp. $\Delta\tau$=1 H$_2$O', color='magenta')
plt.legend()
plt.xlabel(r'V($\tau + \Delta\tau$) - V($\tau$)')
plt.show()