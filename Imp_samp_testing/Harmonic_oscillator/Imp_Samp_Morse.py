import numpy as np
import copy
from scipy import interpolate

# DMC parameters
dtau = 1.
N_0 = 5000
time_total = 10000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# parameters for the potential and for the analytic wavefuntion
omega = 3704.47/har2wave
omega_mon = 3884.81/har2wave

wexe = 75.26/har2wave
wexe_mon = 86.9175/har2wave
De_mon = omega_mon**2/4/wexe_mon
De = omega**2/4/wexe
sigmaOH = np.sqrt(dtau/m_red)

mw = m_red * omega
A_mon = np.sqrt(omega_mon**2 * m_red/(2*De_mon))
A = np.sqrt(omega**2 * m_red/(2*De))

# Loads the wavefunction from the DVR for interpolation
Psi_t = np.load('Water_Morse_trial_wvfn_ground.npy')
Psi_t1 = np.load('Water_Morse_trial_wvfn_excite.npy')
wvfn1 = np.load('Water_dimer_Morse_trial_wvfn_ground.npy')
wvfn2 = np.load('Water_dimer_Morse_trial_wvfn_excite.npy')
# print(wvfn1[0, np.argmax(wvfn1[1])])
# shift = wvfn1[0, np.argmax(wvfn1[1])]
# shift = 0.175
interp_w = interpolate.splrep(Psi_t[0], Psi_t[1], s=0)
interp_w2 = interpolate.splrep(Psi_t1[0], Psi_t1[1], s=0)
shift = 0.011457*ang2bohr
shift2 = 0.11457*ang2bohr
interp = interpolate.splrep(wvfn1[0], wvfn1[1], s=0)
interp2 = interpolate.splrep(wvfn2[0], -wvfn2[1], s=0)
re_mon = 0.961369*ang2bohr
re_dimer = 0.972826*ang2bohr


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, shift, psit):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.ones(walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.shift = shift
        self.psit = psit


# function that plugs in the coordinates of the walkers and gets back the values of the trial wavefunction
# def psi_t(coords, shift=0):
# def psi_t(coords):
#     coords = coords - shift
#     return (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*coords**2))
    # return (mw / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw * coords ** 2)) * \
    #                 (2 * mw) ** (1 / 2) * coords

def psi_t(coords, shift, psit):
    coords = coords - shift
    if psit == 'water':
        return interpolate.splev(coords, interp_w, der=0)
    elif psit == 'dimer':
        return interpolate.splev(coords, interp, der=0)


def excite_psi_t(coords, shift, psit):
    coords = coords - shift
    if psit == 'water':
        return interpolate.splev(coords, interp_w2, der=0)
    elif psit == 'dimer':
        return interpolate.splev(coords, interp2, der=0)


def anharm_psi(coords):
    return interpolate.splev(coords, interp, der=0)


def anharm_excite_psi(coords):
    return interpolate.splev(coords, interp2, der=0)


# Calculation of the drift term
# def drift(coords, shift=0):
# def drift(coords):
#     coords = coords - shift
#     return -2.*mw*coords

def drift(coords, shift, psit):
    coords = coords - shift
    if psit == 'water':
        return 2 * interpolate.splev(coords, interp_w, der=1) / interpolate.splev(coords, interp_w, der=0)
    elif psit == 'dimer':
        return 2*interpolate.splev(coords, interp, der=1)/interpolate.splev(coords, interp, der=0)


def excite_drift(coords, shift, psit):
    coords = coords - shift
    if psit == 'water':
        return 2 * interpolate.splev(coords, interp_w2, der=1) / interpolate.splev(coords, interp_w2, der=0)
    elif psit == 'dimer':
        return 2*interpolate.splev(coords, interp2, der=1)/interpolate.splev(coords, interp2, der=0)


def anharm_drift(coords, psi):
    return 2*interpolate.splev(coords, interp, der=1)/psi


def anharm_excite_drift(coords, psi):
    return 2*interpolate.splev(coords, interp2, der=1)/psi


# Calculates the second derivative of the trial wavefunction for the kinetic energy of the local energy
# def sec_dir(coords, shift=0):
# def sec_dir(coords):
#     coords = coords - shift
#     return mw**2*coords**2 - mw

def sec_dir(coords, shift, psit):
    coords = coords - shift
    if psit == 'water':
        return interpolate.splev(coords, interp_w, der=2) / interpolate.splev(coords, interp_w, der=0)
    elif psit == 'dimer':
        return interpolate.splev(coords, interp, der=2)/interpolate.splev(coords, interp, der=0)


def excite_E_loc(coords, pot, shift, psit):
    coords = coords - shift
    if psit == 'water':
        sd = interpolate.splev(coords, interp_w2, der=2)/interpolate.splev(coords, interp_w2, der=0)
    elif psit == 'dimer':
        sd = interpolate.splev(coords, interp2, der=2)/interpolate.splev(coords, interp2, der=0)
    kin = -1./(2.*m_red)*sd
    return kin + pot


def anharm_E_loc(coords, pot, psi):
    sd = interpolate.splev(coords, interp, der=2)/psi
    kin = -1/(2*m_red)*sd
    return kin + pot


def anharm_excite_E_loc(coords, pot, psi):
    sd = interpolate.splev(coords, interp2, der=2)/psi
    kin = -1/(2*m_red)*sd
    return kin + pot


# Metropolis step to determine the ratio of Green's functions
def metropolis(x, y, Fqx, Fqy, shift, psit):
    psi_x = psi_t(x, shift, psit)
    psi_y = psi_t(y, shift, psit)
    pre_factor = (psi_y/psi_x)**2
    M = pre_factor*np.exp(1./2.*(Fqx + Fqy)*(sigmaOH**2/4.*(Fqx-Fqy) - (y-x)))
    return M


def Kinetic(Psi, Fqx):
    randomwalk = np.random.normal(0.0, sigmaOH, N_0)
    Drift = sigmaOH**2/2.*Fqx
    y = Psi.coords + randomwalk + Drift
    Fqy = drift(y, Psi.shift, Psi.psit)
    a = metropolis(Psi.coords, y, Fqx, Fqy, Psi.shift, Psi.psit)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a > check)
    nah = np.argwhere(a <= check)
    Psi.coords[accept] = y[accept]
    Fqy[nah] = Fqx[nah]
    return Psi, Fqy


def potential(Psi):
    Psi.V = De*(1. - np.exp(-A*(Psi.coords-re_dimer)))**2  # Morse potential
    return Psi


def mon_potential(coords):
    return De_mon*(1 - np.exp(-A_mon*(coords-re_mon)))**2


# Calculates the local energy of the trial wavefunction
# def E_loc(Psi, shift=0):
#     # psi = psi_t(Psi.coords, shift)
#     kin = -1./(2.*m_red)*sec_dir(Psi.coords, shift)
#     pot = Psi.V
#     Psi.El = kin + pot
#     return Psi

def E_loc(Psi):
    kin = -1./(2.*m_red)*sec_dir(Psi.coords, Psi.shift, Psi.psit)
    pot = Psi.V
    Psi.El = kin + pot
    return Psi


# Calculate Eref from the local energy and the weights of the walkers
def E_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/P0)
    return E_ref


# The weighting calculation that gets the weights of each walker in the simulation
def Weighting(Vref, Psi):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Vref) * dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    # if len(death) >= 1:
    #         print('%s walkers dead' %len(death))
    for i in death:  # iterate over the list of dead walkers
        ind = np.argmax(Psi.weights)  # find the walker with with most weight
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_El = float(Psi.El[ind])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_El
    return Psi


# Descendant weighting where the descendants of the walkers that replace those that die are kept track of
def desWeight(Vref, Psi):
    Psi.weights = Psi.weights*np.exp(-(Psi.El-Vref)*dtau)
    # Conditions to prevent one walker from obtaining all the weight
    threshold = 1. / float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_num = float(Psi.walkers[ind])  # make sure to keep track of the walker that is donating its weight
        Biggo_pot = float(Psi.V[ind])
        Biggo_El = float(Psi.El[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.walkers[i[0]] = Biggo_num
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_El
    return Psi


# Calculates the descendant weight for the walkers after descendant weighting
def descendants(Psi):
    for i in range(N_0):
        Psi.d[i] = np.sum(Psi.weights[Psi.walkers == i])
    return Psi.d


def run(propagation, shift, psit):
    psi = Walkers(N_0, shift, psit)
    Fqx = drift(psi.coords, psi.shift, psi.psit)
    Psi, Fqx = Kinetic(psi, Fqx)
    Psi = potential(Psi)
    Psi = E_loc(Psi)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi)
    wvfn = np.zeros((3, 19, N_0))
    counter = 0
    dw_counter = propagation

    # initial parameters before running the calculation
    DW = False  # a parameter that will implement descendant weighting when True
    Psi_dtau = 0
    for i in range(int(time_total)):
        if i % 1000 == 0:
            print(i)
        if DW is False:
            prop = float(propagation)

        Psi, Fqx = Kinetic(new_psi, Fqx)
        Psi = potential(Psi)
        Psi = E_loc(Psi)

        if DW is False:
            new_psi = Weighting(Eref, Psi)
        elif DW is True:
            if Psi_dtau == 0:
                Psi_tau = copy.deepcopy(Psi)
                Psi_dtau = copy.deepcopy(Psi_tau)
                new_psi = desWeight(Eref, Psi_dtau)
            else:
                new_psi = desWeight(Eref, Psi)
            prop -= 1.

        if (i+1) % 500 == 0 and i != 9999:
            DW = True
            wvfn[0, counter] = new_psi.coords
            wvfn[1, counter] = new_psi.weights
            counter += 1

        if (i+1-250) % 500 == 0 and i != 249:
            wvfn[2, counter-1] = descendants(new_psi)
            new_psi.walkers = np.linspace(0, N_0-1, num=N_0)
            DW = False

        Eref = E_ref_calc(new_psi)

        Eref_array = np.append(Eref_array, Eref)

        if i >= (time_total - 1. - float(propagation)) and prop > 0:  # start of descendant weighting
            DW = True
        elif i >= (time_total - 1. - float(propagation)) and prop == 0.:  # end of descendant weighting
            d_values = descendants(new_psi)
            Psi_tau.d += d_values

    # wvfn = np.zeros((3, N_0))
    # wvfn[0, :] += Psi_tau.coords
    # wvfn[1, :] += Psi_tau.weights
    # wvfn[2, :] += Psi_tau.d
    # np.save('Imp_samp_morse_energy', Eref_array)
    # np.save('Imp_samp_morse_Psi', wvfn)
    return Eref_array, wvfn


# run(100)
energies = np.zeros((4, 5))
# coords = np.zeros((500, 100))
# weights = np.zeros((500, 100))
# des = np.zeros((500, 100))
coords = np.zeros((5, 19, 5000))
weights = np.zeros((5, 19, 5000))
des = np.zeros((5, 19, 5000))
for i in np.arange(0, 5):
    # en, psi = run(250, 0, 'water')
    # np.savez(f'water_no_shift_{i+1}', energy=en, wvfn=psi)
    blah = np.load(f'water_shift_{i+1}.npz')['energy']
    energies[0, i] = np.mean(blah[2000:])
#     en, psi = run(250, shift, 'water')
#     np.savez(f'water_shift_{i+1}', energy=en, wvfn=psi)
    blah = np.load(f'water_no_shift_{i + 1}.npz')
    coords[i] = blah['wvfn'][0]
    weights[i] = blah['wvfn'][1]
    des[i] = blah['wvfn'][2]
    energies[1, i] = np.mean(blah['energy'][2000:])
#     en, psi = run(250, shift2, 'water')
#     np.savez(f'water_big_shift_{i+1}', energy=en, wvfn=psi)
    blah = np.load(f'water_big_shift_{i + 1}.npz')['energy']
    energies[2, i] = np.mean(blah[2000:])
#     en, psi = run(250, 0, 'dimer')
#     np.savez(f'dimer_no_shift_{i+1}', energy=en, wvfn=psi)
    blah = np.load(f'dimer_no_shift_{i + 1}.npz')['energy']
    energies[3, i] = np.mean(blah[2000:])
# for i in np.arange(5, 100):
#     en, psi = run(250, 0, 'water')
#     np.savez(f'water_no_shift_{i+1}', energy=en, wvfn=psi)
#
#
avg_en = np.average(energies*har2wave - 1833.419, axis=-1)
std_en = np.std(energies*har2wave, axis=-1)

coords = coords.reshape((5, 5000*19)).T
weights = weights.reshape((5, 5000*19)).T
des = des.reshape((5, 5000*19)).T
desc = np.average(coords-re_dimer, weights=des, axis=0)/ang2bohr
avg_desc = np.average(desc)
std_desc = np.std(desc)

f = np.average(coords-re_dimer, weights=weights, axis=0)/ang2bohr
avg_f = np.average(f)
std_f = np.std(f)

phi = np.dot(Psi_t[1], Psi_t[1]*(Psi_t[0]-re_dimer))/ang2bohr

psi = np.dot(wvfn1[1], wvfn1[1]*(wvfn1[0]-re_dimer))/ang2bohr

print(f'desc {avg_desc} +/- {std_desc}')
print(f'f {avg_f} +/- {std_f}')
print(f'mod f {avg_f*2 - phi} +/- {std_f*2}')
print(f'exact {psi}')
#
#
term1 = np.zeros(5)
frac = interpolate.splev(coords, interp_w2, der=0)/interpolate.splev(coords, interp_w, der=0)
for i in range(5):
    # term1[i] = np.average(-frac[:, i]*coords[:, i], weights=weights[:, i])
    term1[i] = np.dot(weights[:, i], -frac[:, i]*(coords[:, i]-re_dimer)/ang2bohr)/np.sum(weights[:, i])
avg_term1 = np.average(term1)
std_term1 = np.std(term1)

term3 = np.dot(Psi_t[1], -Psi_t1[1]*(Psi_t[0]-re_dimer))/ang2bohr

exact = np.dot(wvfn1[1], wvfn2[1]*(wvfn1[0]-re_dimer))/ang2bohr

print(f'term1 {avg_term1} +/- {std_term1}')
print(f'term 3 {term3}')
print(f'exact {exact}')
#
# # avg = np.average(energies, axis=-1)
# # std = np.std(energies, axis=-1)
# # print(avg*har2wave - 1833.4200149265587)
# # print(avg[0]*har2wave - 1833.4200149265587)
# # print(avg[1]*har2wave - 1833.4200149265587)
# # print(avg[2]*har2wave - 1833.4200149265587)
# # print(avg[3]*har2wave - 1833.4200149265587)
# # print(std*har2wave)
coords = coords.flatten()
weights = weights.flatten()
des = des.flatten()
# ang2bohr = 1.e-10/5.291772106712e-11
#
amp, xx = np.histogram(coords, weights=weights, bins=75, range=(0.6*ang2bohr, 1.5*ang2bohr), density=True)
amp2, xx = np.histogram(coords, weights=des, bins=75, range=(0.6*ang2bohr, 1.5*ang2bohr), density=True)
bin = (xx[1:] + xx[:-1]) / 2.
truth = interpolate.splev(bin, interp, der=0)**2
phi = interpolate.splev(bin, interp_w, der=0)
phi1 = interpolate.splev(bin, interp_w2, der=0)
phi1 = phi1/np.linalg.norm(phi1)
truth1 = interpolate.splev(bin, interp2, der=0)
truth0 = interpolate.splev(bin, interp, der=0)
truth1 = truth1/np.linalg.norm(truth1)
truth0 = truth0/np.linalg.norm(truth0)
truth1 = truth1*truth0
# phi = phi/np.max(phi)*np.max(truth)/np.linalg.norm(truth)
# phi = phi/np.dot(phi, phi)
amp = amp/np.linalg.norm(amp)
amp2 = amp2/np.linalg.norm(amp2)
truth = truth/np.linalg.norm(truth)
mod = -amp*phi1/phi
mod = mod/np.max(mod)*np.max(truth1)
phi_sq = phi**2/np.linalg.norm(phi**2)
# truth = truth/np.max(truth)*np.max(phi)
#
# #
import matplotlib.pyplot as plt
# # plt.plot(bin/ang2bohr, mod, label=r'f($\rm{\Delta r_{OH}}$)*$\frac{\rm{\Phi_{T, 1}(\Delta r_{OH})}}{\rm{\Phi_{T, 0}(\Delta r_{OH})}}$')
# # plt.plot(bin/ang2bohr, )
# # plt.plot(bin/ang2bohr, truth1, label=r'$\rm{\Psi_1(\Delta r_{OH})}$$\rm{\Psi_0(\Delta r_{OH})}$')
# # plt.xlabel(r'r$_{\rm{OH}}$ $\rm\AA$', fontsize=20)
# # plt.tick_params(axis='both', labelsize=18)
# # plt.legend(frameon=False, fontsize=12)
# # plt.tight_layout()
# # plt.savefig('f_compared_to_psi1psi0')
# # plt.show()
#
# # plt.figure(figsize=(12,7))
plt.plot(bin/ang2bohr, amp, color='blue', linestyle='dashed', label=r'f$_0$($\rm{\Delta r_{OH}}$)')
# plt.plot(bin/ang2bohr, amp2, color='red', linestyle='dotted', label=r'Desc. Weight')
# plt.plot(bin/ang2bohr, 2*amp - phi_sq, color='green', label=r'Eq. 19')
plt.plot(bin/ang2bohr, truth, color='black', linestyle='dashdot', label=r'Exact')
plt.xlabel(r'r$_{\rm{OH}}$ $[/\rm\AA]$', fontsize=20)
plt.ylabel(r'$\rm{(\Psi_0(r_{OH}))^2}$', fontsize=20)
plt.tick_params(axis='both', labelsize=18)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig('f_compared_to_psi_sq')
plt.show()


psi = Walkers(500, 0, 'water')
psi.coords = np.linspace(0.65*ang2bohr, 1.65*ang2bohr, 500)
psi = potential(psi)
psi = E_loc(psi)
wvfn = psi_t(psi.coords, psi.shift, psi.psit)
d = drift(psi.coords, psi.shift, psi.psit)*sigmaOH**2/2


import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(psi.coords/ang2bohr, psi.V*har2wave, color='black', label=r'$\rm{V_{dim}}$')
ax[0, 0].plot(psi.coords/ang2bohr, mon_potential(psi.coords)*har2wave, linestyle='--', color='gray', label=r'$\rm{V_{mon}}$')
ax[0, 0].set_ylabel(r'V($\rm{r_{OH}}$) [/cm$^{-1}$]', fontsize=20)
ax[0, 1].plot(psi.coords/ang2bohr, wvfn, color='blue', linestyle='--', label=r'$\rm{\Phi_{0}^mon} (\rm{r_{OH}})$')
ax[0, 1].set_ylabel(r'$\rm{\Phi_T(\rm{r_{OH}})}$', fontsize=20)
ax[1, 0].plot(psi.coords/ang2bohr, psi.El*har2wave, color='blue', linestyle='--', label=r'$\rm{\Phi_{0}^mon} (\rm{r_{OH}})$')
ax[1, 0].set_ylabel(r'E$_{\rm{L}}$($\rm{r_{OH}}$) [/cm$^{-1}$]', fontsize=20)
ax[1, 1].plot(psi.coords/ang2bohr, d/ang2bohr, color='blue', linestyle='--', label=r'$\rm{\Phi_{0}^{mon}} (\rm{r_{OH}})$')
ax[1, 1].set_ylabel(r'D($\rm{r_{OH}}$)$\rm{\Delta\tau}$ [/$\rm{\AA}$]', fontsize=20)

ax[0, 0].set_xlabel(r'$\rm{r_{OH}}$ [/$\rm{\AA}$]', fontsize=20)
ax[0, 1].set_xlabel(r'$\rm{r_{OH}}$ [/$\rm{\AA}$]', fontsize=20)
ax[1, 1].set_xlabel(r'$\rm{r_{OH}}$ [/$\rm{\AA}$]', fontsize=20)
ax[1, 0].set_xlabel(r'$\rm{r_{OH}}$ [/$\rm{\AA}$]', fontsize=20)

ax[0, 0].set_ylim(0, 10000)
ax[1, 0].set_ylim(0, 10000)
ax[1, 1].set_ylim(-0.008, 0.012)

psi.shift = shift2
psi = E_loc(psi)
wvfn = psi_t(psi.coords, psi.shift, psi.psit)
d = drift(psi.coords, psi.shift, psi.psit)*sigmaOH**2/2
#
ax[0, 1].plot(psi.coords/ang2bohr, wvfn, color='orange', label=r'$\rm{\Phi_{0}^mon} (\rm{r_{OH} - \delta})$')
ax[1, 0].plot(psi.coords/ang2bohr, psi.El*har2wave, color='orange', label=r'$\rm{\Phi_{0}^mon} (\rm{r_{OH} - \delta})$')
ax[1, 1].plot(psi.coords/ang2bohr, d/ang2bohr, color='orange', label=r'$\rm{\Phi_{0}^{mon}} (\rm{r_{OH} - \delta})$')


wvfn1 = anharm_psi(psi.coords)
d = anharm_drift(psi.coords, wvfn1)*sigmaOH**2/2
psi.El = anharm_E_loc(psi.coords, psi.V, wvfn1)
wvfn1 = wvfn1/np.max(wvfn1)*np.max(wvfn)

ax[0, 1].plot(psi.coords/ang2bohr, wvfn1, color='black', linestyle='dashdot', label=r'$\rm{\Phi_0}$')
ax[1, 0].plot(psi.coords/ang2bohr, psi.El*har2wave, color='black', linestyle='dashdot', label=r'$\rm{\Phi_0}$')
ax[1, 1].plot(psi.coords/ang2bohr, d/ang2bohr, color='black', linestyle='dashdot', label=r'$\rm{\Phi_0^{dim}}(\rm{r_{OH}})$')

wvfn = excite_psi_t(psi.coords, 0, psi.psit)
d = excite_drift(psi.coords, 0, psi.psit)*sigmaOH**2/2
psi.El = excite_E_loc(psi.coords, psi.V, 0, psi.psit)
#
# ax[0, 1].plot(psi.coords/ang2bohr, wvfn, color='green', linestyle='--', label=r'$\rm{\Phi_{1}^{mon}}$ Not Shifted')
# ax[1, 0].plot(psi.coords/ang2bohr, psi.El*har2wave, color='green', linestyle='--', label=r'$\rm{\Phi_{T, 1}}$ Not Shifted')
# ax[1, 1].plot(psi.coords[:np.argmax(d)]/ang2bohr, d[:np.argmax(d)]/ang2bohr, color='green', linestyle='--', label=r'$\rm{\Phi_{1}^{mon}}(\rm{r_{OH}})$')
# ax[1, 1].plot(psi.coords[np.argmax(d):]/ang2bohr, d[np.argmax(d):]/ang2bohr, color='green', linestyle='--')
#
#

wvfn = excite_psi_t(psi.coords, psi.shift, psi.psit)
d = excite_drift(psi.coords, psi.shift, psi.psit)*sigmaOH**2/2
psi.El = excite_E_loc(psi.coords, psi.V, psi.shift, psi.psit)
#
# ax[0, 1].plot(psi.coords/ang2bohr, wvfn, color='red', label=r'$\rm{\Phi_{T, 1}}$ Shifted')
# ax[1, 0].plot(psi.coords/ang2bohr, psi.El*har2wave, color='red', label=r'$\rm{\Phi_{T, 1}}$ Shifted')
# ax[1, 1].plot(psi.coords[:np.argmax(d)]/ang2bohr, d[:np.argmax(d)]/ang2bohr, color='red', label=r'$\rm{\Phi_{1}^{mon}}(\rm{r_{OH} - \delta})$ ')
# ax[1, 1].plot(psi.coords[np.argmax(d):]/ang2bohr, d[np.argmax(d):]/ang2bohr, color='red')

wvfn1 = anharm_excite_psi(psi.coords)
d = anharm_excite_drift(psi.coords, wvfn1)*sigmaOH**2/2
psi.El = anharm_excite_E_loc(psi.coords, psi.V, wvfn1)
wvfn1 = wvfn1/np.max(wvfn1)*np.max(wvfn)
#
# ax[0, 1].plot(psi.coords/ang2bohr, wvfn1, color='black', linestyle='dashdot', label=r'$\rm{\Phi_1}$')
# ax[1, 0].plot(psi.coords/ang2bohr, psi.El*har2wave, color='black', linestyle='dashdot', label=r'$\rm{\Phi_1}$')
# ax[1, 1].plot(psi.coords[:np.argmax(d)]/ang2bohr, d[:np.argmax(d)]/ang2bohr, color='black', linestyle='dashdot', label=r'$\rm{\Phi_1^{dim}}(\rm{r_{OH}})$')
# ax[1, 1].plot(psi.coords[np.argmax(d):]/ang2bohr, d[np.argmax(d):]/ang2bohr, color='black', linestyle='dashdot')


ax[0, 0].legend(frameon=False, loc='upper right', fontsize=12)
# ax[0, 1].legend()
ax[0, 0].tick_params(axis='both', labelsize=16)
ax[0, 1].tick_params(axis='both', labelsize=16)
ax[1, 1].tick_params(axis='both', labelsize=16)
ax[1, 0].tick_params(axis='both', labelsize=16)

ax[1, 1].legend(frameon=False, loc='upper right', fontsize=12)
# ax[1, 0].legend()
plt.tight_layout()
plt.savefig('Ground_and_excite_state_4_panel_imp_samp123')
plt.show()


