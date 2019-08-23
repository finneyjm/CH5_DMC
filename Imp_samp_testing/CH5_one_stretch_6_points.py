from Coordinerds.CoordinateSystems import *
from scipy import interpolate
import matplotlib.pyplot as plt
import copy

dtau = 1.
N_0 = 5000
time_steps = 10000.
alpha = 1./(2.*dtau)
# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_CH = (m_C*m_H)/(m_H+m_C)
m_CH5 = ((m_C + m_H*4)*m_H)/(m_H*5 + m_C)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# Values for Simulation
sigmaH = np.sqrt(dtau/m_H)
sigmaC = np.sqrt(dtau/m_C)
sigmaCH = np.array([[sigmaC]*3, [sigmaH]*3])
# Starting orientation of walkers
coords_initial = np.array([[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]])

ch_stretch = 2
Psi_t = np.load(f'Switch_min_wvfn_speed_51.0.npy')
interp = interpolate.splrep(Psi_t[0, :], Psi_t[1, :], s=0)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.arange(0, N_0)
        self.coords = np.array([coords_initial]*walkers)
        self.zmat = distance(self.coords)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


def psi_t(dist):
    return interpolate.splev(dist, interp, der=0)


def distance(coords):
    dist = np.zeros((len(coords[:, 0, 0]), 3))
    for i in range(3):
        dist[:, i] += coords[:, 1, i] - coords[:, 0, i]
    return np.linalg.norm(dist, axis=1)


def potential(Psi, CH):
    Psi.V = interpolate.splev(distance(Psi.coords), CH, der=0)
    return Psi


def drdx(dist, coords):
    chain = np.zeros((N_0, 2, 3))
    for xyz in range(3):
        chain[:, 0, xyz] += ((coords[:, 0, xyz]-coords[:, 1, xyz])/dist)
        chain[:, 1, xyz] += ((coords[:, 1, xyz]-coords[:, 0, xyz])/dist)
    return chain


def drift(dist, coords):
    psi = psi_t(dist)
    dr1 = drdx(dist, coords)
    der = (interpolate.splev(dist, interp, der=1)/psi)
    a = dr1.reshape((N_0, 1, 6))
    b = der.reshape((N_0, 1, 1))
    drift = np.matmul(b, a)
    return 2.*drift.reshape((N_0, 2, 3))


def metropolis(r1, r2, Fqx, Fqy, x, y):
    psi_1 = psi_t(r1)
    psi_2 = psi_t(r2)
    psi_ratio = (psi_2/psi_1)**2
    a = psi_ratio
    for atom in range(2):
        for xyz in range(3):
            if atom == 0:
                sigma = sigmaC
            else:
                sigma = sigmaH
            a *= np.exp(1./2.*(Fqx[:, atom, xyz] + Fqy[:, atom, xyz])*(sigma**2/4.*(Fqx[:, atom, xyz]-Fqy[:, atom, xyz])
                                                                       - (y[:, atom, xyz]-x[:, atom, xyz])))
    return a


def Kinetic(Psi, Fqx):
    Drift = sigmaCH**2/2.*Fqx
    randomwalk = np.zeros((N_0, 2, 3))
    randomwalk[:, 1, :] = np.random.normal(0.0, sigmaH, size=(N_0, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaC, size=(N_0, 3))
    y = randomwalk + Drift + np.array(Psi.coords)
    disty = distance(y)
    Fqy = drift(disty, y)
    a = metropolis(Psi.zmat, disty, Fqx, Fqy, Psi.coords, y)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    nah = np.argwhere(a <= check)
    Fqy[nah] = Fqx[nah]
    Psi.zmat[accept] = disty[accept]
    return Psi, Fqy


def E_loc(Psi):
    dist = distance(Psi.coords)
    psi = psi_t(dist)

    der1 = interpolate.splev(dist, interp, der=1)/psi
    der2 = interpolate.splev(dist, interp, der=2)/psi

    kin = -1./(2.*m_CH)*(der2 + der1*2./dist)
    Psi.El = kin + Psi.V
    return Psi


def E_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = np.average(Psi.El, weights=Psi.weights) - alpha*np.log(P/P0)
    return E_ref


def Weighting(Eref, Psi, DW):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    threshold = 1./float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        if DW is True:
            Biggo_num = int(Psi.walkers[ind])
            Psi.walkers[i[0]] = Biggo_num
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_el = float(Psi.El[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    return d


def run(propagation):
    DW = False
    psi = Walkers(N_0)
    pot = interpolate.splrep(np.linspace(1, 4, num=500), np.load(f'Potential_CH_stretch{ch_stretch}.npy'), s=0)
    Fqx = drift(psi.zmat, psi.coords)
    Psi, Fqx = Kinetic(psi, Fqx)
    Psi = potential(Psi, pot)
    Psi = E_loc(Psi)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, DW)

    Psi_tau = 0
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)
        Psi, Fqx = Kinetic(new_psi, Fqx)
        Psi = potential(Psi, pot)
        Psi = E_loc(Psi)

        if DW is False:
            prop = float(propagation)
        elif DW is True:
            prop -= 1.
            if Psi_tau == 0:
                Psi_tau = copy.deepcopy(Psi)
        new_psi = Weighting(Eref, Psi, DW)
        Eref = E_ref_calc(new_psi)
        Eref_array = np.append(Eref_array, Eref)

        if i >= (time_steps - 1. - float(propagation)) and prop > 0.:
            DW = True
        elif i >= (time_steps - 1. - float(propagation)) and prop == 0.:
            d_values = descendants(new_psi)
    print(np.mean(Eref_array[1000:]*har2wave))
    return Eref_array


# for i in range(5):
#     run(50)
#     print(f'test {i+1} done')

# Psi = Walkers(N_0)
# Psi.coords[:, 1, :] = np.array([np.linspace(0.46188, 1.04, num=N_0)*ang2bohr + 4.]*3).T
# dist = distance(Psi.coords)
# pot = interpolate.splrep(np.linspace(1, 4, num=500), np.load(f'Potential_CH_stretch{ch_stretch}.npy'), s=0)
# Psi = potential(Psi, pot)
# alpha = [1.0, 3.0, 5.0]#[1.0, 2.0, 3.0, 4.0, 5.0]
# plt.plot(dist/ang2bohr, Psi.V*har2wave, label='Potential')
# for i in range(len(alpha)):
#     Psi_t = np.load(f'Switch_min_wvfn_speed_{alpha[i]}.npy')
#     interp = interpolate.splrep(Psi_t[0, :], Psi_t[1, :], s=0)
#     Psi = E_loc(Psi)
#     plt.plot(dist/ang2bohr, Psi.El*har2wave, label=f'Local Energy alpha = {alpha[i]}')
# # # # plt.plot(dist/ang2bohr, kin1*har2wave, label='Local Energy from Kin1')
# # # # plt.plot(dist/ang2bohr, kin2*har2wave, label='Local Energy from Kin2')
# # # # plt.plot(dist/ang2bohr, extra*har2wave, label='Extra Term')
# # # plt.plot(dist/ang2bohr, Psi.El*har2wave+Psi.V*har2wave, label='Local Energy')
# # # # plt.plot(dist/ang2bohr, mass_drdx*har2wave, label='Mass weighted drdx')
# # # # plt.plot(dist/ang2bohr, dpsidr*har2wave, label='dPsidr')
# # # # plt.plot(dist/ang2bohr, -Psi.El*har2wave, label='Local Kinetic Energy')
# plt.xlabel('rCH (Angstrom)')
# plt.ylabel('Energy (cm^-1)')
# plt.legend()
# plt.ylim(-2000, 10000)
# plt.savefig('Testing_6_point_imp_samp_testing_alpha.png')

tests = [100, 500, 1000, 5000, 10000]
# energies1 = np.zeros((10, 5))
# energies2 = np.zeros((10, 5))
# for j in range(10):
#     for i in range(5):
#         N_0 = tests[i]
#         interp = interpolate.splrep(Psi_t[0, :], np.array([1.]*len(Psi_t[1, :])), s=0)
#         Energy = run(50)
#         energies1[j, i] += np.mean(Energy[1000:]*har2wave)
#         interp = interpolate.splrep(Psi_t[0, :], Psi_t[1, :], s=0)
#         Energy = run(50)
#         energies2[j, i] += np.mean(Energy[1000:]*har2wave)
#
# np.save('Non_imp_samp_energies', energies1)
# np.save('Imp_samp_energies', energies2)
#
energies1 = np.load('Non_imp_samp_energies.npy')
energies2 = np.load('Imp_samp_energies.npy')

average1 = np.mean(energies1, axis=0)
average2 = np.mean(energies2, axis=0)
std1 = np.std(energies1, axis=0)
std2 = np.std(energies2, axis=0)

fig, axes = plt.subplots(2, 1)
axes[0].errorbar(tests, average1, yerr=std1)
axes[0].plot(tests, np.array([1218.70588162167]*len(tests)))
axes[1].errorbar(tests, average2, yerr=std2)
axes[1].plot(tests, np.array([1218.70588162167]*len(tests)))
axes[0].set_xlabel('Number of Walkers')
axes[1].set_xlabel('Number of Walkers')
axes[0].set_ylabel('Energy (cm^-1)')
axes[1].set_ylabel('Energy (cm^-1)')
axes[0].set_ylim(1200, 1250)
axes[1].set_ylim(1200, 1250)
plt.tight_layout()
fig.savefig('Convergence_two_atom_switch_alpha_5.png')

