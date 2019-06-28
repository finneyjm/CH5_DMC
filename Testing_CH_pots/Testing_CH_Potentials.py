import numpy as np
import matplotlib.pyplot as plt
from Coordinerds.CoordinateSystems import *
import copy
from scipy import interpolate

# DMC parameters
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
har2wave = 219474.6

# Values for Simulation
sigmaH = np.sqrt(dtau/m_H)
sigmaC = np.sqrt(dtau/m_C)
sigmaCH = np.sqrt(dtau/m_CH)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, min):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.array([min]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


def psi_t(coords, int):
    return interpolate.splev(coords, int, der=0)


def drift(coords, int):
    psi = psi_t(coords, int)
    return 2.*interpolate.splev(coords, int, der=1)/psi


def sec_dir(coords, int):
    return interpolate.splev(coords, int, der=2)


def metropolis(x, y, Fqx, Fqy, int):
    psi_x = psi_t(x, int)
    psi_y = psi_t(y, int)
    pre_factor = (psi_y/psi_x)**2
    return pre_factor*np.exp(1./2.*(Fqx + Fqy)*(sigmaCH**2/4.*(Fqx-Fqy) - (y-x)))


# Random walk of all the walkers
def Kinetic(Psi, Fqx, int):
    randomwalkCH = np.random.normal(0.0, sigmaCH, size=N_0)
    Drift = sigmaCH**2/2.*Fqx
    y = Psi.coords + randomwalkCH + Drift
    Fqy = drift(y, int)
    a = metropolis(Psi.coords, y, Fqx, Fqy, int)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    nah = np.argwhere(a <= check)
    Fqy[nah] = Fqx[nah]
    return Psi, Fqy


def Potential(Psi, CH):
    return interpolate.splev(Psi.coords, CH, der=0)


def E_loc(Psi, int):
    psi = psi_t(Psi.coords, int)
    kin = -1./(2.*m_CH)*sec_dir(Psi.coords, int)/psi
    return kin + Psi.V


def E_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/P0)
    return E_ref


def Weighting(Eref, Psi, DW):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    threshold = 1./float(N_0)
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        if DW is True:
            Biggo_num = float(Psi.walkers[ind])
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
    for i in range(N_0):
        Psi.d[i] = np.sum(Psi.weights[Psi.walkers == i])
    return Psi.d


def run(propagation, CH, type):
    Psi_t = np.load('Average_GSW_CH_stretch%s.npy' %type)
    interp = interpolate.splrep(Psi_t[0, :], Psi_t[1, :], s=0)
    DW = False
    pot = interpolate.splrep(Psi_t[0, :], np.load('Potential_CH_stretch%s.npy' %CH), s=0)
    min = np.argmin(np.load('Potential_CH_stretch%s.npy' %CH))
    psi = Walkers(N_0, Psi_t[0, min])
    Fqx = drift(psi.coords, interp)
    Psi, Fqx = Kinetic(psi, Fqx, interp)
    Psi.V = Potential(Psi, pot)
    Psi.El = E_loc(Psi, interp)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, DW)

    Psi_tau = 0
    for i in range(int(time_steps)):
        Psi, Fqx = Kinetic(new_psi, Fqx, interp)
        Psi.V = Potential(Psi, pot)
        Psi.El = E_loc(Psi, interp)

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
            Psi_tau.d += d_values
    wvfn = np.zeros((3, N_0))
    wvfn[0, :] += Psi_tau.coords
    wvfn[1, :] += Psi_tau.weights
    wvfn[2, :] += Psi_tau.d
    np.save('Imp_samp_CH_pots_Psi_c2v%s' %CH, wvfn)
    np.save('Imp_samp_CH_pots_Energy_c2v%s' %CH, Eref_array)


# for i in range(5):
#     run(50., (i+1), '')
#
# for i in range(5):
#     run(50., (i+1), '')
#     run(50., '_cs_saddle' + str(i+1), '')
#     run(50., '_c2v_saddle' + str(i+1), '')
# for i in range(5):
#     run(50., (i+1), '_cs_saddle')
#     run(50., '_cs_saddle' + str(i + 1), '_cs_saddle')
#     run(50., '_c2v_saddle' + str(i+1), '_cs_saddle')
# for i in range(5):
#     run(50., (i+1), '_c2v_saddle')
#     run(50., '_cs_saddle' + str(i + 1), '_c2v_saddle')
#     run(50., '_c2v_saddle' + str(i+1), '_c2v_saddle')

for i in range(5):
    Energy1 = np.load('Imp_samp_CH_pots_Energy_c2v%s.npy' %(i+1))
    Energy2 = np.load('Imp_samp_CH_pots_Energy_c2v%s.npy' %('_cs_saddle' + str(i+1)))
    Energy3 = np.load('Imp_samp_CH_pots_Energy_c2v%s.npy' %('_c2v_saddle' + str(i+1)))

    plt.figure()
    plt.plot(Energy1[:1000]*har2wave)
    print(str(np.mean(Energy1[400:]*har2wave)) + '+/-' + str(np.std(Energy1[400:]*har2wave)))
    plt.savefig('Imp_samp_CH_energy_c2v%s.png' %(i+1))

    plt.figure()
    plt.plot(Energy2[:1000]*har2wave)
    print(str(np.mean(Energy2[400:]*har2wave)) + '+/-' + str(np.std(Energy2[400:]*har2wave)))
    plt.savefig('Imp_samp_CH_energy_c2v%s.png' %('_cs_saddle' + str(i+1)))

    plt.figure()
    plt.plot(Energy3[:1000]*har2wave)
    print(str(np.mean(Energy3[400:]*har2wave)) + '+/-' + str(np.std(Energy3[400:]*har2wave)))
    plt.savefig('Imp_samp_CH_energy_c2v%s.png' %('_c2v_saddle' + str(i+1)))













