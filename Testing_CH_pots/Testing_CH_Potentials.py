import numpy as np
import matplotlib.pyplot as plt
from Coordinerds.CoordinateSystems import *
import copy
from scipy import interpolate
# import Timing_p3 as tm

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


def run(propagation, CH, type, name):
    Psi_t = np.load('Switch_c2v_wvfn_%s.npy' %type)
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
        if i % 1000 == 0:
            print(i)
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
    np.save('Imp_samp_CH_pots_Psi_c2v%s' %(type + CH + name), wvfn)
    np.save('Imp_samp_CH_pots_Energy_c2v%s' %(type + CH + name), Eref_array)
    return Eref_array


Energy1 = np.zeros((6, 5, 7))
Energy2 = np.zeros((6, 5, 7))
Energy3 = np.zeros((6, 5, 7))
for j in range(6):
    for i in range(5):
        for l in range(7):
            energy1 = run(50., str(i+1), str(1.0 + 0.05*float(l)), '_job' + str(j+1))
            print('min CH stretch ' + str(i+1) + ' with ' + str(1.0 + 0.05*float(l)) + ' switch point job ' + str(j+1) + ' is done!')

            energy2 = run(50., '_cs_saddle' + str(i+1), str(1.0 + 0.05*float(l)), '_job' + str(j+1))
            print('cs CH stretch ' + str(i+1) + ' with ' + str(1.0 + 0.05*float(l)) + ' switch point job ' + str(j+1) + ' is done!')

            energy3 = run(50., '_c2v_saddle' + str(i+1), str(1.0 + 0.05*float(l)), '_job' + str(j+1))
            print('c2v CH stretch ' + str(i+1) + ' with ' + str(1.0 + 0.05*float(l)) + ' switch point job ' + str(j+1) + ' is done!')

            # save the mean energies to be entered into the table later
            Energy1[j, i, l] += np.mean(energy1[500:])
            Energy2[j, i, l] += np.mean(energy2[500:])
            Energy3[j, i, l] += np.mean(energy3[500:])


    #     run(50., str(i + 1), '', '_job' + str(j + 1))
    #     print('min CH stretch ' + str(i + 1) + ' job ' + str(j+1) + ' with min GSW is done!')
    #     run(50., '_cs_saddle' + str(i + 1), '', '_job' + str(j + 1))
    #     print('cs CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with min GSW is done!')
    #     run(50., '_c2v_saddle' + str(i + 1), '', '_job' + str(j + 1))
    #     print('c2v CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with min GSW is done!')
    # for i in range(5):
    #     run(50., str(i + 1), '_cs_saddle', '_job' + str(j + 1))
    #     print('min CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with cs GSW is done!')
    #     run(50., '_cs_saddle' + str(i + 1), '_cs_saddle', '_job' + str(j + 1))
    #     print('cs CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with cs GSW is done!')
    #     run(50., '_c2v_saddle' + str(i + 1), '_cs_saddle', '_job' + str(j + 1))
    #     print('c2v CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with cs GSW is done!')
    # for i in range(5):
    #     run(50., str(i + 1), '_c2v_saddle', '_job' + str(j + 1))
    #     print('min CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with c2v GSW is done!')
    #     run(50., '_cs_saddle' + str(i + 1), '_c2v_saddle','_job' + str(j + 1))
    #     print('cs CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with c2v GSW is done!')
    #     run(50., '_c2v_saddle' + str(i + 1), '_c2v_saddle', '_job' + str(j + 1))
    #     print('c2v CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with c2v GSW is done!')
    #
    # for i in range(5):
    #     energy1 = np.load('Imp_samp_CH_pots_Energy_' + str(i + 1) + '_job' + str(j + 1) + '.npy')
    #     Energy1[j, i, 0] = np.mean(energy1[500:])
    #     energy2 = np.load('Imp_samp_CH_pots_Energy_%s.npy' % ('_cs_saddle' + str(i + 1) + '_job' + str(j + 1)))
    #     Energy2[j, i, 0] = np.mean(energy2[500:])
    #     energy3 = np.load('Imp_samp_CH_pots_Energy_%s.npy' % ('_c2v_saddle' + str(i + 1) + '_job' + str(j + 1)))
    #     Energy3[j, i, 0] = np.mean(energy3[500:])
    #
    # for i in range(5):
    #     energy1 = np.load('Imp_samp_CH_pots_Energy__cs_saddle' + str(i + 1) + '_job' + str(j + 1) + '.npy')
    #     Energy1[j, i, 1] = np.mean(energy1[500:])
    #     energy2 = np.load('Imp_samp_CH_pots_Energy__cs_saddle%s.npy' % ('_cs_saddle' + str(i + 1) + '_job' + str(j + 1)))
    #     Energy2[j, i, 1] = np.mean(energy2[500:])
    #     energy3 = np.load('Imp_samp_CH_pots_Energy__cs_saddle%s.npy' % ('_c2v_saddle' + str(i + 1) + '_job' + str(j + 1)))
    #     Energy3[j, i, 1] = np.mean(energy3[500:])
    #
    # for i in range(5):
    #     energy1 = np.load('Imp_samp_CH_pots_Energy_c2v' + str(i + 1) + '_job' + str(j + 1) + '.npy')
    #     Energy1[j, i, 2] = np.mean(energy1[500:])
    #     energy2 = np.load('Imp_samp_CH_pots_Energy_c2v%s.npy' % ('_cs_saddle' + str(i + 1) + '_job' + str(j + 1)))
    #     Energy2[j, i, 2] = np.mean(energy2[500:])
    #     energy3 = np.load('Imp_samp_CH_pots_Energy_c2v%s.npy' % ('_c2v_saddle' + str(i + 1) + '_job' + str(j + 1)))
    #     Energy3[j, i, 2] = np.mean(energy3[500:])

np.save('Imp_min_energies_c2v', Energy1)
np.save('Imp_cs_energies_c2v', Energy2)
np.save('Imp_c2v_energies_c2v', Energy3)



# DW, time_list = tm.time_me(run, 50., str(1), '', 'test')
# tm.print_time_list(run, time_list)
#
# energy1 = np.load('Imp_min_energies.npy')*har2wave
# energy2 = np.load('Imp_cs_energies.npy')*har2wave
# energy3 = np.load('Imp_c2v_energies.npy')*har2wave
# #
# # print(energy1[:, :, 0])
# # print(energy2[:, :, 0])
# # print(energy3[:, :, 0])
#
# for i in range(5):
#     print(energy2[5, i, 0])





