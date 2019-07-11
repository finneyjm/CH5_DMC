import numpy as np
import copy
from scipy import interpolate
import matplotlib.pyplot as plt
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


def Kinetic(Psi):
    randomwalk = np.random.normal(0.0, sigmaCH, size=N_0)
    Psi.coords += randomwalk
    return Psi


def Potential(Psi, CH):
    return interpolate.splev(Psi.coords, CH, der=0)


def E_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.V)/P - alpha*np.log(P/P0)
    return E_ref


def Weighting(Eref, Psi, DW):
    Psi.weights = Psi.weights * np.exp(-(Psi.V - Eref) * dtau)
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
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
    return Psi


def descendants(Psi):
    for i in range(N_0):
        Psi.d[i] = np.sum(Psi.weights[Psi.walkers == i])
    return Psi.d


def run(propagation, CH, type, name):
    Psi_t = np.load('Average_GSW_CH_stretch%s.npy' %type)
    DW = False
    pot = interpolate.splrep(Psi_t[0, :], np.load('Potential_CH_stretch%s.npy' %CH), s=0)
    min = np.argmin(np.load('Potential_CH_stretch%s.npy' %CH))
    psi = Walkers(N_0, Psi_t[0, min])
    Psi = Kinetic(psi)
    Psi.V = Potential(Psi, pot)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, DW)

    Psi_tau = 0
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)
        Psi = Kinetic(new_psi)
        Psi.V = Potential(Psi, pot)

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
    np.save('non_Imp_samp_CH_pots_Psi_%s' %(type + CH + name), wvfn)
    np.save('non_Imp_samp_CH_pots_Energy_%s' %(type + CH + name), Eref_array)
    return DW


# Energy1 = np.zeros((6, 5, 3))
# Energy2 = np.zeros((6, 5, 3))
# Energy3 = np.zeros((6, 5, 3))
# for j in range(6):
#     for i in range(5):
# #         run(50., str(i + 1), '', '_job' + str(j + 1))
# #         print('min CH stretch ' + str(i + 1) + ' job ' + str(j+1) + ' with min GSW is done!')
#         run(50., '_cs_saddle' + str(i + 1), '', '_job' + str(j + 1))
#         print('cs CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with min GSW is done!')
# #         run(50., '_c2v_saddle' + str(i + 1), '', '_job' + str(j + 1))
# #         print('c2v CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with min GSW is done!')
# #     for i in range(5):
# #         run(50., str(i + 1), '_cs_saddle', '_job' + str(j + 1))
# #         print('min CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with cs GSW is done!')
# #         run(50., '_cs_saddle' + str(i + 1), '_cs_saddle', '_job' + str(j + 1))
# #         print('cs CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with cs GSW is done!')
# #         run(50., '_c2v_saddle' + str(i + 1), '_cs_saddle', '_job' + str(j + 1))
# #         print('c2v CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with cs GSW is done!')
# # #     for i in range(5):
# # #         run(50., str(i + 1), '_c2v_saddle', '_job' + str(j + 1))
# # #         print('min CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with c2v GSW is done!')
# # #         run(50., '_cs_saddle' + str(i + 1), '_c2v_saddle','_job' + str(j + 1))
# # #         print('cs CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with c2v GSW is done!')
# # #         run(50., '_c2v_saddle' + str(i + 1), '_c2v_saddle', '_job' + str(j + 1))
# # #         print('c2v CH stretch ' + str(i + 1) + ' job ' + str(j + 1) + ' with c2v GSW is done!')
# # #
#     for i in range(5):
#         energy1 = np.load('non_Imp_samp_CH_pots_Energy_' + str(i + 1) + '_job' + str(j + 1) + '.npy')
#         Energy1[j, i, 0] = np.mean(energy1[500:])
#         energy2 = np.load('non_Imp_samp_CH_pots_Energy_%s.npy' % ('_cs_saddle' + str(i + 1) + '_job' + str(j + 1)))
#         Energy2[j, i, 0] = np.mean(energy2[500:])
#         energy3 = np.load('non_Imp_samp_CH_pots_Energy_%s.npy' % ('_c2v_saddle' + str(i + 1) + '_job' + str(j + 1)))
#         Energy3[j, i, 0] = np.mean(energy3[500:])
#
#     for i in range(5):
#         energy1 = np.load('non_Imp_samp_CH_pots_Energy__cs_saddle' + str(i + 1) + '_job' + str(j + 1) + '.npy')
#         Energy1[j, i, 1] = np.mean(energy1[500:])
#         energy2 = np.load('non_Imp_samp_CH_pots_Energy__cs_saddle%s.npy' % ('_cs_saddle' + str(i + 1) + '_job' + str(j + 1)))
#         Energy2[j, i, 1] = np.mean(energy2[500:])
#         energy3 = np.load('non_Imp_samp_CH_pots_Energy__cs_saddle%s.npy' % ('_c2v_saddle' + str(i + 1) + '_job' + str(j + 1)))
#         Energy3[j, i, 1] = np.mean(energy3[500:])
#
#     for i in range(5):
#         energy1 = np.load('non_Imp_samp_CH_pots_Energy_c2v' + str(i + 1) + '_job' + str(j + 1) + '.npy')
#         Energy1[j, i, 2] = np.mean(energy1[500:])
#         energy2 = np.load('non_Imp_samp_CH_pots_Energy_c2v%s.npy' % ('_cs_saddle' + str(i + 1) + '_job' + str(j + 1)))
#         Energy2[j, i, 2] = np.mean(energy2[500:])
#         energy3 = np.load('non_Imp_samp_CH_pots_Energy_c2v%s.npy' % ('_c2v_saddle' + str(i + 1) + '_job' + str(j + 1)))
#         Energy3[j, i, 2] = np.mean(energy3[500:])
#
# np.save('non_Imp_min_energies', Energy1)
# np.save('non_Imp_cs_energies', Energy2)
# np.save('non_Imp_c2v_energies', Energy3)


# DW, time_list = tm.time_me(run, 50., str(1), '', 'test')
# tm.print_time_list(run, time_list)


energy1 = np.load('non_Imp_min_energies.npy')*har2wave
energy2 = np.load('non_Imp_cs_energies.npy')*har2wave
energy3 = np.load('non_Imp_c2v_energies.npy')*har2wave

for i in range(5):
    print(energy2[4, i, 0])

