import copy
import CH5pot
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
# import Timing_p3 as tm
import matplotlib.pyplot as plt
import multiprocessing as mp

# DMC parameters
dtau = 1.
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
sigmaCH = np.array([[sigmaC]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3])
bonds = 5
# Starting orientation of walkers
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]


# ch_stretch = 4
# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial]*walkers)
        # rand_idx = np.random.rand(walkers, 5).argsort(axis=1) + 1
        # b = self.coords[np.arange(walkers)[:, None], rand_idx]
        # self.coords[:, 1:6, :] = b
        self.zmat = CoordinateSet(self.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


def psi_t(zmatrix, interp):
    psi = np.zeros((len(zmatrix), bonds))
    for i in range(bonds):
        psi[:, i] = interpolate.splev(zmatrix[:, i, 1], interp, der=0)
    return psi


def drdx(zmatrix, coords):
    chain = np.zeros((len(coords), 5, 6, 3))
    for xyz in range(3):
        for CH in range(bonds):
            chain[:, CH, 0, xyz] = ((coords[:, 0, xyz]-coords[:, CH+1, xyz])/zmatrix[:, CH, 1])
            chain[:, CH, CH+1, xyz] = ((coords[:, CH+1, xyz]-coords[:, 0, xyz])/zmatrix[:, CH, 1])
    return chain


def drdx2(zmatrix, coords):
    chain = np.zeros((len(coords), 5, 6, 3))
    for xyz in range(3):
        for CH in range(bonds):
            chain[:, CH, 0, xyz] = (1./zmatrix[:, CH, 1] - (coords[:, 0, xyz]-coords[:, CH+1, xyz])**2/zmatrix[:, CH, 1]**3)
            chain[:, CH, CH + 1, xyz] = (1./zmatrix[:, CH, 1] - (coords[:, CH + 1, xyz] - coords[:, 0, xyz])**2 / zmatrix[:, CH, 1]**3)
    return chain


def dpsidx2(zmatrix, coords, interp):
    psi = psi_t(zmatrix, interp)
    dr1 = drdx(zmatrix, coords)
    dr2 = drdx2(zmatrix, coords)
    der1 = np.zeros((len(coords), bonds))
    der2 = np.zeros((len(coords), bonds))
    for i in range(bonds):
        der1[:, i] = (interpolate.splev(zmatrix[:, i, 1], interp, der=1)/psi[:, i])
        der2[:, i] = (interpolate.splev(zmatrix[:, i, 1], interp, der=2)/psi[:, i])
    c = dr1.reshape((len(coords), 5, 18))**2
    b = der1.reshape((len(coords), 1, 5))
    a = dr2.reshape((len(coords), 5, 18))
    d = der2.reshape((len(coords), 1, 5))
    pt1 = np.matmul(b, a)
    pt1 = pt1.reshape((len(coords), 6, 3))
    pt2 = np.matmul(d, c)
    pt2 = pt2.reshape((len(coords), 6, 3))
    blah = pt1 + pt2
    pt3 = np.zeros((len(coords), 5, 5, 3))
    for i in range(bonds):
        for j in range(i+1, bonds):
            for xyz in range(3):
                pt3[:, i, j, xyz] = der1[:, j]*dr1[:, j, 0, xyz]*der1[:, i]*dr1[:, i, 0, xyz]
    pt3 += np.transpose(pt3, (0, 2, 1, 3))
    pt3 = np.sum(np.sum(pt3, axis=1), axis=1)
    blah[:, 0] += pt3
    blah2 = -1./2.*(np.sum(np.sum(sigmaCH**2/dtau*blah, axis=1), axis=1))
    return blah2


def drift(zmatrix, coords, interp):
    psi = psi_t(zmatrix, interp)
    dr1 = drdx(zmatrix, coords)
    der = np.zeros((len(coords), bonds))
    for i in range(bonds):
        der[:, i] += (interpolate.splev(zmatrix[:, i, 1], interp, der=1)/psi[:, i])
    a = dr1.reshape((len(coords), 5, 18))
    b = der.reshape((len(coords), 1, 5))
    drift = np.matmul(b, a)
    return 2.*drift.reshape((len(coords), 6, 3)), dr1


def metropolis(r1, r2, Fqx, Fqy, x, y, interp):
    psi_1 = psi_t(r1, interp)
    psi_2 = psi_t(r2, interp)
    psi_ratio = 1.
    for i in range(bonds):
        psi_ratio *= (psi_2[:, i]/psi_1[:, i])**2
    a = psi_ratio
    for atom in range(6):
        for xyz in range(3):
            if atom == 0:
                sigma = sigmaC
            else:
                sigma = sigmaH
            a *= np.exp(1./2.*(Fqx[:, atom, xyz] + Fqy[:, atom, xyz])*(sigma**2/4.*(Fqx[:, atom, xyz]-Fqy[:, atom, xyz])
                                                                       - (y[:, atom, xyz]-x[:, atom, xyz])))
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, interp):
    Drift = sigmaCH**2/2.*Fqx
    randomwalk = np.zeros((len(Psi.coords), 6, 3))
    randomwalk[:, 1:6, :] = np.random.normal(0.0, sigmaH, size=(len(Psi.coords), 5, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaC, size=(len(Psi.coords), 3))
    y = randomwalk + Drift + np.array(Psi.coords)
    zmatriy = CoordinateSet(y, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    Fqy = drift(zmatriy, y, interp)
    a = metropolis(Psi.zmat, zmatriy, Fqx, Fqy, Psi.coords, y, interp)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    nah = np.argwhere(a <= check)
    Fqy[nah] = Fqx[nah]
    Psi.zmat[accept] = zmatriy[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqy, acceptance


# Function for the potential for the mp to use
def get_pot(coords):
    V = CH5pot.mycalcpot(coords, len(coords))
    return V


# Split up those coords to speed up dat potential
def Potential(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    # Psi.V = np.array(CH5pot.mycalcpot(Psi.coords, len(Psi.coords)))
    return Psi


def local_kinetic(Psi, interp, dr1):
    psi = psi_t(Psi.zmat, interp)
    der1 = np.zeros((len(Psi.coords), bonds))
    der2 = np.zeros((len(Psi.coords), bonds))
    for i in range(bonds):
        der1[:, i] += (interpolate.splev(Psi.zmat[:, i, 1], interp, der=1)/psi[:, i]*(2./Psi.zmat[:, i, 1]))
        der2[:, i] += (interpolate.splev(Psi.zmat[:, i, 1], interp, der=2)/psi[:, i])
    kin = -1./(2.*m_CH)*np.sum(der2+der1, axis=1)
    a = dr1[:, :, 0]*np.broadcast_to(der1[:, :, None], (len(Psi.coords), 5, 3))
    carb_correct = np.sum(np.sum(a, axis=1)**2-np.sum(a**2, axis=1), axis=1)
    kin += -1./(2.*m_C)*carb_correct
    return kin


def E_loc_test(Psi, interp, dr1):
    Psi.El = local_kinetic(Psi, interp, dr1) + Psi.V
    return Psi


def E_loc(Psi, interp):
    Psi.El = dpsidx2(Psi.zmat, Psi.coords, interp) + Psi.V
    return Psi


def E_ref_calc(Psi):
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/len(Psi.coords))
    return E_ref


def Weighting(Eref, Psi, DW, Fqx):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    # print(np.max(Psi.weights))
    threshold = 1./float(len(Psi.coords))
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
        Biggo_zmat = np.array(Psi.zmat[ind])
        Biggo_force = np.array(Fqx[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Psi.zmat[i[0]] = Biggo_zmat
        Fqx[i[0]] = Biggo_force
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


def run(N_0, time_steps, propagation, test_number, bro):
    Psi_t = np.load(f'min_wvfns/Average_min_broadening_{bro}x.npy')
    interp = interpolate.splrep(Psi_t[0, :], Psi_t[1, :], s=0)
    # interp = interpolate.splrep(Psi_t[0, :], np.array([1.]*len(Psi_t[1, :])), s=0)
    DW = False
    psi = Walkers(N_0)
    Fqx = drift(psi.zmat, psi.coords, interp)
    Psi, Fqx, acceptance = Kinetic(psi, Fqx, interp)
    Psi = Potential(Psi)
    Psi = E_loc(Psi, interp)
    time = np.array([])
    weights = np.array([])
    accept = np.array([])
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, DW, Fqx)
    time = np.append(time, 1)
    weights = np.append(weights, np.sum(new_psi.weights))
    accept = np.append(accept, acceptance)

    Psi_tau = 0
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)

        Psi, Fqx, acceptance = Kinetic(new_psi, Fqx, interp)
        Psi = Potential(Psi)
        Psi = E_loc(Psi, interp)
        new_psi = Weighting(Eref, Psi, DW, Fqx)

        if DW is False:
            prop = float(propagation)
        elif DW is True:
            prop -= 1.
            if Psi_tau == 0:
                Psi_tau = copy.deepcopy(Psi)

        Eref = E_ref_calc(new_psi)
        Eref_array = np.append(Eref_array, Eref)
        time = np.append(time, 2 + i)
        weights = np.append(weights, np.sum(new_psi.weights))
        accept = np.append(accept, acceptance)

        if i >= (time_steps - 1. - float(propagation)) and prop > 0.:
            DW = True
        elif i >= (time_steps - 1. - float(propagation)) and prop == 0.:
            d_values = descendants(new_psi)
    # np.save(f'Trial_wvfn_testing/broad_{bro}/coords/Imp_samp_DMC_CH5_coords_min_avg_{N_0}_walkers_{test_number}', Psi_tau.coords)
    # np.save(f'Trial_wvfn_testing/broad_{bro}/weights/Imp_samp_DMC_CH5_weights_min_avg_{N_0}_walkers_{test_number}', np.vstack((Psi_tau.weights, d_values)))
    # np.save(f'Trial_wvfn_testing/broad_{bro}/energies/Imp_samp_CH5_energy_min_avg_{N_0}_walkers_{test_number}', np.vstack((time, Eref_array, weights, accept)))
    # np.save(f'Non_imp_sampled/DMC_CH5_Energy_{N_0}_walkers_{test_number}.npy', np.vstack((time, Eref_array, weights, accept)))
    # np.save(f'Non_imp_sampled/DMC_CH5_coords_{N_0}_walkers_{test_number}.noy', Psi_tau.coords)
    # np.save(f'Non_imp_sampled/DMC_CH5_weights_{N_0}_walkers_{test_number}.npy', np.vstack((Psi_tau.weights, d_values)))
    np.savez(f'Trial_wvfn_testing/broad_{bro}/all_da_things/Imp_samp_DMC_min_avg_{N_0}_walkers_{test_number}', coords=Psi_tau.coords, weights=Psi_tau.weights,
             descendants=d_values, time=time, Eref=Eref_array, sum_weights=weights, accept=accept)
    return Eref_array

pool = mp.Pool(mp.cpu_count()-1)
# N_0 = 100
# # run(250, 0, 0)
# run(100, 100, 25, 1, 2.0)
# tests = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
# # for j in range(5):
# j = 3
# for i in range(len(tests)):
#     run(tests[i], 20000, 250, j+1, 1.1)
#     print(f'{tests[i]} Walker Test {j+1} is done!')
# broad = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1]
# # for j in range(5):
# #     b = 8
# j = 0
# for i in range(8):
#     N_0 = tests[i]
#     run(250, j+1, 1.1)
#     print(f'{tests[i]} Walker Test {j+1} is done!')

N_0 = 10000
# # alpha_test = [1, 2, 3, 4, 5, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
# # for j in range(9):
# bro = 1.0
Psi_t = np.load(f'params/Switch_wvfns/Switch_min_wvfn_speed_1.0.npy')
x = np.linspace(0.4, 6., 5000)
# Psi_t = np.ones(5000)
interp = interpolate.splrep(x, Psi_t[1, :], s=0)
Psi = Walkers(N_0)
Psi.coords = np.load('Non_imp_sampled/DMC_CH5_coords_10000_walkers_5.npy')
Psi.zmat = CoordinateSet(Psi.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
f, dr1 = drift(Psi.zmat, Psi.coords, interp)
import Timing_p3 as tm
testytest, testy_time = tm.time_me(dpsidx2, Psi.zmat, Psi.coords, interp)
tm.print_time_list(dpsidx2, testy_time)
Psi = Potential(Psi)
Psi = E_loc(Psi, interp)
Psi2 = E_loc_test(Psi, interp, dr1)
psi, E_time = tm.time_me(E_loc, Psi, interp)
tm.print_time_list(E_loc, E_time)
psi2, E_time = tm.time_me(E_loc_test, Psi2, interp, dr1)
tm.print_time_list(E_loc_test, E_time)
blah = 4
# import matplotlib.pyplot as plt
# plt.plot(Psi.El*har2wave)
# plt.plot(Psi.V*har2wave)
# plt.show()
# psi_t2 = np.load(f'Switch_min_wvfn_speed_1.0.npy')
# interp2 = interpolate.splrep(psi_t2[0, :], psi_t2[1, :], s=0)
# fig, axes = plt.subplots(1, 5, figsize=(20, 8))
# for i in range(5):
#     Psi = Walkers(N_0)
#     Psi.zmat[:, i, 1] = np.linspace(0.6, 1.8, N_0)*ang2bohr
#     Psi.coords = CoordinateSet(Psi.zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     Psi = Potential(Psi)
#     Psi = E_loc(Psi, interp)
#     # psi = psi_t(Psi.zmat, interp)
#     axes[i].plot(Psi.zmat[:, i, 1]/ang2bohr, Psi.V*har2wave, label='Potential')
#     # axes[i].plot(Psi.zmat[:, i, 1]/ang2bohr, psi[:, i], label='Psi t')
#     axes[i].plot(Psi.zmat[:, i, 1]/ang2bohr, Psi.El*har2wave, label=f'Local Energy')
#     axes[i].plot(Psi.zmat[:, i, 1] / ang2bohr, Psi.El * har2wave - Psi.V*har2wave, label=f'Kinetic')
#     if i is 0:
#         asdf = np.load('asdfasdfasdf.npy')
#         blah = asdf/(Psi.El-Psi.V)
#         one = 1
#     # axes[i].plot(Psi.coords[:, ])
#     # loc_1 = np.array(Psi.El)
#     # Psi = E_loc(Psi, interp2)
#     # diff = Psi.El - loc_1
#     # axes[i].plot(Psi.zmat[:, ch-1, 1]/ang2bohr, diff*har2wave)
#     # axes[i].plot(Psi.zmat[:, ch-1, 1]/ang2bohr, Psi.El*har2wave, label=f'Local Energy no fit')
#     # axes[i].plot(Psi.zmat[:, ch-1, 1]/ang2bohr, Psi.El*har2wave - Psi.V*har2wave, label='Kinetic no fit')
#     # psi = psi_t(Psi.zmat, interp)
#     # # axes[i].plot(Psi.zmat[:, ch-1, 1]/ang2bohr, psi[:, ch-1], label='Gaussians')
#     # psi1 = psi_t(Psi.zmat, interp2)
#     # diff = psi1-psi
#     # axes[i].plot(Psi.zmat[:, ch-1, 1]/ang2bohr, Psi.El, label='difference')
#     axes[i].set_xlabel('rCH (Angstrom)')
#     # axes[i].set_xlim(0.8, 1.6)
#     # axes[i].set_ylim(-5, 5)
#     axes[i].set_ylabel('Energy (cm^-1)')
#     axes[i].set_ylim(-20000, 20000)
#     axes[i].legend(loc='lower left')
# plt.tight_layout()
# fig.savefig(f'local_energy_plots/Average_min_broadening_{bro}x.png')
# plt.show()
# plt.close()
