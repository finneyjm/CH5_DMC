import copy
import CH5pot
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import Timing_p3 as tm
import matplotlib.pyplot as plt

# DMC parameters
dtau = 1.
N_0 = 10000
time_steps = 1.
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
sigmaCH = np.sqrt(dtau/m_CH)
bonds = 5
# Starting orientation of walkers
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]

ch_stretch = 4
Psi_t = np.load(f'GSW_min_CH_{ch_stretch+1}.npy')
interp = interpolate.splrep(np.linspace(1, 4, num=500), Psi_t, s=0)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.array([coords_initial]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


def psi_t(zmatrix):
    psi = np.zeros((N_0, bonds))
    for i in range(bonds):
        psi[:, i] += interpolate.splev(zmatrix[:, i, 1], interp, der=0)
    return psi


def drift(zmatrix):
    psi = psi_t(zmatrix)
    der = np.zeros((N_0, bonds))
    for i in range(bonds):
        der[:, i] += interpolate.splev(zmatrix[:, i, 1], interp, der=1)
    return der/psi


def metropolis(x, y, Fqx, Fqy):
    psi_x = psi_t(x)
    psi_y = psi_t(y)
    a = 1.
    for i in range(bonds):
        a *= (psi_y[:, i]/psi_x[:, i])**2 * np.exp(1./2.*(Fqx[:, i] + Fqy[:, i])*(sigmaCH**2/4.*(Fqx[:, i]-Fqy[:, i])
                                                                                  - (y[:, i, 1]-x[:, i, 1])))
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx):
    zmatrix = CoordinateSet(Psi.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    Drift = sigmaCH**2/2.*Fqx
    zmatrix[:, :, 1] += Drift
    y = CoordinateSet(zmatrix, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    randomwalkH = np.zeros((N_0, 6, 3))
    randomwalkH[:, 1:6, :] = np.random.normal(0.0, sigmaH, size=(N_0, 5, 3))
    y += randomwalkH
    zmatriy = CoordinateSet(y, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    Fqy = drift(zmatriy)
    a = metropolis(zmatrix, zmatriy, Fqx, Fqy)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    nah = np.argwhere(a <= check)
    Fqy[nah] = Fqx[nah]
    return Psi, Fqy


def Potential(Psi):
    V = CH5pot.mycalcpot(Psi.coords, N_0)
    Psi.V = np.array(V)
    return Psi


def drdx(coords, zmatrix):
    chain = np.zeros((N_0, 6, 1))
    for j in range(3):
        for l in range(1):
            l = l+4
            chain[:, j, l-4] -= (coords[:, l+1, j]/zmatrix[:, l, 1])
            chain[:, (j + 3)+(3*(l-4)), l-4] += (coords[:, l+1, j]/zmatrix[:, l, 1])
    return chain


def drdx2(coords, zmatrix):
    chain = np.zeros((N_0, 6, 1))
    for j in range(3):
        for l in range(1):
            l += 4
            chain[:, j, l-4] += (1./zmatrix[:, l, 1] - coords[:, l+1, j]**2/zmatrix[:, l, 1]**3)
            chain[:, (j+3)+(3*(l-4)), l-4] += (1./zmatrix[:, l, 1] - coords[:, l+1, j]**2/zmatrix[:, l, 1]**3)
    return chain


def local_kinetic(coords):
    zmatrix = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    psi = psi_t(zmatrix)
    # kin = 0
    # psi_new = np.zeros(N_0)
    # for i in range(bonds):
    #     psi_new *= psi[:, i]





    dr1 = drdx(coords, zmatrix)
    dr2 = drdx2(coords, zmatrix)

    der1 = np.zeros((N_0, 1))
    for i in range(1):
        i += 4
        der1[:, i-4] += (interpolate.splev(zmatrix[:, i, 1], interp, der=1)/psi[:, i])

    der2 = np.zeros((N_0, 1))
    for i in range(1):
        i += 4
        der2[:, i-4] += (interpolate.splev(zmatrix[:, i, 1], interp, der=2)/psi[:, i])

    masses = np.zeros(6)
    for i in range(6):
        if i < 3:
            masses[i] += 1./m_C
        else:
            masses[i] += 1./m_H
    kin1 = np.tensordot(masses, dr1**2, axes=([0], [1]))
    kin1 = np.sum(der2*kin1, axis=1)
    kin2 = np.tensordot(masses, dr2, axes=([0], [1]))
    kin2 = np.sum(der1*kin2, axis=1)
    kin = kin1 + kin2
    # derivatives = np.tensordot(der2, dr1**2, axes=([1], [2]))
    # derivatives += np.tensordot(der1, dr2, axes=([1], [2]))
    # derivatives = np.diagonal(derivatives, 0, 0, 1)
    # kin = np.tensordot(masses, derivatives, axes=([0], [0]))


    # for j in range(bonds):
    # kin += (1./m_CH*der2[:, 4])
    return -1./2.*kin


def E_loc(Psi):
    Psi.El = local_kinetic(Psi.coords) + Psi.V
    return Psi, -1.*local_kinetic(Psi.coords)



Psi = Walkers(N_0)
zmatrix = CoordinateSet(Psi.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
zmatrix[:, ch_stretch, 1] = np.linspace(0.8, 1.4, num=N_0)*ang2bohr
Psi.coords = CoordinateSet(zmatrix, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
Psi, psi_list = tm.time_me(Potential, Psi)
tm.print_time_list(Potential, psi_list)
Psi, kin, psi_list = tm.time_me(E_loc, Psi)
tm.print_time_list(E_loc, psi_list)
plt.plot(zmatrix[:, ch_stretch, 1]/ang2bohr, Psi.V*har2wave, label='Potential')
plt.plot(zmatrix[:, ch_stretch, 1]/ang2bohr, Psi.El*har2wave, label='Local Energy')
# plt.plot(zmatrix[:, ch_stretch, 1]/ang2bohr, kin*har2wave, label='Local Kinetic Energy')
# plt.plot(zmatrix[:, ch_stretch, 1]/ang2bohr, psi_t(zmatrix)[:, ch_stretch]*20000., label='Trial Wavefunction')
plt.ylim(0, 22000)
plt.legend()
plt.savefig('Testing_local_energy.png')


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


def run(propagation):
    DW = False
    psi = Walkers(N_0)
    Fqx = drift(psi.coords)
    Psi, Fqx = Kinetic(psi, Fqx)
    Psi = Potential(Psi)
    Psi = E_loc(Psi)
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, DW)

    Psi_tau = 0
    for i in range(int(time_steps)):
        Psi, Fqx = Kinetic(new_psi, Fqx)
        Psi = Potential(Psi)
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
    return DW


# dw, time = tm.time_me(run, 0)
# tm.print_time_list(run, time)

