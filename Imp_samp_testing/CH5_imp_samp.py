import numpy as np
import copy
import CH5pot
from scipy import interpolate
from Coordinerds.CoordinateSystems import *

# DMC parameters
dtau = 1.
N_0 = 500
time_steps = 1000.
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
# Starting orientation of walkers
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]

Psi_t = np.load('Average_GSW_CH_stretch.npy')
interp = interpolate.splrep(Psi_t[0, :], Psi_t[1, :], s=0)


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
    Psi_t = np.zeros((len(zmatrix[:, 1]), N_0))
    for i in range(len(zmatrix[:, 1])):
        Psi_t[i, :] += interpolate.splev(zmatrix[i, 1], interp, der=0)
    return Psi_t


def drift(zmatrix):
    psi = psi_t(zmatrix)
    der = np.zeros((len(zmatrix[:, 1]), N_0))
    for i in range(len(zmatrix[:, 1])):
        der[i, :] += interpolate.splev(zmatrix[i, 1], interp, der=1)
    return der/psi


def metropolis(x, y, Fqx, Fqy):
    psi_x = psi_t(x)
    psi_y = psi_t(y)
    dof = len(x[:, 1])
    a = np.zeros(dof) + 1.
    for i in range(dof):
        a *= (psi_y[i, :]/psi_x[i, :])**2 * np.exp(1./2.*(Fqx + Fqy)*(sigmaH**2/4.*(Fqx-Fqy) - (y[i, 1]-x[i, 1])))
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx):
    # randomwalkC = np.random.normal(0.0, sigmaC, size=(N_0, 3))
    zmatrix = CoordinateSet(Psi.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    Drift = sigmaCH**2/2.*Fqx
    zmatrix[:, 1] += Drift
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
    # Psi.coords[:, 0, :] += randomwalkC
    return Psi, Fqy


def Potential(Psi):
    V = CH5pot.mycalcpot(Psi.coords, N_0)
    Psi.V = np.array(V)
    return Psi


def sec_dir(coords):
    return interpolate.splev(coords, interp, der=2)


def E_loc(Psi):
    psi = psi_t(Psi.coords)
    kin = -1./(2.*m_CH)*sec_dir(Psi.coords)/psi
    pot = Psi.V
    Psi.El = kin + pot
    return Psi


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
            Psi_tau.d += d_values





