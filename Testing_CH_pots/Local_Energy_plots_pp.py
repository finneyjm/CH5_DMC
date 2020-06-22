from Coordinerds.CoordinateSystems import *
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
ang2bohr = 1.e-10/5.291772106712e-11

# Values for Simulation
sigmaH = np.sqrt(dtau/m_H)
sigmaC = np.sqrt(dtau/m_C)
sigmaCH = np.sqrt(dtau/m_CH)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.linspace(0, walkers-1, num=walkers)
        self.coords = np.array([0]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


def psi_t(coords, int):
    return interpolate.splev(coords, int, der=0)


def sec_dir(coords, int):
    return interpolate.splev(coords, int, der=2)


def Potential(Psi, CH):
    return interpolate.splev(Psi.coords, CH, der=0)


def E_loc(Psi, int):
    psi = psi_t(Psi.coords, int)
    kin = -1./(2.*m_CH)*sec_dir(Psi.coords, int)/psi
    return kin + Psi.V, psi


def let_do_some_plotting():
    Psi = Walkers(N_0)
    Psi.coords = np.linspace(0.7, 1.75, N_0)*ang2bohr
    grid = np.linspace(0.4, 6, 5000)
    GSW = np.zeros((3, 5, 5000))
    for i in range(5):
        GSW[0, i, :] += np.load(f'GSW_min_CH_{i + 1}.npy')
    avg_wvfn = np.mean(GSW[0], axis=0)/np.linalg.norm(np.mean(GSW[0], axis=0))
    psi = np.zeros((3, 5000))
    psi[0] = GSW[0, -1]
    psi[1] = GSW[0, 2]
    psi[2] = avg_wvfn
    pot = interpolate.splrep(grid, np.load('Potential_CH_stretch2.npy'), s=0)
    Psi.V = Potential(Psi, pot)
    # np.savetxt('Potential_CH_stretch_5.csv', Psi.V*har2wave, delimiter=',')
    fig, axes = plt.subplots()
    axes.plot(Psi.coords/ang2bohr, Psi.V*har2wave, linewidth=3, color='black', label='V')
    # axes.set_ylim(-10000, 10000)
    axes.set_xlabel(r'r$_{\rm{CH}^{(2)}}$ ($\rm\AA$)', fontsize=22)
    axes.set_ylabel(r'E$_0$(cm$^{-1}$)', fontsize=22)
    # axes.legend(loc='lower right')
    fig.savefig('Local_energy_powerpoint_potential_1.png')
    colors = ['red', 'orange', 'indigo']
    labels = [r'E$_L^{(\rm{CH}^{\rm{5}})}$', r'E$_L^{(\rm{CH}^{\rm{2}})}$', r'E$_L^{(\rm{CH}^{\rm{avg}})}$']
    line = ['--', 'solid', 'dotted']
    for i in range(3):
        interp = interpolate.splrep(grid, psi[i, :], s=0)
        Psi.El, wvfn = E_loc(Psi, interp)
        axes.plot(Psi.coords/ang2bohr, Psi.El*har2wave, linewidth=3, color=colors[i], linestyle=line[i], label=labels[i])
    axes.set_xlim(0.8, 1.6)
    axes.set_ylim(-7500, 10000)
    leg = plt.legend(loc='lower left', fontsize=14)
    leg.get_frame().set_edgecolor('white')
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


let_do_some_plotting()

