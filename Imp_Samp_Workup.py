import numpy as np
import matplotlib.pyplot as plt
import matplotlib

har2wave = 219474.6
# ang2bohr = (1.e-10)/(5.291772106712e-11)


def do_the_things(pot):
    Wvfn = np.load('Imp_samp_%s_Psi.npy' %pot)
    Energy = np.load('Imp_samp_%s_energy.npy' %pot)
    # HOwvfn = np.load('Ground_state_wavefunction_%s.npy' %pot)

    class wvfn(object):

        def __init__(self, Psi):
            self.coords = Psi[0, :]
            self.weights = Psi[1, :]
            self.d = Psi[2, :]


    psi = wvfn(Wvfn)
    plt.figure()
    plt.plot(Energy*har2wave)
    # plt.ylim(1799.99, 1800.01)
    print(np.mean(Energy[400:])*har2wave)
    plt.savefig('Imp_samp_%s_energy.png' %pot)

    amp, xx = np.histogram(psi.coords, weights=psi.weights, bins=25, range=(-0.8, 0.8), density=True)
    bins = (xx[1:] + xx[:-1])/2.
    # amp_2, xx_2 = np.histogram(psi.coords, weights=psi.d, bins=50, range=(-0.8, 0.8), density=True)
    # bins_2 = (xx_2[1:] + xx_2[:-1])/2.

# plt.figure()
# plt.plot(HOwvfn[0, :], HOwvfn[1, :])
# plt.plot(HOwvfn[0, :], HOwvfn[2, :])
# plt.plot(HOwvfn[0, :], HOwvfn[3, :])
# plt.savefig('HO_wvfn.png')

    plt.figure()
    plt.plot(bins, amp)
    # plt.plot(bins_2, amp_2)
    plt.savefig('Imp_samp_%s_psi.png' %pot)


do_the_things('morse')

