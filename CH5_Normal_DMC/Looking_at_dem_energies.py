import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from Coordinerds.CoordinateSystems import *

order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
ni = ['_CD', '_1H', '_2H', '_3H', '_4H', '_5H']
ens = [8044, 8563, 9097, 9698.52384684, 10304.93858414, 10917]
ens = [8044, 8563, 9097, 9698.52384684, 10304.93858414, 23370.6]
lowlim = [8020, 8540, 9070, 9670, 10280, 10901]
highlim = [8070, 8600, 9120, 9720, 10330, 10949]


def lets_get_some_energies(non_imp_samp_walkers, imp_samp_walkers, trials_ni, trials_i, isotop):
    N_i = len(imp_samp_walkers)
    N_n = len(non_imp_samp_walkers)
    energies_imp = np.zeros((N_i, trials_i))
    energies_non = np.zeros((N_n, trials_ni))
    for j in range(trials_ni):
        for i in range(N_n):
            # Energy = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled{ni[isotop]}_ts_10/' +
            #                  f'Non_imp_sampled{ni[isotop]}_ts_10_{non_imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
            # energies_non[i, j] += np.mean(Energy[5000:])
            #
            # Energy = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled_pdimer/' +
            #                   f'Non_imp_sampled_pdimer_{non_imp_samp_walkers[i]}_' +
            #                   f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
            # energies_non[i, j] += np.mean(Energy[5000:]) + 11792.2255946
            # energies_non[i,j] += np.mean(Energy[5000:]) + (9.129961343400107E-002*har2wave)

            # Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp/' +
            #                  f'ptrimer_non_imp_samp_{non_imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # a = -9.129961343400107E-002 * har2wave
            # energies_non[i, j] = np.mean(Energy[5000:]) - a

            Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
                            f'ptetramer_non_imp_samp_ts_10_{non_imp_samp_walkers[i]}_' +
                            f'Walkers_Test_{j+1}.npz')
            Energy = Energy['Eref']*har2wave
            a = -0.122146858971399 * har2wave
            energies_non[i, j] = np.mean(Energy[10000:]) - a

            # Energy = np.load(f'Trial_wvfn_testing/results/pmonomer_non_imp_samp/' +
            #                  f'pmonomer_non_imp_samp_{non_imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            # a = 3.404489490321794E-006 * har2wave
            # a = 0.00395373*har2wave
            # a = 0.00500124*har2wave
            # a = -9.129961343400107E-002 * har2wave
            # a = 0*har2wave
            # energies_non[i, j] = np.mean(Energy[5000:]) - a

    # for j in range(trials_i):
    #     for i in range(N_i):
            # Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_{isotop}H_GSW2/' +
            #                  f'HH_to_rCHrCD_{isotop}H_GSW2_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref']*har2wave

            # Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_{isotop}H_GSW2/' +
            #                  f'HH_to_rCHrCD_{isotop}H_GSW2_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
            # energies_imp[i, j] += np.mean(Energy[5000:])

            # Energy = np.load(f'Trial_wvfn_testing/results/pdimer_waters_described/' +
            #                   f'pdimer_waters_described_{non_imp_samp_walkers[i]}_' +
            #                   f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            # energies_imp[i, j] = np.mean(Energy[5000:]) + 11792.22559467
            # energies_imp[i, j] = np.mean(Energy[5000:]) + (9.129961343400107E-002*har2wave)

            # Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_full_imp_samp/' +
            #                  f'ptrimer_full_imp_samp_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # a = -9.129961343400107E-002 * har2wave
            # energies_imp[i, j] = np.mean(Energy[5000:]) - a

            # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp/' +
            #                 f'ptetramer_full_imp_samp_{imp_samp_walkers[i]}_' +
            #                 f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # a = -0.122146858971399 * har2wave
            # energies_imp[i, j] = np.mean(Energy[5000:]) - a
            # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_w_tetramer_params_imp_samp/' +
            #                  f'ptetramer_w_tetramer_params_imp_samp_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # energies_imp[i, j] = np.mean(Energy[5000:]) - a

            # Energy = np.load(f'Trial_wvfn_testing/results/pmonomer_imp_samp/' +
            #                  f'pmonomer_imp_samp_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # a = 0 * har2wave
            # energies_imp[i, j] = np.mean(Energy[5000:]) - a

    avg_imp = np.mean(energies_imp, axis=1)
    print(avg_imp)
    std_imp = np.std(energies_imp, axis=1)
    # std_imp = np.zeros(std_imp.shape)
    print(std_imp)
    avg_non = np.mean(energies_non, axis=1)
    print(avg_non)
    std_non = np.std(energies_non, axis=1)
    # std_non = np.zeros(std_non.shape)
    print(std_non)
    fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
    axes[0].errorbar(non_imp_samp_walkers, avg_non, yerr=std_non, marker='s', markerfacecolor='none', color='red', label='No Impt. Samp.')
    axes[0].scatter(non_imp_samp_walkers, [ens[isotop]] * len(non_imp_samp_walkers), color='purple', label="Anne's Value", linestyle='--')
    # axes[1].errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='blue', label='Impt. Samp.', marker='o')
    # axes[1].plot(imp_samp_walkers, [ens[isotop]] * len(imp_samp_walkers), color='purple', linestyle='--', label="Anne's Value")
    # axes[0].set_xlabel('Number of Walkers')
    axes[1].set_xlabel('Number of Walkers', fontsize=16)
    axes[0].set_ylabel(r'Energy (cm$^{-1}$)', fontsize=16)
    axes[1].set_ylabel(r'Energy (cm$^{-1}$)', fontsize=16)
    # axes[0].set_ylim(lowlim[isotop], highlim[isotop])
    # axes[1].set_ylim(lowlim[isotop], highlim[isotop])
    axes[1].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                     bottom=True, top=True, left=True, right=False, labelsize=12)
    axes[0].tick_params(labelsize=12)
    # axes[0].set_ylim(10850, 10950)
    # axes[1].set_ylim(10850, 10950)
    # axes[0].set_ylim(8020, 8070)
    # axes[1].set_ylim(8020, 8070)
    # axes[0].set_ylim(10280, 10330)
    # axes[1].set_ylim(10280, 10330)
    # axes[1].set_ylim(9670, 9720)
    # axes[0].set_ylim(9670, 9720)
    # axes[1].set_ylim(9070, 9120)
    # axes[0].set_ylim(9070, 9120)
    # axes[1].set_ylim(8540, 8600)
    # axes[0].set_ylim(8540, 8600)
    # axes[1].set_ylim(18000, 18199)
    # axes[0].set_ylim(18001, 18200)
    axes[1].set_ylim(23300, 23799)
    axes[0].set_ylim(23301, 23800)
    # axes[1].set_ylim(20580, 20649)
    # axes[0].set_ylim(20581, 20650)
    # axes[0].set_ylim(7421, 7500)
    # axes[1].set_ylim(7420, 7499)
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    # fig.savefig(f'Convergence_plots/Energy_convergence_CH5_for_ppt.png')
    # plt.close(fig)


walkers1 = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
walkers10 = [100, 200, 500, 1000, 2000, 5000, 10000]
walkers11 = [100, 200, 500, 1000, 2000, 5000, 15000]
walkers67 = [100, 200, 500, 1000, 2000, 5000, 20000]
walkers89 = [10000, 20000, 40000]
walkers68 = [5000]
walkers99 = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
walkers3 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
walkers5 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 60000]
walkers6 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 40000, 50000, 60000]
walkers2 = [500, 1000, 2000, 5000, 10000]
walkers4 = [100, 200, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
            5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 20000]
lets_get_some_energies(walkers89, walkers89, 4, 5, 5)
plt.show()

# blah = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
#                              f'ptetramer_non_imp_samp_ts_10_{20000}_' +
#                              f'Walkers_Test_{2}.npz')
# # blah2 = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp/' +
# #                              f'ptetramer_full_imp_samp_{5000}_' +
# #                              f'Walkers_Test_{1}.npz')
# # #
# # # a = blah['weights']
# a = -0.122146858971399 * har2wave
# # # b = blah['des']
# # eref = blah2['Eref']*har2wave
# eref_non = blah['Eref']*har2wave - a
# # # plt.plot(blah2['Eref'])
# # plt.plot(eref - a, label='imp samp')
# plt.plot(eref_non, label = 'non imp samp')
# plt.xlabel(r'$\tau$')
# plt.ylabel(r'Eref cm$^{-1}$')
# plt.legend()
# plt.show()
# plt.close()
