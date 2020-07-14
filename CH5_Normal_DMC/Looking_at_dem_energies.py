import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from Coordinerds.CoordinateSystems import *

order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
ni = ['_CD', '_1H', '_2H', '_3H', '_4H', '']
ens = [8044, 8563, 9097, 9694, 10300, 10917]
ens_std = [2, 3, 6, 3, 6, 3]
# ens = [8044, 8563, 9097, 9698.52384684, 10304.93858414, 23370.6]
lowlim = [8020, 8540, 9080, 9680, 10280, 10895]
highlim = [8070, 8600, 9130, 9730, 10330, 10945]
avg_beg = 5000
avg_end = 20000


def lets_get_some_energies(non_imp_samp_walkers, imp_samp_walkers, trials_ni, trials_i, isotop):
    N_i = len(imp_samp_walkers)
    N_n = len(non_imp_samp_walkers)
    energies_imp = np.zeros((N_i, trials_i))
    energies_imp2 = np.zeros((N_i+1, trials_i))
    energies_imp3 = np.zeros((N_i, trials_i))
    energies_non = np.zeros((N_n, trials_ni))

    for j in range(trials_ni):
        for i in range(N_n):
            # Energy = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled{ni[isotop]}/' +
            #                  f'Non_imp_sampled{ni[isotop]}_{non_imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
            # energies_non[i, j] += np.mean(Energy[5000:])
            #
            Energy = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled_pdimer/' +
                              f'Non_imp_sampled_pdimer_{non_imp_samp_walkers[i]}_' +
                              f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
            energies_non[i, j] += np.mean(Energy[5000:]) + 11804.25054877
            #
            # Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp/' +
            #                  f'ptrimer_non_imp_samp_{non_imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # a = -9.129961343400107E-002 * har2wave
            # energies_non[i, j] = np.mean(Energy[5000:]) - a
            #
            # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp/' +
            #                 f'ptetramer_non_imp_samp_{non_imp_samp_walkers[i]}_' +
            #                 f'Walkers_Test_{j+1}.npz')
            # Energy = Energy['Eref']*har2wave
            # a = -0.122146858971399 * har2wave
            # energies_non[i, j] = np.mean(Energy[avg_beg:avg_end]) - a

            # Energy = np.load(f'Trial_wvfn_testing/results/pmonomer_non_imp_samp/' +
            #                  f'pmonomer_non_imp_samp_{non_imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            #
            # a = 0*har2wave
            # energies_non[i, j] = np.mean(Energy[5000:]) - a

    for j in range(trials_i):
        for i in range(N_i):
            # Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_{isotop}H_GSW2/' +
            #                  f'HH_to_rCHrCD_{isotop}H_GSW2_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
            # energies_imp[i, j] += np.mean(Energy[5000:])
            #
            # Energy = np.load(f'Trial_wvfn_testing/results/Average_fd/' +
            #                  f'Average_fd_{imp_samp_walkers[i]}_Walkers' +
            #                  f'_Test_{j+1}.npz')['Eref']*har2wave
            # energies_imp[i, j] += np.mean(Energy[5000:])

            Energy = np.load(f'Trial_wvfn_testing/results/pdimer_full_imp_samp/' +
                              f'pdimer_full_imp_samp_{imp_samp_walkers[i]}_' +
                              f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            energies_imp[i, j] = np.mean(Energy[5000:]) + 11804.25054877
            #
            #
            # Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_full_imp_samp/' +
            #                  f'ptrimer_full_imp_samp_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # a = -9.129961343400107E-002 * har2wave
            # energies_imp[i, j] = np.mean(Energy[5000:]) - a
            #
            # Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_imp_samp_waters/' +
            #                  f'ptrimer_imp_samp_waters_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            # a = -9.129961343400107E-002 * har2wave
            # energies_imp2[i, j] = np.mean(Energy[5000:]) - a
            #
            # Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_full_imp_samp_w_tetramer_params_patched/' +
            #                  f'ptrimer_full_imp_samp_w_tetramer_params_patched_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            # a = -9.129961343400107E-002 * har2wave
            # energies_imp3[i, j] = np.mean(Energy[5000:]) - a
            # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
            #                 f'ptetramer_full_imp_samp_patched_{imp_samp_walkers[i]}_' +
            #                 f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # a = -0.122146858971399 * har2wave
            # energies_imp[i, j] = np.mean(Energy[avg_beg:avg_end]) - a
            # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_imp_samp_waters/' +
            #                  f'ptetramer_imp_samp_waters_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # energies_imp2[i, j] = np.mean(Energy[avg_beg:avg_end]) - a
            # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
            #                  f'ptetramer_full_imp_samp_w_trimer_params_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            # a = -0.122146858971399 * har2wave
            # energies_imp3[i, j] = np.mean(Energy[avg_beg:avg_end]) - a

            # Energy = np.load(f'Trial_wvfn_testing/results/pmonomer_full_imp_samp/' +
            #                  f'pmonomer_full_imp_samp_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
            # a = 0 * har2wave
            # energies_imp[i, j] = np.mean(Energy[5000:]) - a
            # Energy = np.load(f'Trial_wvfn_testing/results/pmonomer_full_imp_samp_water/' +
            #                  f'pmonomer_full_imp_samp_water_{imp_samp_walkers[i]}_' +
            #                  f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            # a = 0 * har2wave
            # energies_imp2[i, j] = np.mean(Energy[5000:]) - a



    # for j in range(trials_i):
    #     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_imp_samp_waters/' +
    #                      f'ptetramer_imp_samp_waters_{30000}_' +
    #                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
    #     energies_imp2[-1, j] = np.mean(Energy[avg_beg:avg_end]) - a
    #     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
    #                      f'ptetramer_full_imp_samp_patched_{30000}_' +
    #                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
    #     a = -0.122146858971399 * har2wave
    #     energies_imp[-2, j] = np.mean(Energy[avg_beg:avg_end]) - a
    #     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
    #                      f'ptetramer_full_imp_samp_patched_{40000}_' +
    #                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
    #     a = -0.122146858971399 * har2wave
    #     energies_imp[-1, j] = np.mean(Energy[avg_beg:avg_end]) - a

    # energies_non2 = np.zeros((6, 5))
    # walkers = [10000, 20000, 40000, 50000, 60000, 75000]
    # for i in range(len(walkers)):
    #     for j in range(trials_ni):
    #         Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
    #                          f'ptetramer_non_imp_samp_ts_10_{walkers[i]}_' +
    #                          f'Walkers_Test_{j + 1}.npz')
    #         Energy = Energy['Eref'] * har2wave
    #         a = -0.122146858971399 * har2wave
    #         energies_non2[i, j] = np.mean(Energy[avg_beg:avg_end]) - a
    # print(energies_non2)
    # avg_non_2 = np.mean(energies_non2, axis=1)
    # std_non_2 = np.std(energies_non2, axis=1)

    avg_imp = np.mean(energies_imp, axis=1)
    #tetramer
    # avg_imp = np.append(avg_imp, 23400.131892377227)
    # imp_samp_walkers = np.append(imp_samp_walkers, 30000)

    #trimer
    # avg_imp = np.append(avg_imp, (17998.764572511674))
    # imp_samp_walkers = np.append(imp_samp_walkers, (30000))


    # avg_imp = np.append(avg_imp, 23415.825656307297)
    print('average energy full importance sampling ' + str(avg_imp))
    std_imp = np.std(energies_imp, axis=1)

    #tetramer
    # std_imp = np.append(std_imp, 7.867248786868424)

    #trimer
    # std_imp = np.append(std_imp, (3.350799897507758))


    # std_imp = np.append(std_imp, 5.013309173053622)
    # std_imp = np.zeros(std_imp.shape)
    print('std full importance sampling ' + str(std_imp))

    avg_imp2 = np.mean(energies_imp2, axis=1)
    std_imp2 = np.std(energies_imp2, axis=1)

    #tetramer
    # avg_imp2 = np.append(avg_imp2, 23408.44127258477)
    # std_imp2 = np.append(std_imp2, 4.948843844437155)

    #trimer
    # avg_imp2 = np.append(avg_imp2, (18002.15435066399))
    # std_imp2 = np.append(std_imp2, (6.0249577448096945))


    # avg_imp2 = np.append(avg_imp2, 23414.60385447639)
    # std_imp2 = np.append(std_imp2, 2.9568529258685885)
    print('average energy imp samp waters ' + str(avg_imp2))
    print('std imp samp waters ' + str(std_imp2))

    avg_non = np.mean(energies_non, axis=1)
    #dimer
    avg_non = np.append(avg_non, 12359.101235609602-11792.22559467+11804.25054877)
    non_imp_samp_walkers = np.append(non_imp_samp_walkers, 20000)

    #trimer
    # avg_non = np.append(avg_non, 18006.18049926232)

    #tetramer
    # avg_non = np.append(avg_non, 23420.789982136594)
    # non_imp_samp_walkers = np.append(non_imp_samp_walkers, 100000)
    print('average energy non importance sampling ' + str(avg_non))

    std_non = np.std(energies_non, axis=1)
    #dimer
    std_non = np.append(std_non, 4.1243809112563055)

    #trimer
    # std_non = np.append(std_non, 6.935699043543936)
    # non_imp_samp_walkers = np.append(non_imp_samp_walkers, 100000)

    #tetramer
    # std_non = np.append(std_non, 9.913278209112672)
    #
    # ind = np.argsort(non_imp_samp_walkers)
    # non_imp_samp_walkers = non_imp_samp_walkers[ind]
    # std_non = std_non[ind]
    # avg_non = avg_non[ind]

    # std_non = np.append(std_non, (25.397939130287906, 11.337989742324408))
    # std_non = np.zeros(std_non.shape)
    print('std non importance sampling ' + str(std_non))

    # walk = [2000, 5000, 10000, 15000, 20000]
    # avg = [23473.828756335515, 23429.425616104454, 23417.28002768523, 23409.6225454949, 23398.88512361662]
    # std = [31.906986434699753, 33.82868049291804, 30.389336012701083, 11.063387347818585, 7.657193029344564]

    #trimer w tetramer params
    walk = [2000, 5000, 10000, 15000, 20000]
    avg = np.mean(energies_imp3, axis=1)
    std = np.std(energies_imp3, axis=1)

    # print(energies_imp)
    # np.savetxt(f'energies_imp_monomer_{avg_beg}_to_{avg_end}.csv', energies_imp, delimiter=',')
    # np.savetxt(f'energies_imp2_monomer_{avg_beg}_to_{avg_end}.csv', energies_imp2, delimiter=',')
    # np.savetxt(f'energies_imp3_{avg_beg}_to_{avg_end}.csv', energies_imp3, delimiter=',')
    # np.savetxt(f'energies_non_monomer_{avg_beg}_to_{avg_end}.csv', energies_non, delimiter=',')
    # print(energies_imp2)
    # print(energies_non)
    # print(energies_imp3)
    fig, axes = plt.subplots()
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [18002.9] * 5000, yerr= 14.4, color='purple', label="Anne's Std")
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [18002.9] * 5000, yerr= 0, color='black', label="Anne's Value")
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [7453.9] * 5000, yerr= 5.84, color='grey')
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [7453.9] * 5000, yerr= 0, color='black', label="Literature")
    axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [12393] * 5000, yerr=5, color='grey')
    axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [12393] * 5000, yerr=0, color='black',
                  label="Literature")
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [18018] * 5000, yerr=3, color='grey')
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [18018] * 5000, yerr=0, color='black',
    #               label="Literature")
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [23405] * 5000, yerr=21, color='grey')
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [23405] * 5000, yerr=0, color='black',
    #               label="Literature")
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [ens[isotop]] * 5000, yerr=ens_std[isotop],
    #               color='grey', alpha=0.1)
    # axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [ens[isotop]] * 5000, yerr=0, color='black', label="Literature")

    axes.errorbar(non_imp_samp_walkers, avg_non, yerr=std_non, marker='s', markerfacecolor='none', color='red', label='Unguided')
    # axes.errorbar(walkers,  avg_non_2,
    #               yerr=std_non_2, marker='s', markerfacecolor='none', color='green',
    #               label='Unguided ts = 10 a.u.')
    # axes.errorbar(walkers, avg_non_2, yerr=std_non_2, marker='s', markerfacecolor='none', color='black',
    #               label='No Impt. Samp. ts = 10')
    # axes.errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='blue', label=r'Guided (H$_7$O$_3^+$)', marker='o')
    axes.errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='orange', label='Guided (water)', marker='v')
    # axes.errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='blue', label='Guided', marker='o')

    # axes.errorbar(np.append(imp_samp_walkers, (30000, 40000)), avg_imp, yerr=std_imp, color='purple', label=r'Guided (H$_9$O$_4^+$)', marker='s')
    # axes.errorbar(np.append(imp_samp_walkers, 30000), avg_imp2, yerr=std_imp2, color='orange', label='Guided (water)', marker='v')
    #
    # axes.errorbar(np.append(imp_samp_walkers, 30000), np.append(avg_imp2, 23408.84),
    #               yerr=np.append(std_imp2, 6.134369), color='orange', label='Impt. Samp. waters', marker='v')

    #trimer w tetramer params
    # axes.errorbar(walk, avg, yerr=std, color='purple', label=r'Guided (H$_9$O$_4^+$)', marker='s')

    #tetramer w trimer params
    # axes.errorbar(imp_samp_walkers, avg, yerr=std, color='blue', label=r'Guided (H$_7$O$_3^+$)', marker='o')
    # axes.scatter(imp_samp_walkers, [ens[isotop]] * len(imp_samp_walkers), color='purple', linestyle='--', label="Anne's Value")

    # axes[0].set_xlabel('Number of Walkers')
    axes.set_xlabel(r'N$_{\rmw}$', fontsize=22)
    axes.set_ylabel(r'E$_0$ (cm$^{-1}$)', fontsize=22)
    # axes[1].set_ylabel(r'Energy (cm$^{-1}$)', fontsize=16)
    axes.set_ylim(lowlim[isotop], highlim[isotop])
    # axes.set_ylim(lowlim[isotop], highlim[isotop])
    axes.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                     bottom=True, top=False, left=True, right=False, labelsize=14)
    # axes[0].tick_params(labelsize=12)
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
    # axes.set_ylim(23330, 23530)
    # axes.set_ylim(23301, 23800)
    # axes[1].set_ylim(20580, 20649)
    # axes[0].set_ylim(20581, 20650)
    # axes[0].set_ylim(7421, 7500)
    # axes[1].set_ylim(7420, 7499)
    # axes.set_ylim(7430, 7530)
    # axes.set_ylim(17970, 18070)
    axes.set_ylim(12350, 12450)
    # leg = plt.legend(loc='upper right', fontsize=14)
    # leg.get_frame().set_edgecolor('white')
    # axes[1].legend()
    plt.tight_layout()
    # fig.savefig(f'Convergence_plots/Energy_convergence_CH5_for_ppt.png')
    # plt.close(fig)


walkers1 = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
walkers10 = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
walkers12 = [100, 200, 500, 1000, 2000, 5000, 10000]
walkers11 = [100, 200, 500, 2000, 5000]
# walkers11 = [100, 200, 500, 1000, 2000, 5000, 15000]
# walkers67 = [100, 200, 500, 1000, 2000, 5000, 20000]
walkers89 = [2000, 5000, 10000, 15000, 20000]
walkers68 = [2000, 5000, 10000, 15000, 20000, 30000]
# walkers67 = [2000, 5000, 10000, 20000]
walkers73 = [10000, 20000, 40000]
walkers74 = [10000, 20000, 30000, 40000, 50000, 75000, 100000]
walkers75 = [10000, 20000, 30000, 40000, 50000, 100000]
# walkers_dis = [40000]
walkers99 = [50, 100, 200, 500, 2000, 5000]
# walkers3 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
# walkers5 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 60000]
# walkers6 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 40000, 50000, 60000]
# walkers2 = [500, 1000, 2000, 5000, 10000]
# walkers4 = [100, 200, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
#             5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 20000]
# for i in range(6):
lets_get_some_energies(walkers12, walkers11, 5, 5, 0)
plt.show()
plt.close()

# blah = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp/' +
#                              f'ptetramer_full_imp_samp_{5000}_' +
#                              f'Walkers_Test_{5}.npz')
# blah2 = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp/' +
# #                              f'ptetramer_full_imp_samp_{5000}_' +
# #                              f'Walkers_Test_{1}.npz')
# # #
# # # a = blah['weights']
# a = -0.122146858971399 * har2wave
# b = blah['des']
# eref = blah2['Eref']*har2wave
# eref_non = blah['Eref']*har2wave - a
# plt.plot(blah2['Eref'])
# plt.plot(eref - a, label='imp samp')
# plt.plot(eref_non, label = 'non imp samp')
# plt.xlabel(r'$\tau$')
# plt.ylabel(r'Eref cm$^{-1}$')
# plt.legend()
# plt.show()
# plt.close()
