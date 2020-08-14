import numpy as np
import matplotlib as plt

har2wave = 219474.6
#
# non_imp_samp_walkers = 100000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp/' +
#                      f'ptrimer_non_imp_samp_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -9.129961343400107E-002 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print('trimer full imp 100000 mean = ' + str(np.mean(energies_imp)))
# print('trimer full imp 100000 std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 30000
# energies_imp = np.zeros(3)
#
# for j in np.arange(0, 3):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_full_imp_samp/' +
#                      f'ptrimer_full_imp_samp_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -9.129961343400107E-002 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print('trimer full imp 30000 mean = ' + str(np.mean(energies_imp)))
# print('trimer full imp 3000 std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 20000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_full_imp_samp_w_tetramer_params/' +
#                      f'ptrimer_full_imp_samp_w_tetramer_params_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -9.129961343400107E-002 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print('trimer imp w tetramer 20000 mean = ' + str(np.mean(energies_imp)))
# print('trimer imp w tetramer 20000 std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 30000
# energies_imp = np.zeros(3)
#
# for j in np.arange(0, 3):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_imp_samp_waters/' +
#                      f'ptrimer_imp_samp_waters_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -9.129961343400107E-002 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print('trimer imp 30000 mean = ' + str(np.mean(energies_imp)))
# print('trimer imp 30000 std = ' + str(np.std(energies_imp)))
#
# imp_samp_walkers = 20000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/pdimer_non_imp_samp/' +
#                      f'pdimer_non_imp_samp_{imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) + 11792.22559467
#
# print('full imp 20000 = ' + str(energies_imp))
# print('full imp 20000 mean = ' + str(np.mean(energies_imp)))
# print('full imp 20000 std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 2000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
#                      f'ptetramer_full_imp_samp_w_trimer_params_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
# # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
# #                      f'ptetramer_full_imp_samp_w_trimer_params_{non_imp_samp_walkers}_' +
# #                      f'Walkers_Test_{1}.npz')['Eref'] * har2wave
# # a = -0.122146858971399 * har2wave
# # energies_imp[0] = np.mean(Energy[5000:]) - a
#
# print('tet w trim imp 2000 mean = ' + str(np.mean(energies_imp)))
# print('tet w trim imp 2000 std = ' + str(np.std(energies_imp)))
# print(energies_imp)
#
#
# non_imp_samp_walkers = 5000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
#                      f'ptetramer_full_imp_samp_w_trimer_params_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
# # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
# #                      f'ptetramer_full_imp_samp_w_trimer_params_{non_imp_samp_walkers}_' +
# #                      f'Walkers_Test_{1}.npz')['Eref'] * har2wave
# # a = -0.122146858971399 * har2wave
# # energies_imp[0] = np.mean(Energy[5000:]) - a
#
# print('tet w trim imp 5000 mean = ' + str(np.mean(energies_imp)))
# print('tet w trim imp 5000 std = ' + str(np.std(energies_imp)))
# print(energies_imp)
#
# non_imp_samp_walkers = 10000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
#                      f'ptetramer_full_imp_samp_w_trimer_params_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j-2] = np.mean(Energy[5000:]) - a
#
# print(f'tet w trim imp {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'tet w trim imp {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# print(energies_imp)
#
# non_imp_samp_walkers = 15000
# energies_imp = np.zeros(4)
#
# for j in np.arange(1, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
#                      f'ptetramer_full_imp_samp_w_trimer_params_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j-1] = np.mean(Energy[5000:]) - a
#
# print(f'tet w trim imp {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'tet w trim imp {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# print(energies_imp)
#
# non_imp_samp_walkers = 20000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
#                      f'ptetramer_full_imp_samp_w_trimer_params_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'tet w trim imp {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'tet w trim imp {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# print(energies_imp)
#
# non_imp_samp_walkers = 40000
# energies_imp = np.zeros(4)
#
# for j in np.arange(0, 3):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
#                      f'ptetramer_full_imp_samp_patched_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
# Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
#                      f'ptetramer_full_imp_samp_patched_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{5}.npz')['Eref'] * har2wave
# a = -0.122146858971399 * har2wave
# energies_imp[-1] = np.mean(Energy[5000:]) - a
#
# print(energies_imp)
# print(f'tet patched imp {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'tet patched imp {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
#
#
# non_imp_samp_walkers = 5000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
#                      f'ptetramer_full_imp_samp_patched_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'tet patched imp {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'tet patched imp {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 10000
# energies_imp = np.zeros(3)
#
# for j in np.arange(2, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
#                      f'ptetramer_full_imp_samp_patched_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j-2] = np.mean(Energy[5000:]) - a
#
# print(f'tet patched imp {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'tet patched imp {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 15000
# energies_imp = np.zeros(3)
#
# for j in np.arange(2, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
#                      f'ptetramer_full_imp_samp_patched_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j-2] = np.mean(Energy[5000:]) - a
#
# print(f'tet patched imp {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'tet patched imp {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 20000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
#                      f'ptetramer_full_imp_samp_patched_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'tet patched imp {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'tet patched imp {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
#
# imp_samp_walkers = 50000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
#                      f'ptetramer_non_imp_samp_ts_10_{imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j-2] = np.mean(Energy[500:2000]) - a
#
# # Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp/' +
# #                      f'ptetramer_non_imp_samp_{imp_samp_walkers}_' +
# #                      f'Walkers_Test_{5}.npz')['Eref'] * har2wave
# # a = -0.122146858971399 * har2wave
# # energies_imp[-1] = np.mean(Energy[5000:]) - a
#
# print('full non imp ts 10 50000 mean = ' + str(np.mean(energies_imp)))
# print('full non imp ts 10 50000 std = ' + str(np.std(energies_imp)))
#
# imp_samp_walkers = 75000
# energies_imp = np.zeros(5)
#
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
#                      f'ptetramer_non_imp_samp_ts_10_{imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j - 2] = np.mean(Energy[5000:20000]) - a
#
# print('full non imp ts 10 75000 mean = ' + str(np.mean(energies_imp)))
# print('full non imp ts 10 75000 std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 50000
# energies_imp = np.zeros(4)
# ts = 8
# for j in np.arange(0, 4):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_{ts}/' +
#                      f'ptetramer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:20000]) - a
#
# print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 20000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/pdimer_non_imp_samp_ts_{ts}/' +
#                      f'pdimer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -11804.25054877
#     energies_imp[j] = np.mean(Energy[5500:7000]) - a
#
# print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 75000
# energies_imp = np.zeros(5)
# ts = 1
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp/' +
#                      f'ptetramer_non_imp_samp_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:20000]) - a
#
# print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
#
#
# # non_imp_samp_walkers = 60000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_{ts}/' +
#                      f'ptetramer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     # a = -0.122146858971399 * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[500:2000]) - a
#
# print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
# # non_imp_samp_walkers = 10000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_{ts}/' +
#                      f'ptetramer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     # a = -0.122146858971399 * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5500:7000]) - a
#
# print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
#
# # non_imp_samp_walkers = 10000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_{ts}/' +
#                      f'ptetramer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     # a = -0.122146858971399 * har2wave
#     a = -0.122146858971399 * har2wave
#     energies_imp[j] = np.mean(Energy[5000:20000]) - a
#
# print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
# #
# # non_imp_samp_walkers = 20000
# # energies_imp = np.zeros(5)
# # ts = 1
# # for j in np.arange(0, 5):
# #     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp/' +
# #                      f'ptrimer_non_imp_samp_{non_imp_samp_walkers}_' +
# #                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
# #     a = -9.129961343400107E-002 * har2wave
# #     energies_imp[j] = np.mean(Energy[5000:20000]) - a
# #
# # print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# # print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
#
# # non_imp_samp_walkers = 20000
# # energies_imp = np.zeros(5)
# # ts = 10
# # for j in np.arange(0, 5):
# #     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp_ts_{ts}/' +
# #                      f'ptrimer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
# #                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
# #     a = -9.129961343400107E-002 * har2wave
# #     energies_imp[j] = np.mean(Energy[5000:20000]) - a
# #
# # print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# # print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
# #
# non_imp_samp_walkers = 15000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp_ts_{ts}/' +
#                      f'ptrimer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -9.129961343400107E-002 * har2wave
#     energies_imp[j] = np.mean(Energy[5500:7000]) - a
#
# print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
#
# non_imp_samp_walkers = 25000
# energies_imp = np.zeros(4)
# ts = 10
# for j in np.arange(0, 2):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp_ts_{ts}/' +
#                      f'ptrimer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -9.129961343400107E-002 * har2wave
#     energies_imp[j] = np.mean(Energy[500:2000]) - a
# for j in np.arange(3, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp_ts_{ts}/' +
#                      f'ptrimer_non_imp_samp_ts_{ts}_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = -9.129961343400107E-002 * har2wave
#     energies_imp[j-1] = np.mean(Energy[500:2000]) - a
#
# print(f'non imp ts {ts} mean = ' + str(np.mean(energies_imp)))
# print(f'non imp ts {ts} std = ' + str(np.std(energies_imp)))
#
# average = np.zeros(5)
# std = np.zeros(5)
# non_imp_samp_walkers = 5000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/imp_sampled_analytic_5H_ts_1/' +
#                      f'imp_sampled_analytic_5H_ts_1_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = 0
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# average[4] = np.mean(energies_imp)
# std[4] = np.std(energies_imp)
#
# non_imp_samp_walkers = 4000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/imp_sampled_analytic_5H_ts_1/' +
#                      f'imp_sampled_analytic_5H_ts_1_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = 0
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# average[3] = np.mean(energies_imp)
# std[3] = np.std(energies_imp)
#
#
# non_imp_samp_walkers = 3000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/imp_sampled_analytic_5H_ts_1/' +
#                      f'imp_sampled_analytic_5H_ts_1_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = 0
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# average[2] = np.mean(energies_imp)
# std[2] = np.std(energies_imp)
#
#
# non_imp_samp_walkers = 2000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/imp_sampled_analytic_5H_ts_1/' +
#                      f'imp_sampled_analytic_5H_ts_1_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = 0
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# average[1] = np.mean(energies_imp)
# std[1] = np.std(energies_imp)
#
# non_imp_samp_walkers = 1000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/imp_sampled_analytic_5H_ts_1/' +
#                      f'imp_sampled_analytic_5H_ts_1_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = 0
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'imp samp analytic CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# average[0] = np.mean(energies_imp)
# std[0] = np.std(energies_imp)
#
# import matplotlib.pyplot as plt
# plt.errorbar((1000, 2000, 3000, 4000, 5000), average, yerr=std, color='orange', label='Guided (harmonic)')
#
# average = np.zeros(3)
# std = np.zeros(3)
# non_imp_samp_walkers = 5000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_5H_GSW2/' +
#                      f'HH_to_rCHrCD_5H_GSW2_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = 0
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'imp samp  CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'imp samp  CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# average[2] = np.mean(energies_imp)
# std[2] = np.std(energies_imp)
#
# # non_imp_samp_walkers = 4000
# # energies_imp = np.zeros(5)
# # ts = 10
# # for j in np.arange(0, 5):
# #     Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_5H_GSW2/' +
# #                      f'HH_to_rCHrCD_5H_GSW2_{non_imp_samp_walkers}_' +
# #                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
# #     a = 0
# #     energies_imp[j] = np.mean(Energy[5000:]) - a
# #
# # print(f'imp samp  CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# # print(f'imp samp  CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# # average[3] = np.mean(energies_imp)
# # std[3] = np.std(energies_imp)
# #
# #
# # non_imp_samp_walkers = 3000
# # energies_imp = np.zeros(5)
# # ts = 10
# # for j in np.arange(0, 5):
# #     Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_5H_GSW2/' +
# #                      f'HH_to_rCHrCD_5H_GSW2_{non_imp_samp_walkers}_' +
# #                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
# #     a = 0
# #     energies_imp[j] = np.mean(Energy[5000:]) - a
# #
# # print(f'imp samp  CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# # print(f'imp samp  CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# # average[2] = np.mean(energies_imp)
# # std[2] = np.std(energies_imp)
#
#
# non_imp_samp_walkers = 2000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_5H_GSW2/' +
#                      f'HH_to_rCHrCD_5H_GSW2_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = 0
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'imp samp  CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'imp samp  CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# average[1] = np.mean(energies_imp)
# std[1] = np.std(energies_imp)
#
# non_imp_samp_walkers = 1000
# energies_imp = np.zeros(5)
# ts = 10
# for j in np.arange(0, 5):
#     Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_5H_GSW2/' +
#                      f'HH_to_rCHrCD_5H_GSW2_{non_imp_samp_walkers}_' +
#                      f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
#     a = 0
#     energies_imp[j] = np.mean(Energy[5000:]) - a
#
# print(f'imp samp  CH5 {non_imp_samp_walkers} mean = ' + str(np.mean(energies_imp)))
# print(f'imp samp  CH5 {non_imp_samp_walkers} std = ' + str(np.std(energies_imp)))
# average[0] = np.mean(energies_imp)
# std[0] = np.std(energies_imp)
#
# plt.errorbar((1000, 2000, 5000), average, yerr=std, color='blue', label='Guided (DVR wave function)')
# plt.plot(np.linspace(1000, 5000, 5000), np.array([10917]*5000), color='black')
# plt.show()

walk = [1000, 2000, 3000, 4000, 5000]
energies = np.zeros(2)
average = np.zeros(5)
std = np.zeros(5)
for i in range(len(walk)):
    for j in range(2):
        Energy = np.load(f'Trial_wvfn_testing/results/imp_sampled_discrete_analytic_5H_ts_1/'
                         f'imp_sampled_discrete_analytic_5H_ts_1_{walk[i]}_Walkers_Test_{j+1}.npz')['Eref']*har2wave
        energies[j] = np.mean(Energy[5000:])
    average[i] = np.mean(energies)
    std[i] = np.std(energies)
print(average, std)
