import numpy as np

har2wave = 219474.6


energy1 = np.load('Imp_min_energies_min_low.npy')*har2wave
energy2 = np.load('Imp_cs_energies_min_low.npy')*har2wave
energy3 = np.load('Imp_c2v_energies_min_low.npy')*har2wave

energy1cs = np.load('Imp_min_energies_cs_low.npy')*har2wave
energy2cs = np.load('Imp_cs_energies_cs_low.npy')*har2wave
energy3cs = np.load('Imp_c2v_energies_cs_low.npy')*har2wave

energy1c2v = np.load('Imp_min_energies_c2v_low.npy')*har2wave
energy2c2v = np.load('Imp_cs_energies_c2v_low.npy')*har2wave
energy3c2v = np.load('Imp_c2v_energies_c2v_low.npy')*har2wave

DVR_correct = [1488.113887, 1218.705882, 1224.753528, 1577.102303, 1577.102303]
DVR_correct_cs = [1626.480859, 1242.570938, 1242.570937, 1554.300194, 1554.300197]
DVR_correct_c2v = [1670.720578, 1919.949044, 1919.949044, 1713.462511, 1713.462511]

num_pts = 8
mean = np.zeros((3, 5, num_pts))
std = np.zeros((3, 5, num_pts))
diff = np.zeros((3, 5, num_pts))
mean_cs = np.zeros((3, 5, num_pts))
std_cs = np.zeros((3, 5, num_pts))
diff_cs = np.zeros((3, 5, num_pts))
mean_c2v = np.zeros((3, 5, num_pts))
std_c2v = np.zeros((3, 5, num_pts))
diff_c2v = np.zeros((3, 5, num_pts))
for j in range(5):
    for l in range(num_pts):
        mean[0, j, l] += np.mean(energy1[:, j, l])
        std[0, j, l] += np.std(energy1[:, j, l])
        diff[0, j, l] += mean[0, j, l] - DVR_correct[j]

        mean[1, j, l] += np.mean(energy2[:, j, l])
        std[1, j, l] += np.std(energy2[:, j, l])
        diff[1, j, l] += mean[1, j, l] - DVR_correct_cs[j]

        mean[2, j, l] += np.mean(energy3[:, j, l])
        std[2, j, l] += np.std(energy3[:, j, l])
        diff[2, j, l] += mean[2, j, l] - DVR_correct_c2v[j]

        mean_cs[0, j, l] += np.mean(energy1cs[:, j, l])
        std_cs[0, j, l] += np.std(energy1cs[:, j, l])
        diff_cs[0, j, l] += mean_cs[0, j, l] - DVR_correct[j]

        mean_cs[1, j, l] += np.mean(energy2cs[:, j, l])
        std_cs[1, j, l] += np.std(energy2cs[:, j, l])
        diff_cs[1, j, l] += mean_cs[1, j, l] - DVR_correct_cs[j]

        mean_cs[2, j, l] += np.mean(energy3cs[:, j, l])
        std_cs[2, j, l] += np.std(energy3cs[:, j, l])
        diff_cs[2, j, l] += mean_cs[2, j, l] - DVR_correct_c2v[j]

        mean_c2v[0, j, l] += np.mean(energy1c2v[:, j, l])
        std_c2v[0, j, l] += np.std(energy1c2v[:, j, l])
        diff_c2v[0, j, l] += mean_c2v[0, j, l] - DVR_correct[j]

        mean_c2v[1, j, l] += np.mean(energy2c2v[:, j, l])
        std_c2v[1, j, l] += np.std(energy2c2v[:, j, l])
        diff_c2v[1, j, l] += mean_c2v[1, j, l] - DVR_correct_cs[j]

        mean_c2v[2, j, l] += np.mean(energy3c2v[:, j, l])
        std_c2v[2, j, l] += np.std(energy3c2v[:, j, l])
        diff_c2v[2, j, l] += mean_c2v[2, j, l] - DVR_correct_c2v[j]

start = 0.6
with open("Switch_min_wvfns_data_low.txt", 'w') as my_file:
    my_file.write('Minimum Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' %(start + 0.05*float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean[0, i, j]) + ', ' + str(std[0, i, j]) + ', ' + str(diff[0, i, j]) + '\n')
    my_file.write('Cs Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' % (start + 0.05 * float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean[1, i, j]) + ', ' + str(std[1, i, j]) + ', ' + str(diff[1, i, j]) + '\n')
    my_file.write('Csv Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' % (start + 0.05 * float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean[2, i, j]) + ', ' + str(std[2, i, j]) + ', ' + str(diff[2, i, j]) + '\n')
    my_file.close()


with open("Switch_cs_wvfns_data_low.txt", 'w') as my_file:
    my_file.write('Minimum Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' %(start + 0.05*float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean_cs[0, i, j]) + ', ' + str(std_cs[0, i, j]) + ', ' + str(diff_cs[0, i, j]) + '\n')
    my_file.write('Cs Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' % (start + 0.05 * float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean_cs[1, i, j]) + ', ' + str(std_cs[1, i, j]) + ', ' + str(diff_cs[1, i, j]) + '\n')
    my_file.write('Csv Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' % (start + 0.05 * float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean_cs[2, i, j]) + ', ' + str(std_cs[2, i, j]) + ', ' + str(diff_cs[2, i, j]) + '\n')
    my_file.close()


with open("Switch_c2v_wvfns_data_low.txt", 'w') as my_file:
    my_file.write('Minimum Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' %(start + 0.05*float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean_c2v[0, i, j]) + ', ' + str(std_c2v[0, i, j]) + ', ' + str(diff_c2v[0, i, j]) + '\n')
    my_file.write('Cs Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' % (start + 0.05 * float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean_c2v[1, i, j]) + ', ' + str(std_c2v[1, i, j]) + ', ' + str(diff_c2v[1, i, j]) + '\n')
    my_file.write('Csv Potential \n')
    for j in range(num_pts):
        my_file.write('Switching point = %s' % (start + 0.05 * float(j)) + '\n')
        for i in range(5):
            my_file.write(str(mean_c2v[2, i, j]) + ', ' + str(std_c2v[2, i, j]) + ', ' + str(diff_c2v[2, i, j]) + '\n')
    my_file.close()














