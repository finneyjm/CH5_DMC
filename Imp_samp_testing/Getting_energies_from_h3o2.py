import numpy as np

har2wave = 219474.6
ground_erefs = np.zeros((10, 20000))
for i in range(10):
    blah = np.load(f'ground_state_full_h3o2_{i+1}.npz')
    eref = blah['Eref']
    ground_erefs[i] = eref

print(np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_zpe = np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_zpe = np.std(np.mean(ground_erefs[:, 5000:]*har2wave, axis=1))
print(std_zpe)

excite_neg_erefs = np.zeros((5, 20000))
for i in range(5):
    blah = np.load(f'Asym_excite_state_full_h3o2_left_{i+1}.npz')
    eref = blah['Eref']
    excite_neg_erefs[i] = eref

gtg = [0, 1, 4]

print(np.mean(np.mean(excite_neg_erefs[gtg, 5000:], axis=1), axis=0)*har2wave)
average_excite_neg_energy = np.mean(np.mean(excite_neg_erefs[gtg, 5000:], axis=1), axis=0)*har2wave
std_excite_neg_energy = np.std(np.mean(excite_neg_erefs[gtg, 5000:]*har2wave, axis=1))

excite_pos_erefs = np.zeros((5, 20000))
for i in range(5):
    blah = np.load(f'Asym_excite_state_full_h3o2_right_{i+1}.npz')
    eref = blah['Eref']
    excite_pos_erefs[i] = eref

gtg = [2, 3, 4]

print(np.mean(np.mean(excite_pos_erefs[gtg, 5000:], axis=1), axis=0)*har2wave)
average_excite_pos_energy = np.mean(np.mean(excite_pos_erefs[gtg, 5000:], axis=1), axis=0)*har2wave
std_excite_pos_energy = np.std(np.mean(excite_pos_erefs[gtg, 5000:]*har2wave, axis=1))

print(average_excite_neg_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_neg_energy**2))
print(average_excite_pos_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_pos_energy**2))

average_excite_energy = np.average(np.array([average_excite_pos_energy, average_excite_neg_energy]))
std_excite_energy = np.sqrt(std_excite_pos_energy**2 + std_excite_neg_energy**2)
print(average_excite_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_energy**2))

excite_neg_erefs = np.zeros((5, 20000))
for i in range(5):
    blah = np.load(f'XH_excite_state_full_h3o2_left_{i+1}.npz')
    eref = blah['Eref']
    excite_neg_erefs[i] = eref

print(np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_excite_neg_energy = np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_excite_neg_energy = np.std(np.mean(excite_neg_erefs[:, 5000:]*har2wave, axis=1))

excite_pos_erefs = np.zeros((5, 20000))
for i in range(5):
    blah = np.load(f'XH_excite_state_full_h3o2_right_{i+1}.npz')
    eref = blah['Eref']
    excite_pos_erefs[i] = eref

print(np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave)
average_excite_pos_energy = np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave
std_excite_pos_energy = np.std(np.mean(excite_pos_erefs[:, 5000:]*har2wave, axis=1))

print(average_excite_neg_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_neg_energy**2))
print(average_excite_pos_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_pos_energy**2))

average_excite_energy = np.average(np.array([average_excite_pos_energy, average_excite_neg_energy]))
std_excite_energy = np.sqrt(std_excite_pos_energy**2 + std_excite_neg_energy**2)
print(average_excite_energy-average_zpe)
print(np.sqrt(std_zpe**2 + std_excite_energy**2))

