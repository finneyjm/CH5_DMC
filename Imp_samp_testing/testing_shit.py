import numpy as np

x = np.array([])
y = np.array([])
with open('roh_morse.dat', 'r') as f:
    for line in f:
        words = line.split()
        x = np.append(x, float(words[0]))
        y = np.append(y, float(words[3]))

gsw = np.vstack((x, y))
np.save('Water_oh_stretch_GSW', np.vstack((x, y)))



