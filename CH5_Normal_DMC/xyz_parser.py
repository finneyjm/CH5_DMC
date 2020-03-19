import numpy as np


def write_to(coords, atoms, potential, file_name):
    with open(f'{file_name}', 'w') as my_file:
        for walkers in range(len(coords)):
            my_file.write(f'{len(atoms)}\n')
            my_file.write(f'{potential[walkers]}\n')
            for atom in range(len(atoms)):
                my_file.write(f'{atoms[atoms]} {coords[walkers, atom, 0]} {coords[walkers, atom, 1]} {coords[walkers, atom, 2]}\n')
        my_file.close()


def extractXYZ(fname, atmStr):
    import pandas as pd
    """
    Extracts the coordinates from an xyz file and returns it as an np array of dimension nxmx3,
    where n = number of geometries, m = number of atoms, and 3 = cartesian coordinates
    :param fname:
    :type str
    :param atmStr: a list of strings that will be used to parse the file.
    :type str        
    :return: np.ndarray
    """
    k = pd.read_table(fname, delim_whitespace=True, names=['atom', 'x', 'y', 'z'])
    k.dropna(inplace=True)
    someVals = list(set(atmStr))
    j = k.loc[k['atom'].isin(someVals)]
    xx = j.loc[:, ['x', 'y', 'z']].to_numpy()
    nLines = len(xx)
    nAts = len(atmStr)
    return xx.reshape(int(nLines / nAts), nAts, 3)


blah = np.load('geom_test_pwater/inputs_allH.npy')[:, [11-1, 12-1, 13-1, 4-1, 10-1, 9-1, 3-1, 5-1, 6-1, 1-1, 7-1, 8-1, 2-1], :]
v1 = np.loadtxt('geom_test_pwater/eng_dip_final.dat')
from ProtWaterPES import Potential
pot = Potential(13)
v2 = pot.get_potential(blah)
a = v2-v1[:, 0]
print(v2-v1[:, 0])
