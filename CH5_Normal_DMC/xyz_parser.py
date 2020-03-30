import numpy as np


def write_to(coords, atoms, potential, file_name):
    with open(f'{file_name}', 'w') as my_file:
        for walkers in range(len(coords)):
            my_file.write(f'{len(atoms)}\n')
            my_file.write(f'{potential[walkers]}\n')
            for atom in range(len(atoms)):
                my_file.write(f'{atoms[atom]} {coords[walkers, atom, 0]} {coords[walkers, atom, 1]} {coords[walkers, atom, 2]}\n')
            my_file.write('\n')
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


# blah = np.load('geom_test_pwater/inputs_allH.npy')[:, [11-1, 12-1, 13-1, 4-1, 10-1, 9-1, 3-1, 5-1, 6-1, 1-1, 7-1, 8-1, 2-1], :]
# v1 = np.loadtxt('geom_test_pwater/eng_dip_final.dat')
# from ProtWaterPES import Potential
# pot = Potential(13)
# v2 = pot.get_potential(blah)
# a = v2-v1[:, 0]
# print(v2-v1[:, 0])
blah = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
                         f'ptetramer_non_imp_samp_ts_10_{10000}_' +
                         f'Walkers_Test_{3}.npz')

des = blah['des']
har2wave = 219474.6
ind = np.argwhere(des > 200)
a = -0.122146858971399 * har2wave
ang2bohr = 1.e-10/5.291772106712e-11
print(len(ind))
coords = blah['coords']
from Lookin_at_projects import rotateBackToFrame
coords = np.reshape(coords, (coords.shape[0]*coords.shape[1], coords.shape[2], coords.shape[3]))
coords = rotateBackToFrame(coords, 7, 10, 13)
ind1 = np.argwhere(coords[:, 4-1, 2] < -2.)
new_des = np.reshape(des, (des.shape[0]*des.shape[1]))[ind1.reshape(ind1.shape[0])]
ind = np.argwhere(new_des > 80)
print(len(ind))
coords = coords[ind1.reshape(ind1.shape[0])]
coords = coords[ind.reshape(ind.shape[0])]
atoms = ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O']
from ProtWaterPES import Potential
pot = Potential(len(atoms))
v = pot.get_potential(coords) * har2wave - a
write_to(coords/ang2bohr, atoms, v, 'geom_test_pwater/testing_geoms_10000.xyz')
# geom = [
# [2.3408386956424763, 3.818565314738952, -1.0602949892624935],
# [-0.46874289556311005, 4.66225946462522, -0.0992408978146262],
# [4.763569725420793, 3.0241003366673516, 0.11037597446109443],
# [4.1360534539560305, 2.9685140341804743, -2.055869594612759],
# [-2.4118479105483552, -0.26861724071490145, -0.15835456183418112],
# [-0.35005510223596603, 1.0006503773507784, 1.75461399037408],
# [0.0, 0.0, 0.0],
# [9.247171290617729, 0.9473464155126506, -1.1340615740889264],
# [7.93314182177458, -2.236758623488915, -0.7409834240744195],
# [7.734023809079242, 3.7601960732845087e-16, 1.1908948940065716e-16],
# [3.6100351426992283, 8.191043562660685, 1.968876215188661],
# [2.9811530459712463, 9.500748358388789, 0.288406882951228],
# [3.5248156808021514, 7.229128475239679, -2.1094509951894596e-16]
# ]
#
#
# geom = np.array([geom]*3)
# from ProtWaterPES import Potential
# # pot = Potential(13)
# # v1 = pot.get_potential(geom)*har2wave - a
# # print(v1)
# # a = np.array(geom[:, 2])
#
# # geom[:, 2] = geom[:, 0]
# # geom[:, 0] = a
# pot = Potential(13)
# v2 = pot.get_potential(geom)*har2wave - a
# print(v2)