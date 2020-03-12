ang2bohr = 1.e-10/5.291772106712e-11


# trimer = [[0.00000,        1.67720,        0.53729],
#           [-0.87302,        0.35992,        0.01707],
#           [0.87302,        0.35993,        0.01707],
#           [0.00000,        0.91527,       -0.05817],
#           [2.56267,       -0.75858,        0.76451],
#           [2.70113,       -0.40578,       -0.73813],
#           [2.07091,       -0.46191,       -0.00993],
#           [-2.70115,       -0.40575,       -0.73811],
#           [-2.56265,       -0.75862,        0.76451],
#           [-2.07092,       -0.46190,       -0.00993]]

trimer = [[1.822639226956,  0.000000000283,  0.000000000000],
          [0.300407646428,  -0.914198709660,  0.000000000000],
          [0.300407645812,  0.914198709862,  0.000000000000],
          [0.840815416177,  0.000000000283,  0.000000000000],
          [-0.714399465504, 2.673546910582, -0.785835530027],
          [-0.714326750385, 2.673421907097, 0.785943106958],
          [-0.420407708818, 2.168150028133, 0.000000000000],
          [-0.714326748587, -2.673421907578, -0.785943106958],
          [-0.714399463705, -2.673546911063, 0.785835530027],
          [-0.420407707359,  -2.168150028415,  -0.000000000000]]

# tetramer = [[-0.31106354,  -0.91215572,  -0.20184621],
#             [0.95094197,   0.18695800,  -0.20259538],
#             [-0.63272209,   0.72926470,  -0.20069859],
#             [0.00253217,   0.00164013,  -0.47227522],
#             [-1.96589559,   2.46292466,  -0.54312627],
#             [-2.13630186,   1.99106023,   0.90604777],
#             [-1.55099190,   1.94067311,   0.14704161],
#             [-1.18003749,  -2.92157144,  -0.54090532],
#             [-0.65410291,  -2.84939169,   0.89772271],
#             [-0.91203521,  -2.30896272,   0.14764850],
#             [2.79828182,   0.87002791,   0.89281564],
#             [3.12620054,   0.43432898,  -0.54032031],
#             [2.46079102,   0.36718848,   0.14815394]]

tetramer = [[1.030703806723,  -0.000000001370,  0.000000000000],
            [-0.515351904550,  -0.892615679713,  -0.000000000000],
            [-0.515351902175,  0.892615681087,  -0.000000000000],
            [-0.000000000001,  0.000000000001,  0.000000000000],
            [-1.556584242874,  -2.696082986637,  -0.781280977010],
            [-1.556438007664,  -2.695829699825,  0.781501264494],
            [-1.262240877623,  -2.186265324717,  -0.000000000000],
            [3.113168478575,  -0.000000004143,  -0.781280977010],
            [3.112876008157,  -0.000000004142,  0.781501264494],
            [2.524481749430,  -0.000000003359,  -0.000000000000],
            [-1.556584235700,  2.696082990777,  -0.781280977010],
            [-1.556438000492,  2.695829703965,  0.781501264494],
            [-1.262240871806,  2.186265328075,  -0.000000000000]]


dimer = [[0.000210, -0.040199, 0.000028],
         [-1.647488, 0.780819, -0.327399],
         [-1.688503, -0.439783, 0.676725],
         [-1.199788, -0.040085, -0.062851],
         [1.647240, 0.780469, 0.328402],
         [1.688077, -0.438127, -0.678047],
         [1.199846, -0.040312, 0.062887]]

monomer = [[0.93273984, -0.03166913, -0.23931206],
           [-0.43894472, 0.82361013, -0.23931163],
           [-0.49379696, -0.79194164, -0.23931138],
           [0.00000012, 0.00000004, 0.04523643]]

# monomer2 = [[-0.00032173,  -1.66883417,  -0.56979199],
#             [0.86680697,  -0.36416991,  -0.02964338],
#             [-0.86669085,  -0.36370215,  -0.02981743],
#             [-0.00008682,  -0.91448211,   0.02996010],
#             [22.55870573,   0.78106280,  -0.75977325],
#             [22.71587410,   0.37986377,   0.72158263],
#             [22.07903488,   0.45956205,   0.00727991]]

monomer2 = [[-0.00032173,  -1.66883417,  -0.56979199],
            [0.86680697,  -0.36416991,  -0.02964338],
            [-0.86669085,  -0.36370215,  -0.02981743],
            [-0.00008682,  -0.91448211,   0.02996010]]
import numpy as np
r = 1.84099353909
beta = np.deg2rad(73.16737677908)
monomer = np.flip([[0, 0, 0],
            [r*np.sin(beta), 0, r*np.cos(beta)],
            [-1/2*r*np.sin(beta), np.sqrt(3)/2*r*np.sin(beta), r*np.cos(beta)],
            [-1/2*r*np.sin(beta), -np.sqrt(3)/2*r*np.sin(beta), r*np.cos(beta)]], 0)/ang2bohr
# print(monomer)
r = 1.827793856
beta = np.deg2rad(90)
monomer4 = [[r*np.sin(beta), 0, r*np.cos(beta)],
            [-1/2*r*np.sin(beta), np.sqrt(3)/2*r*np.sin(beta), r*np.cos(beta)],
            [-1/2*r*np.sin(beta), -np.sqrt(3)/2*r*np.sin(beta), r*np.cos(beta)],
            [0, 0, 0]]


import numpy as np
np.save('trimer_coords', np.array(trimer)*ang2bohr)
np.save('tetramer_coords', np.array(tetramer)*ang2bohr)
np.save('dimer_coords', np.array(dimer)*ang2bohr)
np.save('monomer_coords', np.array(monomer)*ang2bohr)
np.save('monomer_coords_from_Anne', np.array(monomer2)*ang2bohr)

a = np.loadtxt('roh_morse.dat', usecols=(0, 3))

np.save('wvfns/free_oh_wvfn', a)


def dist(coords_initial, bonds):
    cd1 = coords_initial[:, tuple(x[0] for x in np.array(bonds)-1)]
    cd2 = coords_initial[:, tuple(x[1] for x in np.array(bonds)-1)]
    dists = np.linalg.norm(cd2-cd1, axis=2)
    return dists


trimer = [[-0.00032173,  -1.66883417,  -0.56979199],
          [0.86680697,  -0.36416991,  -0.02964338],
          [-0.86669085,  -0.36370215,  -0.02981743],
          [-0.00008682,  -0.91448211,   0.02996010],
          [2.55870573,   0.78106280,  -0.75977325],
          [2.71587410,   0.37986377,   0.72158263],
          [2.07903488,   0.45956205,   0.00727991],
          [-2.55611419,   0.78549017,  -0.75941312],
          [-2.71764621,   0.37748843,   0.71971598],
          [-2.07898673,   0.45950719,   0.00731627]]
# from ProtWaterPES import Potential
# a = np.array([monomer]*3)*ang2bohr
# b = np.array([monomer2]*3)*ang2bohr
# c = np.array([monomer3]*3)
# d = np.array([monomer4]*3)
# pot = Potential(4)
# print(pot.get_potential(a))
# print(pot.get_potential(b))
# print(pot.get_potential(np.flip(c, 1)))
# print(pot.get_potential(d))