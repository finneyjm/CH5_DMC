import numpy as np


def exportCoords(cds):  # for partridge schwinke
    cds = np.array(cds)
    fl = open('coord.dat', "w+")
    fl.write('%d\n' % len(cds))
    for i in range(len(cds)):  # for a walker
        for j in np.flip(range(len(cds[i]))):  # for a certain # of atoms
            fl.write('%5.16f %5.16f %5.16f\n' % (cds[i, j, 0], cds[i, j, 1], cds[i, j, 2]))
    fl.close()


def Dimer_potential(walkerSet):
    import subprocess as sub
    exportCoords(walkerSet)
    proc = sub.Popen('./h5_wrapper')
    proc.wait()
    bigPotz = np.loadtxt(f'eng.dat')
    return bigPotz