import numpy as np


def exportCoords(cds, fn):  # for partridge schwinke
    cds = np.array(cds)
    fl = open(fn, "w+")
    fl.write('%d\n' % len(cds))
    for i in range(len(cds)):  # for a walker
        for j in np.flip(range(len(cds[i]))):  # for a certain # of atoms
            fl.write('%5.16f %5.16f %5.16f\n' % (cds[i, j, 0], cds[i, j, 1], cds[i, j, 2]))
    fl.close()


def PatrickShinglePotential(walkerSet, calc_number, sim=None):
    import subprocess as sub
    exportCoords(walkerSet, f'PES{calc_number}' + '/hoh_coord.dat')
    proc = sub.Popen('./calc_h2o_pot', cwd=f'PES{calc_number}')
    proc.wait()
    bigPotz = np.loadtxt(f'PES{calc_number}' + '/hoh_pot.dat')
    return bigPotz
