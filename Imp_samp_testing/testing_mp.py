import numpy as np
from itertools import repeat
import multiprocessing as mp


def asdf(one, a=None):
    if a is None:
        print('my dad never loved me')
    elif a == 1:
        print('I am redeemed!')
    return 4


pool = mp.Pool(4)
fd = np.random.random((12, 12, 3))
gs = np.array_split(fd, 4)
v = pool.map(asdf, gs)
print(v)
v = pool.starmap(asdf, zip(gs, repeat(1)))
print(v)




