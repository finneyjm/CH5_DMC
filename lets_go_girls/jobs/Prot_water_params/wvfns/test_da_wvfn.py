import numpy as np


a= np.load('free_oh_wvfn.npy')
from scipy import interpolate

interp = interpolate.splrep(a[:, 0], a[:, 1])

f = interpolate.splev(np.linspace(0.9, 1.1, 10), interp, der=0)