import numpy as np
import matplotlib.pyplot as plt

ang2bohr = 1.e-10/5.291772106712e-11

free_oh = np.load('../wvfns/free_oh_wvfn.npy')
free_oh[:, 1] = free_oh[:, 1]/np.linalg.norm(free_oh[:, 1])


def expec(op, wvfn):
    return np.dot(wvfn**2, op)


def std_dev(grid, wvfn):
    exp_x = expec(grid, wvfn)
    exp_x_sq = expec(grid**2, wvfn)
    return np.sqrt(exp_x_sq - exp_x**2)


std = std_dev(free_oh[:, 0], free_oh[:, 1])
ind = np.argmax(free_oh[:, 1])
scaled_x = free_oh[:, 0]/std
shifted_x = free_oh[:, 0]-free_oh[ind, 0]
new_x = (1./std)*(free_oh[:, 0]-free_oh[ind, 0])

print(std_dev(new_x, free_oh[:, 1]))

plt.plot(new_x, free_oh[:, 1], label='bare_bones')
plt.plot(free_oh[:, 0], free_oh[:, 1], label='straigh outa compton')
plt.plot(shifted_x, free_oh[:, 1], label='just a scootchy')
newer_x = 0.085*new_x
plt.plot(newer_x, free_oh[:, 1], label='newer_x')
# plt.legend()
# plt.show()
# plt.close()

np.save('../wvfns/shared_prot_moveable_wvfn', np.vstack((new_x, free_oh[:, 1])).T)

asdf = np.vstack((new_x, free_oh[:, 1])).T
a = np.array(np.loadtxt('bowman_h7o3_Re_Polynomials'))
print(a)

from scipy import interpolate

blah = interpolate.splrep(asdf[:, 0], asdf[:, 1], s=0)

