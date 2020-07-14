import numpy as np
import matplotlib.pyplot as plt

ang2bohr = 1.e-10/5.291772106712e-11

hbond_wvfn = np.load("../wvfns/shared_prot_moveable_wvfn.npy")


def shift_calc(oo_dists, interp):
    if interp is None:
        return np.zeros(oo_dists.shape)
    else:
        f = np.poly1d(interp)
        oh_max = f(oo_dists)
        return oh_max


def scale_calc(oo_dists, interp):
    if interp is None:
        return np.ones(oo_dists.shape)
    else:
        f = np.poly1d(interp)
        oh_std = f(oo_dists)
        return oh_std


from scipy import interpolate


def dists(coords, num_waters, dist_type):
    if num_waters == 1:
        bonds = [[4, 1], [4, 2], [4, 3]]
    elif num_waters == 2:
        if dist_type == 'OH':
            bonds = [[4, 2], [4, 3], [7, 6], [7, 5]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 1]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7]]
    elif num_waters == 3:
        if dist_type == 'OH':
            bonds = [[4, 1], [7, 6], [7, 5], [10, 9], [10, 8]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 3], [4, 2]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7], [4, 10]]
        elif dist_type == 'OO':
            bonds = [[7, 10]]
    elif num_waters == 4:
        if dist_type == 'OH':
            bonds = [[7, 6], [7, 5], [10, 9], [10, 8], [13, 12], [13, 11]]
        elif dist_type == 'hbond_OH':
            bonds = [[4, 3], [4, 2], [4, 1]]
        elif dist_type == 'hbond_OO':
            bonds = [[4, 7], [4, 13], [4, 10]]
        elif dist_type == 'OO':
            bonds = [[7, 10], [7, 13], [10, 13]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds)-1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds)-1)]
    dis = np.linalg.norm(cd2-cd1, axis=2)
    return dis


# structure = np.load('../tetramer_coords.npy')
# coords = np.array([structure]*5)
#
# hbond_oo = dists(coords, 4, 'hbond_OO')
# hbond_oh = dists(coords, 4, 'hbond_OH')
#
# interp_hbond = interpolate.splrep(hbond_wvfn[:, 0], hbond_wvfn[:, 1], s=0)
#
# interp_OO_shift = np.array(np.loadtxt('bowman_h9o4_Re_Polynomials'))
# interp_OO_scale = np.array(np.loadtxt('bowman_h9o4_Std_Polynomials'))
#
# shift = shift_calc(hbond_oo, interp_OO_shift)
# scale = scale_calc(hbond_oo, interp_OO_scale)
# psi = interpolate.splev(scale*(hbond_oh-shift), interp_hbond, der=0)
# print(psi)
#
# interp_OO_shift_patch = np.array(np.loadtxt('bowman_patched_H9O4_Re_Polynomials'))
# interp_OO_scale_patch = np.array(np.loadtxt('bowman_patched_H9O4_Std_Polynomials'))
#
# shift_patch = shift_calc(hbond_oo, interp_OO_shift_patch)
# scale_patch = scale_calc(hbond_oo, interp_OO_scale_patch)
# psi = interpolate.splev(scale_patch*(hbond_oh-shift_patch), interp_hbond, der=0)
# print(psi)

ang2bohr = (1.e-10)/(5.291772106712e-11)

oo_scale_tet = np.array(np.loadtxt('bowman_patched_H9O4_Std_Polynomials'))
oo_shift_tet = np.array(np.loadtxt('bowman_patched_H9O4_Re_Polynomials'))

hbond_wvfn2 = np.load('../wvfns/free_oh_wvfn.npy')

x1 = (hbond_wvfn2[:, 0] - hbond_wvfn2[np.argmax(hbond_wvfn[:, 1]), 0])
x2 = (hbond_wvfn[:, 0])
hyd = np.load('../wvfns/hydronium_oh_wvfn.npy').T

arg = np.argmax(hyd[1, :])
print(hyd[0, arg]/ang2bohr)
#print(x2/x1)
#print(x1/x2)


def calculate_std(x, wvfn):
    return np.sqrt(np.dot(wvfn**2, x**2)-(np.dot(wvfn**2, x)**2))


print(calculate_std(hyd[0]/ang2bohr, hyd[1]/(np.sqrt(np.dot(hyd[1], hyd[1])))))
#print(calculate_std(x1, hbond_wvfn2[:, 1]))
new_wvfn = hbond_wvfn2[:, 1]/(np.sqrt(np.dot(hbond_wvfn2[:, 1], hbond_wvfn2[:, 1])))
#print(np.dot(new_wvfn, new_wvfn))
#print(np.dot(hbond_wvfn2[:, 1], hbond_wvfn2[:, 1]))
#print(np.dot(hbond_wvfn[:, 1], hbond_wvfn[:, 1]))
a = calculate_std(x1, new_wvfn)
#print(calculate_std(x1/a, hbond_wvfn2[:, 1]/np.dot(hbond_wvfn2[:, 1], hbond_wvfn2[:, 1])))
# print(hbond_wvfn2[:, 1]/hbond_wvfn[:, 1])
# print(x1/a/x2)
#print(calculate_std(a*x2, hbond_wvfn[:, 1]))

oo_scale_trim = np.array(np.loadtxt('bowman_h7o3_Std_Polynomials'))
oo_shift_trim = np.array(np.loadtxt('bowman_h7o3_Re_Polynomials'))

x = np.linspace(2.25, 3.55, 1000)*ang2bohr

scale_tet = np.poly1d(oo_scale_tet)
shift_tet = np.poly1d(oo_shift_tet)

scale_trim = np.poly1d(oo_scale_trim)
shift_trim = np.poly1d(oo_shift_trim)

# plt.plot(x1/ang2bohr, hbond_wvfn2[:, 1]/np.sqrt(np.dot(hbond_wvfn2[:, 1], hbond_wvfn2[:, 1])), label='free oh')
# plt.plot(x1/ang2bohr, hbond_wvfn2[:, 1])
# plt.plot(a*x2/ang2bohr, hbond_wvfn[:, 1], label='moveable')
#
# plt.ylabel(r'$\rm\Psi(r_{OH})$', fontsize=22)
# plt.xlabel(r'r$_{\rmOH} (\rm\AA)$', fontsize=22)
# plt.xlim(0.5, 1.5)
# plt.legend()
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#                      bottom=True, top=False, left=True, right=False, labelsize=14)
# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(1, 2)

# ax2 = ax1.twinx()

axes[0].plot(x/ang2bohr, shift_tet(x)/ang2bohr, color='purple', linewidth=4.0, label=r'$\rmH_9O_4^+$')
axes[0].plot(x/ang2bohr, shift_trim(x)/ang2bohr, color='blue', linewidth=4.0, label=r'$\rmH_7O_3^+$')
# ax1.scatter(6, 0.98499, color='black', label='hydronium rmax')
# ax2.scatter(6, 0.070644, color='purple', label='hydronium std')
axes[1].plot(x/ang2bohr, scale_tet(x)/ang2bohr, color='purple', linewidth=4.0, label = r'$\rmH_9O_4^+$')
axes[1].plot(x/ang2bohr, scale_trim(x)/ang2bohr, color='blue', linewidth=4.0, label=r'$\rmH_7O_3^+$')
axes[0].set_xlabel(r'R$_{\rm{OO}} (\rm\AA)$', fontsize=22)
axes[1].set_xlabel(r'R$_{\rm{OO}} (\rm\AA)$', fontsize=22)
axes[0].set_ylabel(r'r$_{\rm{OH}}^{\rm{max}} (\rm\AA)$', fontsize=22)
axes[1].set_ylabel(r'$\rm\sigma_{OH} (\rm\AA)$', fontsize=22)

leg = axes[0].legend(fontsize=14)
leg.get_frame().set_edgecolor('white')
# ax2.legend()
axes[0].tick_params(axis='y', labelleft=True, labelright=False,
                    left=True, right=False, labelsize=14)
axes[0].tick_params(axis='x', labelbottom=True, labeltop=False, bottom=True, top=False, labelsize=14)
axes[1].tick_params(axis='x', labelbottom=True, labeltop=False, bottom=True, top=False, labelsize=14)
axes[1].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True,
                     bottom=True, top=False, left=False, right=True, labelsize=14)
axes[1].yaxis.set_label_position('right')
plt.tight_layout()
plt.show()
