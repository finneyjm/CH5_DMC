import numpy as np
from Coordinerds.CoordinateSystems import *

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
water = np.load('monomer_coords.npy')
order_w = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0]]
hydronium = np.flip(np.load('../../lets_go_girls/jobs/Prot_water_params/monomer_coords.npy'))
order_h = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2]]
order_t = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 4, 1, 2], [6, 4, 1, 2],
           [7, 0, 1, 2], [8, 7, 1, 2], [9, 7, 1, 2]]
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
OH_red = (m_O*m_H) / (m_O + m_H)
OD_red = (m_O*m_D) / (m_O + m_D)


def grid_angle(a, b, num, coords):
    spacing = np.linspace(a, b, num)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    g = np.array([zmat]*num)
    g[:, 1, 3] = spacing
    new_coords = CoordinateSet(g, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


def grid_dis(a, b, num, coords):
    spacing = np.linspace(a, b, num)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    g = np.array([zmat]*num)
    g[:, 0, 1] = spacing
    new_coords = CoordinateSet(g, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


def grid_dis_water(a, b, num, coords):
    spacing = np.linspace(a, b, num)
    g = np.array([coords]*num)
    g[:, 1, 0] = spacing
    return g


def oh_dists(coords):
    bonds = [[1, 2], [1, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def linear_combo_stretch_grid(r1, r2, coords):
    re = np.linalg.norm(coords[0]-coords[1])
    re2 = np.linalg.norm(coords[0]-coords[2])
    coords = np.array([coords] * 1)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).squeeze()
    zmat[:, 0, 1] = re + r1
    zmat[:, 1, 1] = re2 + r2
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


def potential_bare_water(grid):
    from Water_monomer_pot_fns import PatrickShinglePotential as pot
    V = pot(grid)
    return np.diag(np.array(V))


def p_water_trimer(grid):
    from ProtWaterPES import Potential
    pot = Potential(10)
    trimer = [[1.822639226956, 0.000000000283, 0.000000000000],
              [0.300407646428, -0.914198709660, 0.000000000000],
              [0.300407645812, 0.914198709862, 0.000000000000],
              [0.840815416177, 0.000000000283, 0.000000000000],
              [-0.714399465504, 2.673546910582, -0.785835530027],
              [-0.714326750385, 2.673421907097, 0.785943106958],
              [-0.420407708818, 2.168150028133, 0.000000000000],
              [-0.714326748587, -2.673421907578, -0.785943106958],
              [-0.714399463705, -2.673546911063, 0.785835530027],
              [-0.420407707359, -2.168150028415, -0.000000000000]]
    coords = np.array([trimer]*len(grid))*ang2bohr
    coords = coords[:, [3, 0, 1, 2, 6, 4, 5, 9, 7, 8]]
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order_t).coords
    zmat[:, 1, 1] = grid
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    new_coords = new_coords[:, [1, 2, 3, 0, 5, 6, 4, 8, 9, 7]]
    V = pot.get_potential(new_coords)
    return V


def harmonic(grid):
    omega_OH = 3890.7865072878913/har2wave
    mass = OH_red
    return 0.5*mass*omega_OH**2*(grid-0.968)**2


def potential_hydronium(grid):
    from ProtWaterPES import Potential
    pot = Potential(4)
    V = np.diag(pot.get_potential(np.flip(grid, axis=1)))
    return V


def kin(grid, a, b, m_red):
    N = len(grid)
    coeff = (1/((2*m_red)/(((float(N)-1)/(b-a))**2)))

    Tii = np.zeros(N)

    Tii += coeff*((np.pi**2)/3)
    T_initial = np.diag(Tii)
    for i in range(1, N):
         for j in range(i):
              T_initial[i, j] = coeff*((-1)**(i-j))*(2/((i-j)**2))
    T_final = T_initial + T_initial.T - np.diag(Tii)
    return T_final


def Energy(T, V):
    H = T + V
    # import matplotlib.pyplot as plt
    # plt.imshow(T)
    # plt.show()
    En, Eigv = np.linalg.eigh(H)
    ind = np.argsort(En)
    En = En[ind]
    Eigv = Eigv[:, ind]
    return En, Eigv


def run(g, a, b, mass, mc):
    if mc == 'water':
        V = potential_bare_water(g)
    elif mc == 'Harmonic':
        V = np.diag(harmonic(g))
    elif mc == 'Entos':
        V = np.diag(np.load('water_OH_pot.npy')[1] + 76.342463064)
    else:
        V = potential_hydronium(g)
    T = kin(g, a, b, mass)
    En, Eig = Energy(T, V)
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    return En, Eig, np.diag(V)


def angle(coords):
    dists = oh_dists(coords)
    v1 = (coords[:, 1] - coords[:, 0]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
    v2 = (coords[:, 2] - coords[:, 0]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))

    ang1 = np.arccos(np.matmul(v1[:, None, :], v2[..., None]).squeeze())

    return ang1.T


theta = np.deg2rad(104.1747712)
g = grid_angle(theta-1, theta+1, 50, water)
struct1 = g[20]
struct2 = g[30]

anti = np.linspace(-1, 1, 50)
sym = np.linspace(-1, 1, 50)
A = 1/np.sqrt(2)*np.array([[-1, 1], [1, 1]])
eh = np.matmul(np.linalg.inv(A), np.vstack((anti, sym)))
r1 = eh[0]
r2 = eh[1]

g1 = linear_combo_stretch_grid(r1, r2, struct1)
struct1 = g1[20]

g2 = linear_combo_stretch_grid(r1, r2, struct2)
struct2 = g2[30]

print(struct1)

dists = oh_dists(np.array([struct1]*1))
anti = 1 / np.sqrt(2) * (dists[:, 1] - dists[:, 0])
sym = 1 / np.sqrt(2) * (dists[:, 1] + dists[:, 0])
ang = angle(np.array([struct1]*1))
print(anti, sym, ang)

print(struct2)

dists = oh_dists(np.array([struct2]*1))
anti = 1 / np.sqrt(2) * (dists[:, 1] - dists[:, 0])
sym = 1 / np.sqrt(2) * (dists[:, 1] + dists[:, 0])
ang = angle(np.array([struct2]*1))
print(anti, sym, ang)


import matplotlib.pyplot as plt
import scipy.sparse as sp
num_points = 2000
anti = np.linspace(-1, 1, num_points)
# anti = np.zeros(num_points)
# sym = np.linspace(-1, 1, num_points)
sym = np.zeros(num_points)
A = 1/np.sqrt(2)*np.array([[-1, 1], [1, 1]])
eh = np.matmul(np.linalg.inv(A), np.vstack((anti, sym)))
r1 = eh[0]
r2 = eh[1]

lin_combo_grid = linear_combo_stretch_grid(r1, r2, water)

# ang = angle(np.array([water]*1)).squeeze()
# r = 0.9616036495623883*ang2bohr
# m1 = 1/(1/m_H + (1+np.cos(ang))/m_O - np.sqrt(2)*np.sin(ang)/(r*m_O))
# m2 = 1/(-2/r**2*(1/m_H - 1/m_O - np.cos(ang)/m_O) - np.sqrt(2)*np.sin(ang)/(r*m_O))
# m1 = 1/(1/m_O + 1/m_H + np.cos(ang)/m_O - np.sin(ang)/(r*m_O))
# anti_freq = 3944.27284814
anti_gmat_one_over = 1702.9703137654326
# m1 = anti_freq/2
en_wat, eig_wat, v = run(lin_combo_grid, anti[0], anti[-1], anti_gmat_one_over, 'water')
print((en_wat[1]-en_wat[0])*har2wave)
np.savez('antisymmetric_stretch_water_wvfns', grid=anti, ground=eig_wat[:, 0], excite=eig_wat[:, 1])
# ind = np.argmin(v)
# dx = (anti[-1] - anti[0])/num_points
# coeffs = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])/dx**2
# fd_mat = sp.diags(coeffs, np.arange(-3, 4, 1), shape=(len(v), len(v))).toarray()
# second_der = np.dot(fd_mat, v)[3:-3]
# freqs = np.sqrt(g_el*second_der)*har2wave
# print(1/2*freqs[ind])

# anti = np.linspace(-1, 1, num_points)
anti = np.zeros(num_points)
sym = np.linspace(-1, 1, num_points)
# sym = np.zeros(num_points)
A = 1/np.sqrt(2)*np.array([[-1, 1], [1, 1]])
eh = np.matmul(np.linalg.inv(A), np.vstack((anti, sym)))
r1 = eh[0]
r2 = eh[1]

lin_combo_grid = linear_combo_stretch_grid(r1, r2, water)

# ang = angle(np.array([water]*1)).squeeze()
# r = 0.9616036495623883*ang2bohr
# m1 = 1/(1/m_H + (1+np.cos(ang))/m_O - np.sqrt(2)*np.sin(ang)/(r*m_O))
# m1 = 1/(1/m_O + 1/m_H + np.cos(ang)/m_O - np.sin(ang)/(r*m_O))
# m2 = 1/(2/r**2*(1/m_H + 1/m_O - np.cos(ang)/m_O) - np.sqrt(2)*np.sin(ang)/(r*m_O))
# sym_freq = 3832.70931812
sym_gmat_one_over = 1754.307807821817
en_wat, eig_wat, v = run(lin_combo_grid, sym[0], sym[-1], sym_gmat_one_over, 'water')
print((en_wat[1]-en_wat[0])*har2wave)
np.savez('symmetric_stretch_water_wvfns', grid=sym, ground=eig_wat[:, 0], excite=eig_wat[:, 1])

# ind = np.argmin(v)
# dx = (sym[-1] - sym[0])/num_points
# coeffs = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])/dx**2
# fd_mat = sp.diags(coeffs, np.arange(-3, 4, 1), shape=(len(v), len(v))).toarray()
# second_der = np.dot(fd_mat, v)[3:-3]
# freqs = np.sqrt(g_el*second_der)*har2wave
# print(1/2*freqs[ind])


shift = 1.818131256952169-1.8097886540600667
print(shift/ang2bohr)
a = 0.5*ang2bohr
b = 2*ang2bohr
num = 900
print(a/ang2bohr)
print(b/ang2bohr)
g_water = grid_dis_water(a, b, num, water)
g_hyd = grid_dis(a, b, 1000, hydronium, order_h)
g = np.linspace(0.5*ang2bohr, 2*ang2bohr, 900)
v = p_water_trimer(g)
# v2 = np.diag(potential_bare_water(g_water))
en_water, eig_water, v2 = run(g_water, a, b, OH_red, 'water')
import scipy.interpolate
interp = scipy.interpolate.splrep(g, eig_water[:, 0], s=0)
der_2 = scipy.interpolate.splev(g, interp, der=2)
psi = scipy.interpolate.splev(g, interp, der=0)
el_kin = -1/(2*OH_red)*(der_2/psi)
v = v-np.min(v)
g /= ang2bohr
avg = 0.981573
std = 0.007277
max = 1.007023
min = 0.970902
Ohs = np.array([min, max, avg-std, avg+std, avg])
fig, ax = plt.subplots()
line, = ax.plot(g, eig_water[:, 0], color='blue', label='Ground State Wave Function')
bp = ax.boxplot(Ohs, positions=[np.max(eig_water[:, 0])/2], widths=[0.01], vert=0, manage_ticks=False)
plt.xlabel(r'r$_{\rm{OH}}$ [/$\rm\AA$]', fontsize=20)
plt.ylabel(r'$\rm{\Psi_0^{HOH}(r_{OH})}$', fontsize=20)
plt.xlim(0.5, 2.0)
# plt.legend((line, bp['boxes'][0]), (r'$\rm{\Psi_0^{HOH}(r_{OH})}$', r'r$_{\rm{OH}}$ in (H$_2$O)$_{n=1-6}$'),
#            loc='upper right', fontsize=12)
ax.tick_params(axis='both', labelsize=18)
plt.tight_layout()
plt.savefig('water_plot_for_Anne')
plt.show()


plt.plot(g, (v2)*har2wave, label='Water Potential', linewidth=2, color='blue')
plt.plot(g, (el_kin+v2)*har2wave, label='Local Energy', linewidth=2, color='purple')
# plt.plot(g, (v)*har2wave, label=r'H$_7$O$_3^+$ Potential', linewidth=2, color='red')
# plt.plot(g, (el_kin+v)*har2wave, label='Local Energy', linewidth=2, color='green')
# plt.plot(g, eig_water[:, 0], label='Water Ground State Wave Function', linewidth=2, color='purple')
plt.ylim(-12000, 15000)
# plt.ylim(0, 0.1)
plt.xlim(1.2/ang2bohr, 3/ang2bohr)
plt.xlabel(r'r$_{\rm{OH}}$ ($\rm\AA$)', fontsize=18)
plt.ylabel(r'Energy cm$^{-1}$', fontsize=18)
# plt.ylabel('Probability Density', fontsize=18)
plt.tick_params(labelsize=14)
plt.legend(loc='lower left', fontsize=14)
plt.tight_layout()
plt.savefig('water_pot_eloc.png')
plt.show()
# print('hello')
# print(g_water[84])
# r1 = 0
# for i in range(3):
#     d1 = g_water[84, 2, i]-g_water[84, 0, i]
#     r1 = r1 + d1**2
# print(np.sqrt(r1))
entos_wvfn = np.load('free_oh_wvfn_entos_t.npy')
entos_pots = np.load('water_OH_pot.npy')
en_water, eig_water, V = run(g_water, a, b, OH_red, 'water')
entos_en, entos_eig, entos_V = run(g_water, a, b, OH_red, 'Entos')
print(entos_en[0]*har2wave)
print((entos_en[1]-entos_en[0])*har2wave)
print(en_water[0]*har2wave)
print((en_water[1]-en_water[0])*har2wave)
# g = np.linspace(0.3, 1.7, 100)
# en_harm, eig_harm, V_harm = run(g, g[0], g[-1], OH_red, 'Harmonic')
# np.save('harmonic_oh_stretch_GSW', np.vstack((g, eig_harm[:, 0])))
# print(en_harm[0]*har2wave)
# print(en_harm[1]*har2wave-en_harm[0]*har2wave)
# import matplotlib.pyplot as plt
# plt.plot(g, V_harm*har2wave)
# plt.plot(g, eig_harm[:, 0]*5000 + en_harm[0]*har2wave)
# plt.show()

print(np.dot(entos_wvfn[:, 1], eig_water[:, 0]))
print(np.linspace(a, b, 900)[np.argmin(V)])
print(entos_wvfn[np.argmin(entos_pots[1]), 0])
print(np.argmin(V))
print(np.argmin(entos_pots[1]))

min_entos_energy = -76.342463064
entos_pots[1] -= min_entos_energy
# entos_pots[0] = entos_pots[0]*ang2bohr
diff_pot = entos_pots[1]-V
percent_diff_pot = diff_pot/V*100
g = np.linspace(a, b, num)
import matplotlib.pyplot as plt
plt.plot(entos_wvfn[:, 0]/ang2bohr, (entos_wvfn[:, 1])**2, label='MOB-ML')
plt.plot(g/ang2bohr, (eig_water[:, 0])**2, label='PS')
plt.xlabel(r'r$_{\rmOH}$ $\rm\AA$', fontsize=16)
plt.ylabel(r'P(r$_{\rmOH})$', fontsize=16)
plt.tick_params(axis='both', labelsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

plt.plot(entos_pots[0]/ang2bohr, entos_pots[1]*har2wave, label='Entos Energy')
plt.plot(g/ang2bohr, V*har2wave, label='Partridge Schwenke Energy')
plt.xlabel(r'r$_{\rmOH}$ $\rm\AA$', fontsize=16)
plt.ylabel(r'Energy cm$^{-1}$', fontsize=16)
plt.tick_params(axis='both', labelsize=18)
plt.ylim(0, 10000)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
plt.plot(g/ang2bohr, diff_pot*har2wave)
plt.xlabel(r'r$_{\rmCH}$ $\rm\AA$', fontsize=16)
plt.ylabel('Absolute Difference of \nEntos Energy from \nPartridge Schwenke Energy', fontsize=16)
plt.tick_params(axis='both', labelsize=18)
# plt.ylim(-4, 4)
plt.show()


# print('dvr')
print(en_water[1]*har2wave-en_water[0]*har2wave)
print(en_water[0]*har2wave)
dx = (b-a)/num
ind = np.argmin(V)
second_der = ((1/90*V[ind-3] - 3/20*V[ind-2] + 3/2*V[ind-1] - 49/18*V[ind] + 3/2*V[ind+1] - 3/20*V[ind+2] + 1/90*V[ind+3])/dx**2)
print(np.sqrt(second_der/OD_red)*har2wave)
print(np.sqrt(second_der/OH_red)*har2wave)
en_hyd, eig_hyd, V = run(g_hyd, a, b, OH_red, 'else')

x = np.linspace(a, b, 1000)
# print(x)
# print(np.diag(potential_bare_water(g_water)))
shift = x[np.argmax(eig_water[:, 0])] - x[np.argmax(eig_hyd[:, 0])]
od_max = x[np.argmax(eig_water[:, 0])]


def calculate_std(x, wvfn):
    return np.sqrt(np.dot(wvfn**2, x**2)-(np.dot(wvfn**2, x)**2))

a = calculate_std(x, eig_water[:, 0])
print(a)


# shift = 0
x_new = x - shift
# print(eig_water[:, 0])
import matplotlib.pyplot as plt
# plt.plot(x, np.diag(potential_bare_water(g_water))*har2wave, label='water pot')
# plt.plot(x, eig_water[:, 0], label='water')
# plt.plot(x, eig_hyd[:, 0], label='hydronium')
# plt.legend()
# plt.show()
# np.save('Water_od_stretch_GSW', np.vstack((x, eig_water[:, 0])).T)
# np.save('../../lets_go_girls/jobs/Prot_water_params/wvfns/shared_deuterium_moveable_wvfn', np.vstack((x/a, eig_water[:, 0])).T)
# def gmat(mu1, mu2, mu3, r1, r2, ang):
#     return mu1/r1**2 + mu2/r2**2 + mu3*(1/r1**2 + 1/r2**2 - 2*np.cos(ang)/(r1*r2))
#
#
# r1 = 0.95784*ang2bohr
# r2 = 0.95784*ang2bohr
# # r1 = 0.95784
# # r2 = 0.95784
# num = 10000
# theta = np.deg2rad(104.508)
# # theta = 104.5080029
# muH = 1/m_H
# muD = 1/m_D
# muO = 1/m_O
# # muH = 1/1.00782503223
# # muD = 1/2.01410177812
# # muO = 1/15.99491461957
# g = grid(0, np.pi, num)
# V = np.diag(potential(g))
#
# import matplotlib.pyplot as plt
# # plt.plot(np.linspace(0, 2*np.pi, 1000), V*har2wave)
# # plt.show()
# # plt.close()
# x = np.linspace(0, np.pi, num)
#
# ghh = gmat(muH, muH, muO, r1, r2, theta)
# ghd = gmat(muH, muD, muO, r1, r2, theta)
# gdd = gmat(muD, muD, muO, r1, r2, theta)
#
# print(ghh)
# print(ghd)
# print(gdd)
#
# ind = np.argmin(V)
# dx = np.pi/num
# print(dx)
# print(V[ind])
# print(x[ind])
# print(np.rad2deg(x[ind]))
#
# print(CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords[:, 1]/ang2bohr)
#
# second_der = ((1/90*V[ind-3] - 3/20*V[ind-2] + 3/2*V[ind-1] - 49/18*V[ind] + 3/2*V[ind+1] - 3/20*V[ind+2] + 1/90*V[ind+3])/dx**2)
#
# freq_hh = np.sqrt(ghh*second_der)
# freq_hd = np.sqrt(ghd*second_der)
# freq_dd = np.sqrt(gdd*second_der)
#
# print(freq_hh*har2wave)
# print(freq_hd*har2wave)
# print(freq_dd*har2wave)
#
#
#
# def new_grid(a, b, num):
#     spacing = np.linspace(a, b, num)
#     zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
#     g = np.array([zmat] * num)
#     g[:, 1, 3] = spacing
#     new_coords = CoordinateSet(g, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     return new_coords

