import numpy as np
import matplotlib.pyplot as plt
from Coordinerds.CoordinateSystems import *
import CH5pot

# coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
#                   [0.1318851447521099, 2.088940054609643, 0.000000000000000],
#                   [1.786540362044548, -1.386051328559878, 0.000000000000000],
#                   [2.233806981137821, 0.3567096955165336, 0.000000000000000],
#                   [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
#                   [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
# coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
#                        [1.931652478009080, -4.5126502395556294E-008, -0.6830921182334913],
#                        [5.4640011799588715E-017, 0.8923685824271653, 2.083855680290835],
#                        [-5.4640011799588715E-017, -0.8923685824271653, 2.083855680290835],
#                        [-1.145620108130841, -1.659539840225091, -0.4971351597887673],
#                        [-1.145620108130841, 1.659539840225091, -0.4971351597887673]])
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.386992362158741],
                       [0.000000000000000, 0.000000000000000, -1.810066283748844],
                       [1.797239666982623, 0.000000000000000, 1.381637275550612],
                       [-1.797239666982623, 0.000000000000000, 1.381637275550612],
                       [0.000000000000000, -1.895858229423645, -0.6415748897955779],
                       [0.000000000000000, 1.895858229423645, -0.6415748897955779]])
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0106 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_red = (m_C*m_H)/(m_C+m_H)
har2wave = 219474.6
ang2bohr = (1.e-10)/(5.291772106712e-11)


def grid(a, b, N, CH):
    spacing = np.linspace(a, b, num=N)
    if CH == 1:
        new_coords = CoordinateSet(coords_initial, system=CartesianCoordinates3D)
        new_coords = new_coords.convert(ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
        g = np.array([new_coords]*N)
        g[:, 1, 0] += spacing
    if CH is not 1:
        sub = np.array([coords_initial[1]])
        coords_initial[1] = coords_initial[CH]
        coords_initial[CH] = sub
        new_coords = CoordinateSet(coords_initial, system=CartesianCoordinates3D)
        new_coords = new_coords.convert(ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
        g = np.array([new_coords] * N)
        g[:, 1, 0] = spacing
    return g


def Potential(grid, CH):
    V = CH5pot.mycalcpot(grid, len(grid[:, 0, 0]))
    V_final = np.diag(np.array(V))
    # plt.plot(grid[:, 1, 0]/ang2bohr, np.diag(V_final)*har2wave, label='CH stretch %s' %CH, color='C%s' %(CH+1))
    # plt.legend(loc=2)
    # plt.xlabel('Bond Distance (Angstrom)')
    # plt.ylabel(r'Energy (cm${-1}$)')
    # plt.ylim(0, 5000)
    # plt.xlim(0.75, 1.5)

    return V_final


def Kinetic_Calc(grid):
    a = grid[0, 1, 0]
    b = grid[-1, 1, 0]
    N = len(grid[:, 0, 0])
    coeff = (1./((2.*m_red)/(((float(N)-1.)/(b-a))**2)))

    Tii = np.zeros(N)

    Tii += coeff*((np.pi**2.)/3.)
    T_initial = np.diag(Tii)
    for i in range(1, N):
        for j in range(i):
            T_initial[i, j] = coeff*((-1.)**(i-j))*(2./((i-j)**2))
    T_final = T_initial + T_initial.T - np.diag(Tii)
    return T_final


def Energy(T, V):
    H = (T + V)
    En, Eigv = np.linalg.eigh(H)
    ind = np.argsort(En)
    En = En[ind]
    Eigv = Eigv[:, ind]
    return En, Eigv


def run(CH):
    g = grid(1., 4., 500, CH)
    V = Potential(g, CH)
    T = Kinetic_Calc(g)
    En, Eig = Energy(T, V)
    # plt.plot(g[:, 1, 0]/ang2bohr, np.array([En[0]]*50)*har2wave, color='C%s' %(CH+1))
    # plt.savefig('Potential_CH_stretch_c2v_saddle%s.png' % CH)
    print(En[0]*har2wave)
    # np.save('CH_stretch_wavefunction%s' %CH, Eig[:, 0])
    # plt.plot(g[:, 1, 0]/ang2bohr, -Eig[:, 0], label='Ground State Wavefunction CH stretch %s' %CH)
    # plt.xlabel('Bond Distance (Angstrom)')
    # plt.ylabel('Probability Density')
    # plt.legend()
    # plt.savefig('GSW%s.png' %CH)
    return g, Eig[:, 0]


wvfn = np.zeros((5, 500))
for i in np.arange(1, 6):
    g, wvfn[i-1, :] = run(i)

av_wvfn = np.mean(wvfn, axis=0)
avg_wvfn = np.vstack((g[:, 1, 0], av_wvfn))
# np.save('Average_GSW_CH_stretch', avg_wvfn)


# plt.legend(loc=3)
# plt.savefig('Avg_GSW.png')
for i in range(7):
    s_point = 1.0 + 0.05*float(i)
    switch = (np.tanh((g[:, 1, 0]-s_point*ang2bohr)) + 1.)*0.5
    new_wvfn = wvfn[0, :]*switch + wvfn[1, :]*(1.-switch)
    new_wvfn = new_wvfn/np.linalg.norm(new_wvfn)

    plt.plot(g[:, 1, 0] / ang2bohr, av_wvfn, label='Average Ground State Wavefunction')
    plt.plot(g[:, 1, 0]/ang2bohr, new_wvfn, label='Switch wvfn')
    plt.plot(g[:, 1, 0]/ang2bohr, wvfn[0, :], label='CH 1')
    plt.plot(g[:, 1, 0]/ang2bohr, wvfn[1, :], label='CH 2')
    plt.legend()
    plt.savefig('Switch_c2v_wvfn_%s.png' %s_point)
    plt.close()

    new_wvfn = np.vstack((g[:, 1, 0], new_wvfn))
    np.save('Switch_c2v_wvfn_%s' %s_point, new_wvfn)
