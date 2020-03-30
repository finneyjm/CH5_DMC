import matplotlib.pyplot as plt
from Coordinerds.CoordinateSystems import *
import CH5pot

coords_initial_min = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                  [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                  [1.786540362044548, -1.386051328559878, 0.000000000000000],
                  [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                  [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                  [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])
coords_initial_cs = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                       [1.931652478009080, -4.5126502395556294E-008, -0.6830921182334913],
                       [5.4640011799588715E-017, 0.8923685824271653, 2.083855680290835],
                       [-5.4640011799588715E-017, -0.8923685824271653, 2.083855680290835],
                       [-1.145620108130841, -1.659539840225091, -0.4971351597887673],
                       [-1.145620108130841, 1.659539840225091, -0.4971351597887673]])
coords_initial_c2v = np.array([[0.000000000000000, 0.000000000000000, 0.386992362158741],
                       [0.000000000000000, 0.000000000000000, -1.810066283748844],
                       [1.797239666982623, 0.000000000000000, 1.381637275550612],
                       [-1.797239666982623, 0.000000000000000, 1.381637275550612],
                       [0.000000000000000, -1.895858229423645, -0.6415748897955779],
                       [0.000000000000000, 1.895858229423645, -0.6415748897955779]])
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0000000000 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_red = (m_C*m_H)/(m_C+m_H)
m_red_D = (m_C*m_D)/(m_C+m_D)
har2wave = 219474.6
ang2bohr = (1.e-10)/(5.291772106712e-11)


def grid(a, b, N, CH, coords_initial):
    spacing = np.linspace(a, b, num=N)
    if CH == 1:
        new_coords = CoordinateSet(coords_initial, system=CartesianCoordinates3D)
        new_coords = new_coords.convert(ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
        g = np.array([new_coords]*N)
        g[:, 1, 0] = spacing
    if CH is not 1:
        sub = np.array([coords_initial[1]])
        coords_initial[1] = coords_initial[CH]
        coords_initial[CH] = sub
        new_coords = CoordinateSet(coords_initial, system=CartesianCoordinates3D)
        new_coords = new_coords.convert(ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
        g = np.array([new_coords] * N)
        g[:, 1, 0] = spacing
    else:
        g = 0
    return g


def Potential(grid, CH):
    V = CH5pot.mycalcpot(grid, len(grid[:, 0, 0]))
    V_final = np.diag(np.array(V))
    return V_final


def Kinetic_Calc(grid, m_red):
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


def run(CH, type, coords, mass):
    g = grid(0.4, 6., 1000, CH, coords)
    V = Potential(g, CH)
    T = Kinetic_Calc(g, mass)
    En, Eig = Energy(T, V)
    print(En[0]*har2wave)
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    if CH == 2:
    # plt.plot(g[:, 1, 0], np.diag(V)*har2wave)
    # plt.plot(g[:, 1, 0], (Eig[:, 0])*50000 + En[0]*har2wave)
        plt.plot(g[:, 1, 0]/ang2bohr, Eig[:, 0])
        plt.xlim(0.6, 1.8)
    # plt.ylim(0, 20000)
        plt.xlabel(r'r$_{CH}$ ($\AA$)', fontsize=16)
        plt.ylabel('P(r)', fontsize=16)
    # plt.show()
    # plt.close()
        plt.savefig('GSW_2_min_wvfn_for_ppt.png')
    # np.save(f'GSW_{type}_CD_{CH}', Eig[:, 0])
    return g[:, 1, 0], Eig[:, 0]


# wvfns = [1, 2, 3, 4, 5]
# wvfn = np.zeros((5, 1000))
# for i in np.arange(1, 6):
# for i in wvfns:
#     g, wvfn[i-1] = run(i, 'min', coords_initial_min, m_red)
# for i in range(5):
#     plt.plot(g/ang2bohr, wvfn[i], label=f'Ground State Wave function {i+1}')
# plt.xlim(0.6, 1.8)
# plt.xlabel(r'r$_{CH}$ ($\AA$)', fontsize=16)
# plt.ylabel('P(r)', fontsize=16)
# plt.legend(bbox_to_anchor=(1.04, 1.), fontsize=14)
# plt.tight_layout(rect=[0, 0, 0.75, 1])
# plt.savefig('GSWs_min_for_ppt.png', bbox_inches='tight')
# run(i, 'cs', coords_initial_cs, m_red_D)
# run(i, 'c2v', coords_initial_c2v, m_red_D)
g = grid(0.4, 6, 5000, 5, coords_initial_min)
a = np.linspace(0.4, 6, 5000)
V = np.diag(Potential(g, 5))
plt.plot(a/ang2bohr, V)
plt.xlim(0.6, 2.5)
plt.ylim(0, 7000)
# plt.show()
np.save('Potential_CH_stretch5', V)

