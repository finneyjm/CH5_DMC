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
                       [0.000000000000000, -1.895858229423645, -0.6415748897955779],
                       [0.000000000000000, 1.895858229423645, -0.6415748897955779],
                       [1.797239666982623, 0.000000000000000, 1.381637275550612],

                       [-1.797239666982623, 0.000000000000000, 1.381637275550612]])
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0000000000 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_red = (m_C*m_H)/(m_C+m_H)
m_red_D = (m_C*m_D)/(m_C+m_D)
har2wave = 219474.6
ang2bohr = (1.e-10)/(5.291772106712e-11)


struct1 = np.array([[0.000036437, 0.000261250, 0.000020549],
                [0.069682320, 1.105532624, 0.000022669],
                [0.945749639, -0.732895078, -0.000067944],
                [1.182074924, 0.187991209, 0.000009145],
                [-0.436525904, -0.333258609, -0.939243426],
                [-0.436588351, -0.333182565, 0.939259008]])*ang2bohr

struct2 = np.array([[-0.000298856, -0.000017412, -0.000141136],
                [1.022099482, 0.000046092, -0.361081497],
                [0.000552766, 0.471961633, 1.102712489],
                [-0.000044126, -0.471802245, 1.102854612],
                [-0.606229037, -0.878215253, -0.263411220],
                [-0.606365863, 0.878027160, -0.263097351]])*ang2bohr

struct3 = np.array([[-0.000110479, 0.000019746, 0.204885798],
                    [0.000032256, 0.000040871, -0.957945282],
                    [0.950920625, -0.000050787, 0.731398902],
                    [-0.950992179, -0.000020046, 0.731563788],
                    [0.000089457, -1.002878334, -0.339811499],
                    [0.000060320, 1.002888551, -0.339901698]])*ang2bohr

min_en = CH5pot.mycalcpot(np.array([coords_initial_min]*3), 3)*har2wave
a = CH5pot.mycalcpot(np.array([struct1]*3), 3)*har2wave
cs_en = CH5pot.mycalcpot(np.array([coords_initial_cs]*3), 3)*har2wave
b = CH5pot.mycalcpot(np.array([struct2]*3), 3)*har2wave
c2v_en = CH5pot.mycalcpot(np.array([coords_initial_c2v]*3), 3)*har2wave
c = CH5pot.mycalcpot(np.array([struct3]*3), 3)*har2wave



def grid(a, b, N, CH, coords_initial):
    spacing = np.linspace(a, b, num=N)
    if CH == 1:
        new_coords = CoordinateSet(coords_initial, system=CartesianCoordinates3D)
        new_coords = new_coords.convert(ZMatrixCoordinates, ordering=([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]))
        new_coords = new_coords.convert(CartesianCoordinates3D).coords
        print(new_coords[1, 0]/ang2bohr)
        g = np.array([new_coords]*N)
        g[:, 1, 0] = spacing
    elif CH is not 1:
        sub = np.array([coords_initial[1]])
        coords_initial[1] = coords_initial[CH]
        coords_initial[CH] = sub
        new_coords = CoordinateSet(coords_initial, system=CartesianCoordinates3D)
        new_coords = new_coords.convert(ZMatrixCoordinates, ordering=([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]])).convert(CartesianCoordinates3D).coords
        print(new_coords[1, 0]/ang2bohr)
        g = np.array([new_coords] * N)
        g[:, 1, 0] = spacing
    else:
        g = 0
    return g


def Potential(grid):
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


def run(CH, coords, mass, entos=None):
    g = grid(1, 4., 900, CH, coords)
    if entos is not None:
        V = np.diag(np.load('CH5_CH_3_pot.npy')[1] + 40.652825169)
    else:
        V = Potential(g)
    T = Kinetic_Calc(g, mass)
    En, Eig = Energy(T, V)
    print(En[0]*har2wave)
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    print((En[1]-En[0])*har2wave)
    return g[:, 1, 0], Eig[:, 0], np.diag(V)



# print(m_red)
# print(m_red_D)
def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, 5))
    for i in range(5):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch

coords = np.array([[coords_initial_min], [coords_initial_c2v], [coords_initial_cs]]).squeeze()
d = ch_dist(coords)
avg = np.average(d)/ang2bohr
std = np.std(d)/ang2bohr
max = np.max(d)/ang2bohr
min = np.min(d)/ang2bohr
print(avg)
print(std)


entos_wvfn = np.load('CH5_GSW_3.npy')
entos_pot = np.load('CH5_CH_3_pot.npy')
print(entos_wvfn[0, 0]/ang2bohr)
print(entos_wvfn[0, -1]/ang2bohr)
g, eig, V = run(3, coords_initial_min, m_red)
g_ent, eig_ent, V_eng = run(3, coords_initial_min, m_red, "entos")
import matplotlib.pyplot as plt
plt.plot(g/ang2bohr, eig, color='blue', label='Ground State Wave Function')
g, eig, V = run(1, coords_initial_min, m_red)
plt.plot(g/ang2bohr, eig, color='blue')
g, eig, V = run(2, coords_initial_min, m_red)
plt.plot(g/ang2bohr, eig, color='blue')
g, eig, V = run(4, coords_initial_min, m_red)
plt.plot(g/ang2bohr, eig, color='blue')
g, eig, V = run(5, coords_initial_min, m_red)
plt.plot(g/ang2bohr, eig, color='blue')
plt.errorbar(avg, 0.05, yerr=0.007, color='orange', label='Average')
plt.errorbar([avg-std, avg+std], [0.05, 0.05], yerr=[0.005, 0.005], ecolor='green', label='Standard Deviation')
plt.errorbar([min, max], [0.05, 0.05], yerr=[0.003, 0.003], color='red', label='Range')
plt.xlabel(r'r$_{\rm{CH}}$ $\rm\AA$', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('CH5_plot_for_Anne')
plt.show()


entos_norm = np.max(entos_wvfn[1])/np.linalg.norm(eig)
bowman_norm = np.max(eig)/np.linalg.norm(eig)

diff = (entos_wvfn[1]/entos_norm - eig/bowman_norm)
entos_pot[1] -= -40.652825169
diff_pot = entos_pot[1]-V
percent_diff_pot = diff_pot/V*100

print(np.dot(entos_wvfn[1], eig))

# plt.plot(g/ang2bohr, V*har2wave, label='Bowman')
# plt.plot(g/ang2bohr, entos_pot[1]*har2wave, label='Entos')
plt.plot(entos_wvfn[0]/ang2bohr, (entos_wvfn[1])**2, label='MOB-ML')
# plt.plot(g/ang2bohr, diff_pot*har2wave, label='Entos pot - Bowman pot')
plt.plot(g/ang2bohr, (eig)**2, label='Bowman')
# plt.plot(g/ang2bohr, diff, label='Difference')
plt.xlabel(r'r$_{\rmCH}$ $\rm\AA$', fontsize=16)
# plt.ylabel(r'Energy cm$^{-1}$', fontsize=16)
plt.ylabel(r'P(r$_{\rmCH})$', fontsize=16)
plt.tick_params(axis='both', labelsize=18)
# plt.ylabel(r'Entos wave function - Bowman wave function')
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
plt.plot(g/ang2bohr, diff_pot*har2wave)
plt.xlabel(r'r$_{\rmCH}$ $\rm\AA$', fontsize=16)
plt.ylabel('Absolute Difference of \nEntos Energy from \nBowman Energy', fontsize=16)
plt.tick_params(axis='both', labelsize=18)
# plt.ylim(-4, 4)
plt.show()

# dx = (6-0.4)/5000
#
# ind = np.argmin(V)
# second_der = ((1/90*V[ind-3] - 3/20*V[ind-2] + 3/2*V[ind-1] - 49/18*V[ind] + 3/2*V[ind+1] - 3/20*V[ind+2] + 1/90*V[ind+3])/dx**2)
# freq = np.sqrt(1/m_red_D*second_der)
# print(freq*har2wave)

# for i in range(5):
    # print(f'Min {i+1}')
    # g, eig = run(i+1, coords_initial_min, m_red)
    # print(f'Cs {i+1}')
    # g, eig = run(i+1, coords_initial_cs, m_red)
    # print(f'C2v {i+1}')
    # g, eig = run(i+1, coords_initial_c2v, m_red)
    # np.save(f'GSW_c2v_CH_{i+1}', eig)
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
# g = grid(0.4, 6, 5000, 2, coords_initial_min)
# a = np.linspace(0.4, 6, 5000)
# V = np.diag(Potential(g, 2))
# plt.plot(a/ang2bohr, V)
# plt.xlim(0.6, 2.5)
# plt.ylim(0, 7000)
# plt.show()
# np.save('Potential_CH_stretch2', V)

