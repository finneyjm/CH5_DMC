import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)

har2wave = 219474.6
omega = 3600./har2wave


def grid(a, b, N):
    return np.linspace(a, b, num=N)


def Potential_pull(filename, g):
    x = []
    y = []
    with open("%s" %filename, "r") as f:
        for line in f:
            words = line.split()
            x.append(float(words[0]))
            y.append(float(words[1]))

    interp = interpolate.splrep(x, y, s=0)
    V = interpolate.splev(g, interp, der=0)
    return np.diag(V)


def Kinetic_Calc(grid, m_red):
    a = grid[0]
    b = grid[-1]
    N = len(grid)
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


def run(file_name, num, mass):
    g = grid(0.9, 1.1, 500)
    V = Potential_pull(file_name, g)
    T = Kinetic_Calc(g, mass)
    En, Eig = Energy(T, V)
    np.save('%s_GSW' %num, Eig[:, 0])
    return g


mass1 = (m_O*m_H)/(m_O+m_H)
mass2 = ((m_O+m_H)*m_O)/(2.*m_O+m_H)
x = run('mono-roh-pot.dat', 'monomer', mass1)
run('trimer-rohb-pot.dat', 'trimer', mass2)

mono_gsw = np.load('monomer_GSW.npy')
tri_gsw = np.load('trimer_GSW.npy')
fig, axes = plt.subplots()
axes.plot(x, mono_gsw, label='Monomer GSW')
axes.plot(x, mono_gsw**2, label='Monomer GSW sqrd')
axes.plot(x, tri_gsw, label='Trimer GSW')
axes.plot(x, tri_gsw**2, label='Timer GSW sqrd')
axes.plot(x, mono_gsw*tri_gsw, label='Mono*Tri')
axes.set_xlabel('OH stretch (A)')
axes.set_ylabel('Probability Amplitude')
fig.legend()
fig.savefig('mono_tri_gsws.png')
overlap = np.trapz(mono_gsw*mono_gsw, x=x)
overlap2 = np.trapz(mono_gsw*tri_gsw, x=x)
print(overlap2/overlap)
print(np.linalg.norm(mono_gsw*tri_gsw, ord=1))




