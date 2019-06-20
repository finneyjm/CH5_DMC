import numpy as np
import matplotlib.pyplot as plt

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6


De = 0.1896
omega = 3600./har2wave
omegah = 3600./har2wave
alph = m_red*omega
A = np.sqrt(omega**2 * m_red/(2*De))
lam = np.sqrt(2. * m_red * De)/A
asdf = (1./2. - 1./(8.*lam))*omega

print(asdf*har2wave)


g = np.linspace(-4, 4, num=1000)
morse = De*(1. - np.exp(-A*g))**2
HO = 1./2.*m_red*omegah**2*g**2
plt.figure()
plt.plot(g, morse*har2wave, label='morse')
plt.plot(g, HO*har2wave, label='HO')
plt.ylim((0, 0.1896*har2wave))
plt.legend()
plt.savefig('Potentials.png')

min = np.load('Average_GSW_CH_stretch.npy')
cs = np.load('Average_GSW_CH_stretch_cs_saddle.npy')
c2v = np.load('Average_GSW_CH_stretch_c2v_saddle.npy')

avg = np.mean([-min[1, :], -cs[1, :], -c2v[1, :]], axis=0)
plt.figure()
plt.plot(min[0, :], avg, label='Average')
plt.plot(min[0, :], -min[1, :], label='Minimum')
plt.plot(cs[0, :], -cs[1, :], label='Cs Saddle')
plt.plot(c2v[0, :], -c2v[1, :], label='C2v, Saddle')
plt.legend()
plt.savefig('Compare_Avg_wvfns.png')

np.save('Total_Average_GSW_CH_stretch', [min[0, :], avg])
