import numpy as np
import matplotlib.pyplot as plt
import DVR_OH as dvr
from scipy import interpolate
import scipy

# DMC parameters
dtau = 1.
N_0 = 10
time_total = 1000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
har2wave = 219474.6


De = 0.02
sigmaOH = np.sqrt(dtau/m_red)
omega = 3600./har2wave

wvfn2 = dvr.run()


plt.figure()
plt.plot(wvfn2[0, :], wvfn2[1, :], wvfn2[0, :], wvfn2[2, :], wvfn2[0, :], wvfn2[3, :])
plt.savefig('Exploring_derivatives_10000.png')


num = 5000
wvfn3 = np.zeros((4, num))
wvfn3[0, :] += np.random.uniform(-1., 1, num)
int = interpolate.splrep(wvfn2[0, :], wvfn2[1, :], s=0)
wvfn3[1, :] += interpolate.splev(wvfn3[0, :], int, der=0)
wvfn3[2, :] += interpolate.splev(wvfn3[0, :], int, der=1)
wvfn3[3, :] += interpolate.splev(wvfn3[0, :], int, der=2)


plt.figure()
plt.scatter(wvfn3[0, :], wvfn3[1, :])
plt.scatter(wvfn3[0, :], wvfn3[2, :])
plt.scatter(wvfn3[0, :], wvfn3[3, :])
plt.savefig('Exploring_derivatives_spline.png')

a_wvfn = np.zeros((4, num))
x = np.linspace(-1, 1, num)
a_wvfn[0, :] += x
mw = m_red*omega
a_wvfn[1, :] += (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*x**2))
a_wvfn[2, :] += (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*x**2))*-mw*x
a_wvfn[3, :] += (mw/np.pi)**(1./4.)*np.exp(-(1./2.*mw*x**2))*(mw**2*x**2-mw)

plt.figure()
plt.plot(a_wvfn[0, :], a_wvfn[1, :], a_wvfn[0, :], a_wvfn[2, :], a_wvfn[0, :], a_wvfn[3, :])
plt.savefig('Exploring_derivatives_analytic.png')

print(np.max(wvfn2[1, :]))
print(np.max(a_wvfn[1, :]))
print(np.max(a_wvfn[1, :])/np.max(wvfn2[1, :]))

