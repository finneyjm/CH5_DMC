import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

har2wave = 219474.6

freqCH = 2446.8024459517733
freqCD = 1796.51587990911
massCH = 1694.8142311854665
massCD = 3143.820873920939

ch_stretch = np.load('../params/min_wvfns/GSW_min_CH_2.npy')
x = np.linspace(0.4, 6., 5000)
shift = x[np.argmax(ch_stretch)]
# x -= shift
interp = interpolate.splrep(x, ch_stretch, s=0)


CH = np.array((freqCH, massCH))
CD = np.array((freqCD, massCD))


def rch_harm_osc(x, shift, interp):
    freq, atom = interp

    m = 1/atom

    freq /= har2wave
    alpha = freq / m
    return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (x - shift) ** 2 / 2)


plt.plot(x, ch_stretch, label='DVR')

plt.plot(x, rch_harm_osc(x, np.zeros(5000) + shift, CH)/30, label='harmonic')
y = rch_harm_osc(x, np.zeros(5000)+shift, CH)/30
np.save('../params/min_wvfns/GSW_min_CH_2_harm', y)
plt.legend()
plt.xlim(1.2, 3.2)
plt.show()

# np.save('../params/min_wvfns/rch_params_GSW2', CH)
# np.save('../params/min_wvfns/rcd_params_GSW2', CD)