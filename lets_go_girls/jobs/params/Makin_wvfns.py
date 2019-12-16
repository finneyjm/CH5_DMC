import numpy as np

wvfns = np.zeros((5, 5000))
for i in range(5):
    wvfns[i] = np.load(f'min_wvfns/GSW_min_CD_{i+1}.npy')
x = np.linspace(0.4, 6., 5000)

min_average = np.average(wvfns, axis=0)
np.save('min_wvfns/Min_CD_average', np.vstack((x, min_average)))

wvfns_H = np.zeros((5, 5000))
for i in range(5):
    wvfns_H[i] = np.load(f'min_wvfns/GSW_min_CH_{i+1}.npy')
x = np.linspace(0.4, 6., 5000)

min_average_H = np.average(wvfns_H, axis=0)
np.save('min_wvfns/Min_CH_average', np.vstack((x, min_average_H)))

import matplotlib.pyplot as plt
# for i in range(5):
#     plt.plot(x, wvfns[i], label=f'GSW CD {i+1}')
# plt.plot(x, min_average, label='average CD')


for i in range(5):
    plt.plot(x, wvfns_H[i], label=f'GSW CH {i+1}')
plt.plot(x, min_average_H, label='average CH')

plt.xlim(1.5, 3)
plt.legend()
plt.show()
plt.close()












