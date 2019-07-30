import numpy as np
import matplotlib.pyplot as plt

grid = np.linspace(-0.2, 0.2, num=10000)
fig, axes = plt.subplots()
for i in range(11):
    alpha = float(20*i + 2)*0.5
    switch = (np.tanh(alpha*grid) + 1.)*0.5
    axes.plot(grid, switch, label=r'$\alpha$ = %s' %alpha)
axes.legend()
fig.savefig('Switching_functions_vary_alpha.png')
