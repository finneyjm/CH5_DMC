import numpy as np
import matplotlib.pyplot as plt

ang2bohr = 1.e-10/5.291772106712e-11
grid = np.linspace(1., 4., 500)/ang2bohr

GSW = np.zeros((3, 5, 500))
for i in range(5):
    GSW[0, i, :] += np.load(f'GSW_min_CH_{i+1}.npy')
    GSW[1, i, :] += np.load(f'GSW_cs_CH_{i+1}.npy')
    GSW[2, i, :] += np.load(f'GSW_c2v_CH_{i+1}.npy')


def lets_get_these_wvfns(type, switch_speed):
    if type == 'min':
        arg1 = np.argmax(GSW[0, 1, :])
        arg2 = np.argmax(GSW[0, 4, :])
        r1 = grid[arg1]
        r2 = grid[arg2]
        sp = np.average([r1, r2])
        switch = (np.tanh(switch_speed*(grid-sp)) + 1.)*0.5
        new_wvfn = GSW[0, 1, :]*switch + GSW[0, 4, :]*(1.-switch)
        new_wvfn = new_wvfn/np.linalg.norm(new_wvfn)

        plt.plot(grid, new_wvfn, label='Switch wvfn')
        plt.plot(grid, np.load('Average_GSW_CH_stretch_min.npy')[1, :], label='Average GSW')
        plt.plot(grid, GSW[0, 1, :], label='CH 2')
        plt.plot(grid, GSW[0, 4, :], label='CH 5')
        plt.legend()
        plt.savefig(f'Switch_min_wvfn_speed_{switch_speed}.png')
        plt.close()

    elif type == 'cs':
        arg1 = np.argmax(GSW[1, 0, :])
        arg2 = np.argmax(GSW[1, 2, :])
        r1 = grid[arg1]
        r2 = grid[arg2]
        sp = np.average([r1, r2])
        switch = (np.tanh(switch_speed*(grid-sp)) + 1.)*0.5
        new_wvfn = GSW[1, 2, :]*switch + GSW[1, 0, :]*(1.-switch)
        new_wvfn = new_wvfn/np.linalg.norm(new_wvfn)

        plt.plot(grid, new_wvfn, label='Switch wvfn')
        plt.plot(grid, np.load('Average_GSW_CH_stretch_cs_saddle.npy')[1, :], label='Average GSW')
        plt.plot(grid, GSW[1, 0, :], label='CH 1')
        plt.plot(grid, GSW[1, 2, :], label='CH 3')
        plt.legend()
        plt.savefig(f'Switch_cs_wvfn_speed_{switch_speed}.png')
        plt.close()

    elif type == 'c2v':
        arg1 = np.argmax(GSW[2, 0, :])
        arg2 = np.argmax(GSW[2, 2, :])
        r1 = grid[arg1]
        r2 = grid[arg2]
        sp = np.average([r1, r2])
        switch = (np.tanh(switch_speed*(grid-sp)) + 1.)*0.5
        new_wvfn = GSW[2, 0, :]*switch + GSW[2, 2, :]*(1.-switch)
        new_wvfn = new_wvfn/np.linalg.norm(new_wvfn)

        plt.plot(grid, new_wvfn, label='Switch wvfn')
        plt.plot(grid, np.load('Average_GSW_CH_stretch_c2v_saddle.npy')[1, :], label='Average GSW')
        plt.plot(grid, GSW[2, 0, :], label='CH 1')
        plt.plot(grid, GSW[2, 2, :], label='CH 3')
        plt.legend()
        plt.savefig(f'Switch_c2v_wvfn_speed_{switch_speed}.png')
        plt.close()

    else:
        print("That's not how CH5 works!")
        new_wvfn = np.zeros(grid.shape())

    new_wvfn = np.vstack((grid*ang2bohr, new_wvfn))
    np.save(f'Switch_{type}_wvfn_speed_{switch_speed}', new_wvfn)

    return


for i in range(11):
    lets_get_these_wvfns('min', float(20*i + 2)*0.5)
    lets_get_these_wvfns('cs', float(20*i + 2)*0.5)
    lets_get_these_wvfns('c2v', float(20*i + 2)*0.5)














