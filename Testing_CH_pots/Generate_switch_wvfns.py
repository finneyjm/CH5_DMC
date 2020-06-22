import numpy as np
import matplotlib.pyplot as plt

ang2bohr = 1.e-10/5.291772106712e-11
grid = np.linspace(0.4, 6., 5000)/ang2bohr

GSW = np.zeros((3, 5, 5000))
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
        # switch = (np.tanh(switch_speed*(grid-sp)) + 1.)*0.5
        # new_wvfn = GSW[0, 1, :]*switch + GSW[0, 4, :]*(1.-switch)
        # new_wvfn = new_wvfn/np.linalg.norm(new_wvfn)

        # plt.plot(grid, new_wvfn, label='Switch Wavefunction')
        # plt.plot(grid, np.load('Average_GSW_CH_stretch_min.npy')[1, :], label='Average Ground State Wavefunction')
        # plt.plot(grid, np.mean(GSW[0], axis=0)/np.linalg.norm(np.mean(GSW[0], axis=0)), linestyle='dotted', linewidth=3, color='indigo', label='Average')
        # plt.plot(grid, GSW[0, 1, :], color='orange', linewidth=3, label=r'CH$^{(2)}$')
        # plt.plot(grid, GSW[0, 4, :], color='red', linestyle='--', linewidth=3, label=r'CH$^{(5)}$')
        plt.plot(grid, GSW[0, 0, :], color='purple', linestyle='dashdot', linewidth=3, label=r'CH$^{(1)}$')
        plt.plot(grid, GSW[0, 1, :], color='orange', linewidth=3, label=r'CH$^{(2)}$')
        plt.plot(grid, GSW[0, 2, :], color='blue', linestyle='--', linewidth=3, label=r'CH$^{(3)}$')
        plt.plot(grid, GSW[0, 3, :], color='green', linewidth=3, label=r'CH$^{(4)}$')
        plt.plot(grid, GSW[0, 4, :], color='red', linestyle='--', linewidth=3, label=r'CH$^{(5)}$')
        # plt.plot(grid, GSW[0, 4, :], label='Higher Frequency CH stretch')
        plt.xlabel(r'r$_{\rm{CH}}$ ($\rm\AA$)', fontsize=22)
        plt.ylabel(r'$\rm\psi$(r$_{\rm{CH}}$)', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=16)
        leg = plt.legend(loc='upper right', fontsize=14)
        leg.get_frame().set_edgecolor('white')
        plt.tight_layout()
        plt.xlim(0.6, 2.0)
        plt.ylim(0, 0.058)
        plt.show()
        # plt.savefig(f'Switch_min_wvfn_speed_powerpoint_{switch_speed}.png')
        plt.close()

    elif type == 'cs':
        arg1 = np.argmax(GSW[1, 1, :])
        arg2 = np.argmax(GSW[1, 4, :])
        r1 = grid[arg1]
        r2 = grid[arg2]
        sp = np.average([r1, r2])
        # switch = (np.tanh(switch_speed*(grid-sp)) + 1.)*0.5
        # new_wvfn = GSW[0, 1, :]*switch + GSW[0, 4, :]*(1.-switch)
        # new_wvfn = new_wvfn/np.linalg.norm(new_wvfn)

        # plt.plot(grid, new_wvfn, label='Switch Wavefunction')
        # plt.plot(grid, np.load('Average_GSW_CH_stretch_min.npy')[1, :], linewidth=3, label='Average Ground State Wavefunction')
        # plt.plot(grid, np.mean(GSW[1], axis=0)/np.linalg.norm(np.mean(GSW[1], axis=0)), linewidth=3, color='orange', label='Average')
        # plt.plot(grid, GSW[1, 2, :], color='green', label=r'CH$_3$')
        # plt.plot(grid, GSW[1, 4, :], color='red', label=r'CH$_5$')
        plt.plot(grid, GSW[1, 0, :], color='purple', linestyle='dashdot', linewidth=3, label=r'CH$^{(1)}$')
        plt.plot(grid, GSW[1, 1, :], color='orange', linewidth=3, label=r'CH$^{(2)}$')
        plt.plot(grid, GSW[1, 2, :], color='blue', linestyle='--', linewidth=3, label=r'CH$^{(3)}$')
        plt.plot(grid, GSW[1, 3, :], color='green', linewidth=3, label=r'CH$^{(4)}$')
        plt.plot(grid, GSW[1, 4, :], color='red', linestyle='--', linewidth=3, label=r'CH$^{(5)}$')
        plt.xlabel(r'r$_{\rm{CH}}$ ($\rm\AA$)', fontsize=22)
        plt.ylabel(r'$\rm\psi$(r$_{\rm{CH}}$)', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=16)
        leg = plt.legend(loc='upper right', fontsize=14)
        leg.get_frame().set_edgecolor('white')
        plt.tight_layout()
        plt.xlim(0.6, 2.0)
        plt.ylim(0, 0.058)
        plt.show()
        # plt.savefig(f'Switch_min_wvfn_speed_powerpoint_{switch_speed}.png')
        plt.close()

    elif type == 'c2v':
        arg1 = np.argmax(GSW[1, 1, :])
        arg2 = np.argmax(GSW[1, 4, :])
        r1 = grid[arg1]
        r2 = grid[arg2]
        sp = np.average([r1, r2])
        # switch = (np.tanh(switch_speed*(grid-sp)) + 1.)*0.5
        # new_wvfn = GSW[0, 1, :]*switch + GSW[0, 4, :]*(1.-switch)
        # new_wvfn = new_wvfn/np.linalg.norm(new_wvfn)

        # plt.plot(grid, new_wvfn, label='Switch Wavefunction')
        # plt.plot(grid, np.load('Average_GSW_CH_stretch_min.npy')[1, :], label='Average Ground State Wavefunction')
        # plt.plot(grid, np.mean(GSW[1], axis=0)/np.linalg.norm(np.mean(GSW[1], axis=0)), color='orange', label='Average')
        # plt.plot(grid, GSW[1, 2, :], color='green', label=r'CH$_3$')
        # plt.plot(grid, GSW[1, 4, :], color='red', label=r'CH$_5$')
        plt.plot(grid, GSW[2, 0, :], color='purple', linestyle='dashdot', linewidth=3, label=r'CH$^{(1)}$')
        plt.plot(grid, GSW[2, 1, :], color='orange', linewidth=3, label=r'CH$^{(2)}$')
        plt.plot(grid, GSW[2, 2, :], color='blue', linestyle='--', linewidth=3, label=r'CH$^{(3)}$')
        plt.plot(grid, GSW[2, 3, :], color='green', linewidth=3, label=r'CH$^{(4)}$')
        plt.plot(grid, GSW[2, 4, :], color='red', linestyle='--', linewidth=3, label=r'CH$^{(5)}$')
        # plt.plot(grid, GSW[0, 4, :], label='Higher Frequency CH stretch')
        plt.xlabel(r'r$_{\rm{CH}}$ ($\rm\AA$)', fontsize=22)
        plt.ylabel(r'$\rm\psi$(r$_{\rm{CH}}$)', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=16)
        leg = plt.legend(loc='upper right', fontsize=14)
        leg.get_frame().set_edgecolor('white')
        plt.tight_layout()
        plt.xlim(0.6, 2.0)
        plt.ylim(0, 0.058)
        plt.show()
        # plt.savefig(f'Switch_min_wvfn_speed_powerpoint_{switch_speed}.png')
        plt.close()

    else:
        print("That's not how CH5 works!")
        new_wvfn = np.zeros(grid.shape())

    # new_wvfn = np.vstack((grid*ang2bohr, new_wvfn))
    # np.save(f'Switch_{type}_wvfn_speed_{switch_speed}', new_wvfn)

    return


for i in range(1):
    lets_get_these_wvfns('min', 'nah')
    lets_get_these_wvfns('cs', 'nah')
    lets_get_these_wvfns('c2v', 'nah')

# lets_get_these_wvfns('min', 5.)












