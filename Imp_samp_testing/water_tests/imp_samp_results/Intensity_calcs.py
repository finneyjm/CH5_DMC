import numpy as np
import multiprocessing as mp
from itertools import repeat
ang2bohr = 1.e-10/5.291772106712e-11
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_OD = (m_D*m_O)/(m_D+m_O)
m_OH = (m_H*m_O)/(m_H+m_O)
har2wave = 219474.6
omega_OD = 2832.531899782715
omega_OH = 3890.7865072878913
mw_d = m_OD * omega_OD/har2wave
mw_h = m_OH * omega_OH/har2wave


class waveFunction:

    def __init__(self):
        self.wvfn_loaded = False

    def psi_t_lin_combo(self, coords, excite, shift, atoms):
        dists = self.oh_dists(coords)
        r1 = 0.9616036495623883 * ang2bohr
        r2 = 0.9616119936423067 * ang2bohr
        req = [r1, r2]
        dists = dists - req
        if atoms[1].upper() == 'H':
            if atoms[2].upper() == 'H':
                mw1 = mw_h
                mw2 = mw_h
            else:
                mw1 = mw_h
                mw2 = mw_d
        else:
            if atoms[2].upper() == 'H':
                mw1 = mw_d
                mw2 = mw_h
            else:
                mw1 = mw_d
                mw2 = mw_d
        dists = dists - shift[:2]
        if excite == 'asym' or excite == 'sym':
            psi = np.zeros((len(coords), 2))
            psi[:, 0] = angle_function(coords, excite, shift, atoms)
            term1 = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2)) * \
                    (2 * mw1) ** (1 / 2) * dists[:, 0]
            term1_2 = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
            term2 = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2))
            term2_2 = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2)) * \
                     (2 * mw2) ** (1 / 2) * dists[:, 1]
            if excite == 'asym':
                psi[:, 1] = 1/np.sqrt(2)*(term1*term1_2 - term2*term2_2)
            else:
                psi[:, 1] = 1/np.sqrt(2)*(term1*term1_2 + term2*term2_2)
        else:
            psi = np.zeros((len(coords), 3))
            psi[:, 0] = angle_function(coords, excite, shift, atoms)
            psi[:, 1] = (mw1 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw1 * dists[:, 0] ** 2))
            psi[:, 2] = (mw2 / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw2 * dists[:, 1] ** 2))
        return np.prod(psi, axis=-1)

    def psi_t_2d(self, coords, excite, filename=None):
        dists = self.oh_dists(coords)
        anti = 1 / np.sqrt(2) * (dists[:, 1] - dists[:, 0])
        sym = 1 / np.sqrt(2) * (dists[:, 1] + dists[:, 0])
        if self.wvfn_loaded is not True:
            self.load_dvr_wvfn(filename)
        psi = self.interp(anti, sym, self.psi_wvfn)
        return psi

    @staticmethod
    def interp(x, y, poiuy):
        out = np.zeros(len(x))
        for i in range(len(x)):
            out[i] = poiuy(x[i], y[i])
        return out

    def load_dvr_wvfn(self, filename):
        wvfns = np.load(f'{filename}.npz')
        ground = wvfns['no_der']
        gridz = wvfns['grid']
        from scipy import interpolate
        self.psi_wvfn = interpolate.interp2d(gridz[0], gridz[1], ground.T, kind='cubic')
        self.wvfn_loaded = True

    @staticmethod
    def oh_dists(coords):
        bonds = [[1, 2], [1, 3]]
        cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
        cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
        dis = np.linalg.norm(cd2 - cd1, axis=2)
        return dis

    @staticmethod
    def angle_function(coords, excite, shift, atoms):
        angs = self.angle(coords)
        angs = angs - shift[2]
        r1 = 0.9616036495623883 * ang2bohr
        r2 = 0.9616119936423067 * ang2bohr
        theta = np.deg2rad(104.1747712)
        muH = 1 / m_H
        muO = 1 / m_O
        muD = 1 / m_D
        if atoms[1].upper() == 'H':
            if atoms[2].upper() == 'H':
                G = gmat(muH, muH, muO, r1, r2, theta)
                freq = 1668.4590610594878
            else:
                G = gmat(muH, muD, muO, r1, r2, theta)
                freq = 1462.5810039828614
        else:
            if atoms[2].upper() == 'H':
                G = gmat(muD, muH, muO, r1, r2, theta)
                freq = 1462.5810039828614
            else:
                G = gmat(muD, muD, muO, r1, r2, theta)
                freq = 1222.5100195873742
        freq /= har2wave
        alpha = freq / G
        if excite == 'ang':
            return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2) * (2 * alpha) ** (1 / 2) * (
                        angs - theta)
        else:
            return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2)

    @staticmethod
    def angle(coords):
        dists = oh_dists(coords)
        v1 = (coords[:, 1] - coords[:, 0]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
        v2 = (coords[:, 2] - coords[:, 0]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))

        ang1 = np.arccos(np.matmul(v1[:, None, :], v2[..., None]).squeeze())

        return ang1.T


class DMCIntensities:

    def __init__(self, filename, dipole_func, wvfn, ref_struct, mass, walkers, num_wvfns, num_sims,
                 num_time_steps, symmetry, planar, excited_state_wvfn, append, filename1=None, filename2=None, wvfn2=None):
        '''
        Initialize the DMC intensities class with all of the relevant parameters
        :param filename: The filename of the relevant dmc simulations
        :type filename: str
        :param dipole_func: A function to calculate the dipole given a set of coordinates
        :type dipole_func: function
        :param wvfn: A function to calculate the value of the trial wave function
        :type wvfn: function
        :param ref_struct: The reference structure of the system to perform an Eckart rotation
        :type ref_struct: np array
        :param mass: A list of masses of the atoms in the system
        :type mass: list/np array
        :param walkers: The number of walkers in each simulation
        :type walkers: int
        :param num_wvfns: The number of wave functions that are collected in each simulation
        :type num_wvfns: int
        :param num_sims: The number of dmc simulations with the given filename structure to collect info from
        :type num_sims: int
        :param num_time_steps: The number of time steps that the simulations were run with
        :type num_time_steps: int
        :param symmetry: A function to symmeterize the coordiates of the walkers
        :type symmetry: function
        :param planar: Is the reference structure of the system planar or not?
        :type planar: True/False
        :param excited_state_wvfn: Is this an excited state DMC simulation or not?
        :type excited_state_wvfn: True/False
        :param append: Something to append to the saved files to make them unique
        :type append: str
        :param filename1: Filename for one half of an excited state wave function
        :type filename1: str
        :param filename2: Filename for one half of an excited state wave function
        :type filename2: str
        :param wvfn2: A function to calculate the value of the trial wave function for an excited state
        :type wvfn2: function
        '''
        self.filename = filename
        self.dipole = dipole_func
        self.psit = wvfn
        if wvfn2 is None:
            self.psit2 = self.psit
        else:
            self.psit2 = wvfn2
        self.ref = ref_struct
        self.mass = mass
        self.walkers = walkers
        self.num_wvfns = num_wvfns
        self.num_sims = num_sims
        self.num_time_steps = num_time_steps
        self.symmetry = symmetry
        self.planar = planar
        self.excite_wvfn = excited_state_wvfn
        self.filename2 = filename2
        self.filename1 = filename1
        self.append = append

    def eckart_dips(self):
        '''
        A function to Eckart rotate, symmeterize, and calculate the dipoles of a set of coordinates from some dmc
        simulations. It also calculated the average and standard deviation of the energy of those simulations.
        :return: Saves relevant information for future calculations
        :rtype: npz file
        '''

        # import sys
        # path = sys.path.insert(0, '../../')
        from Eckart_turny_turn import EckartsSpinz
        from PAF_spinz import MomentOfSpinz

        MOM = MomentOfSpinz(self.ref, self.mass)
        ref = MOM.coord_spinz()

        if self.excite_wvfn is False:
            coords = np.zeros((self.num_sims, self.walkers, self.num_wvfns, len(self.mass), 3))
            erefs = np.zeros((self.num_sims, self.num_time_steps))
            weights = np.zeros((self.num_sims, self.num_wvfns, self.walkers))
            for i in range(self.num_sims):
                blah = np.load(f'{self.filename}_{i+1}.npz')
                coords[i] = blah['coords']
                erefs[i] = blah['Eref']
                weights[i] = blah['weights']

            avg_eref = np.mean(np.mean(erefs[:, 5000:], axis=1))
            std_eref = np.std(np.mean(erefs[:, 5000:], axis=1))
            coords = coords.reshape((self.num_sims, self.walkers*self.num_wvfns, len(self.mass), 3))
            weights = weights.reshape((self.num_sims, self.walkers*self.num_wvfns))
            coords, weights = self.symmetry(coords, weights)
            dips = np.zeros((self.num_sims, coords.shape[1], 3))
            for i in range(self.num_sims):
                eck = EckartsSpinz(ref, coords[i], self.mass, self.planar)
                coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
                dips[i] = self.dipole(coords[i])


        else:
            coords_left = np.zeros((self.num_sims, self.walkers, self.num_wvfns, len(self.mass), 3))
            coords_right = np.zeros((self.num_sims, self.walkers, self.num_wvfns, len(self.mass), 3))
            erefs_left = np.zeros((self.num_sims, self.num_time_steps))
            erefs_right = np.zeros((self.num_sims, self.num_time_steps))
            weights_left = np.zeros((self.num_sims, self.num_wvfns, self.walkers))
            weights_right = np.zeros((self.num_sims, self.num_wvfns, self.walkers))
            for i in range(self.num_sims):
                blah = np.load(f'{self.filename1}_{i+1}.npz')
                coords_left[i] = blah['coords']
                erefs_left[i] = blah['Eref']
                weights_left[i] = blah['weights']

            for i in range(self.num_sims):
                blah = np.load(f'{self.filename2}_{i+1}.npz')
                coords_right[i] = blah['coords']
                erefs_right[i] = blah['Eref']
                weights_right[i] = blah['weights']

            avg_eref_left = np.mean(np.mean(erefs_left[:, 5000:], axis=1))
            std_eref_left = np.std(np.mean(erefs_left[:, 5000:], axis=1))

            avg_eref_right = np.mean(np.mean(erefs_right[:, 5000:], axis=1))
            std_eref_right = np.std(np.mean(erefs_right[:, 5000:], axis=1))

            avg_eref = np.mean([avg_eref_left, avg_eref_right])
            std_eref = np.linalg.norm([std_eref_left, std_eref_right])/2

            coords = np.hstack((coords_left, coords_right))
            weights = np.hstack((weights_left, weights_right))
            coords = coords.reshape((self.num_sims, self.walkers*self.num_wvfns*2, len(self.mass), 3))
            weights = weights.reshape((self.num_sims, self.walkers*self.num_wvfns*2))
            coords, weights = self.symmetry(coords, weights)
            dips = np.zeros((self.num_sims, coords.shape[1], 3))
            for i in range(self.num_sims):
                eck = EckartsSpinz(ref, coords[i], self.mass, self.planar)
                coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
                dips[i] = self.dipole(coords[i])

        np.savez(f'{self.filename}_eck_dips', coords=coords, weights=weights,
                 dips=dips, a_eref=avg_eref, s_eref=std_eref)

        self.coords = coords
        self.weights = weights
        self.dips = dips
        self.a_eref = avg_eref
        self.s_eref = std_eref

    def frac_calc(self, excite, **wvfn_kwargs):
        '''
        A function to calculate and save the appropiate ratios of excited state to ground state trial wave function
        values for the dmc coordinates
        :param excite: A string that indicates to the wave function calculation, what type of excitation
        is required from the wave function evaluation
        :type excite: str
        :param wvfn_kwargs: Any extra key word arguments for the wave function evaluation
        :type wvfn_kwargs: dict
        :return: saves the fraction of psi1/psi0 for a ground state calculation and psi0/psi1 for an excited
        state calculation
        :rtype: npz file/np array
        '''
        frac = np.zeros((self.coords.shape[0], self.coords.shape[1]))
        for i in range(self.num_sims):
            h0 = self.psit(self.coords[i], excite=None, **wvfn_kwargs)
            h1 = self.psit2(self.coords[i], excite, **wvfn_kwargs)
            if self.excite_wvfn is None:
                frac[i] = h1/h0
            else:
                frac[i] = h0/h1

        np.savez(f'{self.filename}_{self.append}_wvfn_fractions', frac=frac)

        self.frac = frac

    def intensitiessss(self, rel_dis=None):
        '''
        A function that calculates the relevant quantities to obtain intensities of the specific transitions of
        interest. It also calculates overlaps, matrix elements of the dipole, and <0|r|1> where r is a function for a
        relevant distance of interest.
        :param rel_dis: A function to calculate a relevant distance, mainly for invetigative purposes
        :type rel_dis: function
        :return: saves the relevant quantities to an npz file
        :rtype: npz file
        '''
        au_to_Debye = 1 / 0.3934303
        conv_fac = 4.702e-7
        km_mol = 5.33e6
        conversion = conv_fac*km_mol

        term1 = np.zeros((self.num_sims, 3))
        term1_ov = np.zeros(self.num_sims)
        if rel_dis is not None:
            term1_rel_dis = np.zeros(self.num_sims)
        else:
            term1_rel_dis = [7, 11]
        for i in range(self.num_sims):
            blah = np.where(np.isfinite(self.frac[i]))
            term1_ov[i] = np.dot(self.weights[i, blah].squeeze(), self.frac[i, blah].squeeze()) \
                          / np.sum(self.weights[i, blah].squeeze())
            if rel_dis is not None:
                dis = rel_dis(self.coords[i, blah].squeeze())
                term1_rel_dis[i] = np.dot(self.weights[i, blah].squeeze(), self.frac[i, blah].squeeze()*dis)\
                                   / np.sum(self.weights[i, blah].squeeze())
            for j in range(3):
                term1[i, j] = np.dot(self.weights[i, blah].squeeze(), self.frac[i, blah].squeeze()*
                                     self.dips[i, blah, j].squeeze()*au_to_Debye)/\
                              np.sum(self.weights[i, blah].squeeze())
        avg_term1_o = np.average(term1_ov)
        std_term1_o = np.std(term1_ov)
        avg_term1_dis = np.average(term1_rel_dis)
        std_term1_dis = np.std(term1_rel_dis)
        avg_dipole_components = np.average(term1, axis=0)
        std_dipole_components = np.std(term1, axis=0)
        t_dipoles = np.linalg.norm(term1, axis=1)
        avg_t_dips = np.average(t_dipoles)
        std_t_dips = np.std(t_dipoles)

        std_term1_sq = avg_t_dips**2 * np.sqrt(2*(std_t_dips/avg_t_dips)**2)
        std_int_wo_freq = conversion * std_term1_sq
        intensity_wo_freq = conversion * avg_t_dips ** 2

        np.savez(f'{self.filename}_{self.append}_intensity_info', avg_o=avg_term1_o, std_o=std_term1_o, avg_d=avg_term1_dis,
                 std_d=std_term1_dis, avg_dip_comp=avg_dipole_components, std_dip_comp=std_dipole_components,
                 avg_t_mom=avg_t_dips, std_t_mom=std_t_dips, intense=intensity_wo_freq, std_intense=std_int_wo_freq)

        self.std_int = std_int_wo_freq
        self.intens = intensity_wo_freq

    def lets_do_some_calcs(self, rel_dis, excite, **wvfn_kwargs):
        '''
        A list of function calls to generate the neccessary values and npz files.
        :param rel_dis: A function to calculate a relevant distance, mainly for invetigative purposes
        :type rel_dis: function
        :param excite: The key word argument for the trial wave function evaluation for the excited state of interest
        :type excite: str
        :param wvfn_kwargs: Extra key word arguments for the trial wave function evaluation
        :type wvfn_kwargs: dict
        :return: average and standard deviation of intensity (without multiplying by frequency)
        :rtype: list of floats
        '''
        try:
            blah = np.load(f'{self.filename}_eck_dips.npz')
            print('loading in coordinates and dipoles')
            self.coords = blah['coords']
            self.weights = blah['weights']
            self.dips = blah['dips']
            self.a_eref = blah['avg_eref']
            self.s_eref = blah['std_eref']
        except:
            print('starting dipole calcs from scratch')
            self.eckart_dips()

        try:
            blah = np.load(f'{self.filename}_{self.append}_wvfn_fractions.npz')
            print('loading in wvfn fractions')
            self.frac = blah['frac']
        except:
            print('starting wvfn fraction calculations')
            self.frac_calc(excite, **wvfn_kwargs)

        self.intensitiessss(rel_dis)
        return self.intens, self.std_int


    @staticmethod
    def freqs_from_erefs(ground_eref, std_ground_eref, excite_eref, std_excite_eref):
        '''
        A function to determine the average and standard deviation of a frequency of interest from the outputs of a set
        of ground and excited state simulations
        :param ground_eref: Average value of the zero-point energy
        :type ground_eref: Float
        :param std_ground_eref: Standard deviation of the zero-point energy
        :type std_ground_eref: Float
        :param excite_eref: Average value of the excited state energy
        :type excite_eref: Float
        :param std_excite_eref: Standard deviation of the excited state energy
        :type std_excite_eref: Float
        :return: Average and standard deviation of the frequency
        :rtype: List of floats
        '''
        freq = excite_eref - ground_eref
        std_freq = np.sqrt(std_ground_eref**2 + std_excite_eref**2)
        return freq, std_freq

    def intensites_w_freq(self, ground_eref, std_ground_eref, excite_eref, std_excite_eref):
        '''
        A function to calculate the intensity in km/mol from the frequency of a transition of interest
        :param ground_eref: Average value of the zero-point energy
        :type ground_eref: Float
        :param std_ground_eref: Standard deviation of the zero-point energy
        :type std_ground_eref: Float
        :param excite_eref: Average value of the excited state energy
        :type excite_eref: Float
        :param std_excite_eref: Standard deviation of the excited state energy
        :type std_excite_eref: Float
        :return: average and standard deviation of intensity in km/mol
        :rtype: list of floats
        '''
        freq, std_freq = self.freqs_from_erefs(ground_eref, std_ground_eref, excite_eref, std_excite_eref)
        self.std_int = self.intens*freq*np.sqrt((self.std_int/self.intens)**2 + (std_freq/freq)**2)
        self.intens = self.intens*freq
        print(f'intensity = {self.intens} +/- {self.std_int}')
        return self.intens, self.std_int
