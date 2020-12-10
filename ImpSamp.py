"""
Goal: The goal of this exercise is to write an implementation of importance sampling diffusion Monte Carlo.
This guide contains three different options for evaluating the potential. It's not neccessary to write them all out
immediately, but always good to lay out your options. Note: It's easiest if this is done in ATOMIC UNITS.

This builds off of the continuous weighting and discrete weighting DMC exercises so please visit those and get
comfortable before tackling this

Fundamentals: Numpy101, Data&I/O
Related Exercises: DiscreteDMC.py, ContinuousDMC.py
"""

# As with any script we will start by using import statements for any necessary packages
import numpy as np


class DMC():

    def __init__(self, dtau, numTimesteps, threshold, numWalkers, initialCoords, masses, potential_func,
                 potential_opts=None, impsampTrialwvfn=None):
        '''
        This part of our class sets up the class with all of our wanted parameters for the simulation

        :param dtau: The time step size
        :type dtau: float
        :param numTimesteps: The number of time steps to run the simulation for
        :type numTimesteps: int
        :param threshold: Threshold that low weight walkers will be removed if their weight gets below this
        :type threshold: float
        :param numWalkers:The number of walkers used for the simulation
        :type numWalkers: float
        :param initialCoords: An array of the initial geometry for the system of interest
        :type initialCoords: np.array or list
        :param masses: The appropriate masses for the system (in atomic units)
        :type masses: list
        :param potential_func: The potential being used for the simulation (for now lets just put these functions within the
                       class itself and deal with loading them separately later)
        :type potential_func: str
        :param potential_opts: Options for the potential energy function (None uses the default options)
        :type potential_opts: dict
        :param impsampTrialwvfn: Options for the potential energy function (None uses the default options)
        :type impsampTrialwvfn: dict
        '''
        ### Set up the initial parameters for the simulation same as we did in the ContinuousDMC excercise
        ### The added option of impsampTrialwvfn allows us to first check if it is None. If that is the case, then we can
        ### procede with DMC as usual
        ### Else, we will use this function to evaluate our trial wave function

