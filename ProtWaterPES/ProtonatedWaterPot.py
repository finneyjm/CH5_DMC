import os, sys
import numpy as np


class Potential:

    has_been_loaded = False

    def __init__(self, natm):
        if self.has_been_loaded:
            raise Exception("Can't load this Bowman potential twice")
        self._natm = natm
        self.load_potential()

    def init_potential(self):
        cur_dir = os.getcwd()
        if self._natm == 4:
            try:
                where_u_at = os.path.join(
                    os.path.dirname(__file__),
                    "Hydronium_PES"
                )
                # os.chdir('..')
                os.chdir(where_u_at)
                sys.path.insert(0, where_u_at)

                try:
                    import h3o
                except ImportError:
                    self.compile_potential()
                h3o.init_params()
                self._pot = h3o.pot_calc
            finally:
                os.chdir(cur_dir)
        else:
            try:
                where_u_at = os.path.join(
                    os.path.dirname(__file__),
                    "f2py_prot_water_pot"
                )
                os.chdir('..')
                os.chdir(where_u_at)
                sys.path.insert(0, where_u_at)

                try:
                    import ProtWaterPot
                except ImportError:
                    self.compile_potential()
                ProtWaterPot.init_param(self._natm)
                self._pot = ProtWaterPot.proto_potcalc
            finally:
                os.chdir(cur_dir)

    def get_potential(self, coords):
        """Assumes coords in (nwalkers, natm, 3)

        :param coords:
        :type coords:
        :return:
        :rtype:
        """
        one_wvnum = 5e-6
        ptrimer = -9.129961343400107E-002
        ptetramer = -0.122146858971399
        pdimer = -0.053729340865275525
        if self._natm == 4:
            coords = np.flip(coords, 1)
            pot = self._pot(coords, len(coords), self._natm)
        else:
            pot = self._pot(coords.T, len(coords), self._natm)

        if self._natm == 10:
            death = np.argwhere(pot < (ptrimer-one_wvnum))
        elif self._natm == 13:
            death = np.argwhere(pot < (ptetramer-one_wvnum))
        elif self._natm == 7:
            death = np.argwhere(pot < (pdimer-one_wvnum))
        elif self._natm == 4:
            death = np.argwhere(pot < -one_wvnum)
        else:
            death = []

        pot[death] = 100

        return pot  # do the real thing

    def load_potential(self):
        self.init_potential()
        cls = type(self)
        cls.has_been_loaded = True

    @staticmethod
    def compile_potential():
        os.chdir("..")
        import subprocess
        subprocess.call(["bash", "build.sh"])


