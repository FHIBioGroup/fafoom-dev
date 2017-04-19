#   Copyright 2015 Adriana Supady
#
#   This file is part of fafoom.
#
#   Fafoom is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   Fafoom is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with fafoom.  If not, see <http://www.gnu.org/licenses/>.

'''Wrapper for ORCA'''
from __future__ import division
import glob
import os
import subprocess

from utilities import sdf2xyz

hartree2eV = 27.21138602


class OrcaObject():
    '''Create and handle ORCA objects.'''
    def __init__(self, commandline, memory, chargemult="0 1", nprocs=1,
                 optsteps=500):
        """Initialize the OrcaObject.

        Args(required):
            commandline
            memory
        Args(optional):
            chargemult (default="0 1")
            nprocs (default=1)
            optsteps (default=500)
        Raises:
            KeyError: if the commandline or memory is not defined
        """
        self.commandline = commandline
        self.memory = memory
        self.chargemult = chargemult
        self.nprocs = nprocs
        self.optsteps = optsteps

    def generate_input(self, sdf_string):
        """Create input files for ORCA.
        Args:
            sdf_string (str)
        """
        xyz_string = sdf2xyz(sdf_string)
        coord = xyz_string.split('\n')
        string1 = '! '+str(self.commandline)+'\n'
        string2 = '! xyzfile\n'
        string3 = '%geom MaxIter '+str(self.optsteps)+' end\n'
        string4 = '%pal nprocs '+str(self.nprocs)+" end\n"
        string5 = '%maxcore '+str(self.memory)+"\n"
        string6 = '* xyz '+str(self.chargemult)+'\n'
        with open('orca_molecule.inp', 'w') as f:
            f.write(string1)
            f.write(string2)
            f.write(string3)
            f.write(string4)
            f.write(string5)
            f.write(string6)
            f.write('\n'.join(coord[2:]))
            f.write('*\n')
        f.close()

    def run_orca(self, execution_string):
        """Run orca and write output to 'result.out'. The optimized
        geometry is written to 'orca_molecule.xyz'.

        Warning: this function uses subprocessing to invoke the run.
        The subprocess's shell is set to TRUE.
        Args:
            execution_string (str): e.g. ORCA or for parallel version
            /the/complete/path/to/orca
        Raises:
            OSError: if orca_molecule.inp not present in the working directory
        """
        success = False
        if os.path.exists('orca_molecule.inp') is False:
            raise OSError("Required input file not present.")
        orca = subprocess.Popen(
            execution_string+str(" orca_molecule.inp"),
            stdout=subprocess.PIPE, shell=True)
        out = subprocess.Popen(
            ['cat'], stdin=orca.stdout,
            stdout=open('result.out', 'w'), shell=True)
        out.wait()

        searchfile = open("result.out", "r")

        s0 = "THE OPTIMIZATION HAS CONVERGED"
        s = "FINAL SINGLE POINT ENERGY"
        not_conv = True
        for line in searchfile:
            if s0 in line:
                not_conv = False
        searchfile.close()
        if not_conv:
            killfile = open("kill.dat", "w")
            killfile.close()
        else:
            searchfile = open("result.out", "r")
            for line in searchfile:
                if s in line:
                    energy_tmp = float(line.split(" ")[-1].split('\n')[0])
            searchfile.close()
            self.energy = energy_tmp

            with open('orca_molecule.xyz', 'r') as f:
                self.xyz_string_opt = f.read()
            f.close()
            success = True
        return success

    def get_energy(self):
        """Get the energy of the molecule.

        Returns:
            energy (float) in eV
        Raises:
            AttributeError: if energy hasn't been calculated yet
        """
        if not hasattr(self, 'energy'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            return hartree2eV*self.energy

    def get_xyz_string_opt(self):
        """Get the optimized xyz string.

        Returns:
            optimized xyz string (str)
        Raises:
            AttributeError: if the optimization hasn't been performed yet
        """
        if not hasattr(self, 'xyz_string_opt'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            return self.xyz_string_opt

    def clean(self):
        """Clean the working direction after the orca calculation has been
        completed.
        """
        for f in glob.glob("orca_molecule.*"):
            os.remove(f)
