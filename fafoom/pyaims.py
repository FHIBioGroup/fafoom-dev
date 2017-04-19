#    Copyright 2015 Adriana Supady
#
#    This file is part of fafoom.
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
#  You should have received a copy of the GNU Lesser General Public License
#   along with fafoom.  If not, see <http://www.gnu.org/licenses/>.

"""Wrapper for FHI-aims."""
from __future__ import division
import shutil
import os
import subprocess
import itertools

from utilities import remover_file, sdf2aims, string2file


class AimsObject():
    """Create and handle FHI-aims objects."""
    def __init__(self, sourcedir):
        """Initialize the object. The sourcedir is the directory with the
        'control.in' file
        """
        self.sourcedir = sourcedir

    def generate_input(self, sdf_string):
        """Create input files for FHI-aims.
        Args:
            sdf_string (str)
        """
        self.aims_string = sdf2aims(sdf_string)
        string2file(self.aims_string, 'geometry.in')
        name = 'control.in'
        src = os.path.join(self.sourcedir, name)
        shutil.copy(src, os.getcwd())

    def build_storage(self, dirname):
        """Create a directory for storing the FHI-aims input and output.
        Args:
            dirname (str)
        Raises:
            OSError: if the directory is already in use
        """
        if os.path.isdir(dirname):
            raise OSError("The directory already exists.")
        self.dirname = dirname
        os.mkdir(self.dirname)
        shutil.copy('geometry.in', self.dirname)
        shutil.copy('control.in', self.dirname)

    def run_aims(self, execution_string):
        """Run FHI-aims and write output to 'result.out'. The optimized
        geometry is written to 'geometry.out'. If the run fails due to
        convergence issue, file 'kill.dat' will be created.

        Warning: this function uses subprocessing to invoke the run.
        The subprocess's shell is set to TRUE.
        Args:
            execution_string (str): e.g. mpirun -n 4 aims.*.scalapack.mpi.x
        Raises:
            OSError: if geometry.in or control.in not present in the working
            directory
        """
        success = False
        for filename in ['control.in', 'geometry.in']:
            if os.path.exists(filename) is False:
                raise OSError("Required input file not present.")
        aims = subprocess.Popen(
            execution_string, stdout=subprocess.PIPE, shell=True)
        out = subprocess.Popen(
            ['cat'], stdin=aims.stdout,
            stdout=open('result.out', 'w'), shell=True)
        out.wait()
        s0 = "Present geometry is converged"
        s = "Total energy of the DFT / Hartree-Fock s.c.f. calculation      :"
        s2 = "Final atomic structure:"
        not_conv = True
        searchfile = open("result.out", "r")
        for line in searchfile:
            if s0 in line:
                not_conv = False
        searchfile.close()

        if not_conv:
            killfile = open("kill.dat", "w")
            killfile.close()

        else:
            searchfile = open("result.out", "r")
            for i, line in enumerate(searchfile, 1):
                if s in line:
                    a = line.split(" ")
                    self.energy = float('{0:.4f}'.format(float(a[-2])))
                if s2 in line:
                    l_num = int(i)
            searchfile.close()
            atoms = len(self.aims_string.splitlines())
            with open('geometry.out', 'w') as file_geo:
                with open('result.out') as f:
                    for line in itertools.islice(f, l_num+1, l_num+1+atoms):
                        file_geo.write(line)
            file_geo.close()
            f.close()
            with open('geometry.out', 'r') as f:
                self.aims_string_opt = f.read()
            f.close()
            success = True

        return success

    def get_energy(self):
        """Get the energy of the molecule.

        Returns:
            energy (float)
        Raises:
            AttributeError: if energy hasn't been calculated yet
        """
        if not hasattr(self, 'energy'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            return self.energy

    def get_aims_string_opt(self):
        """Get the optimized aims string.

        Returns:
            optimized aims string (str)
        Raises:
            AttributeError: if the optimization hasn't been performed yet
        """
        if not hasattr(self, 'aims_string_opt'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            return self.aims_string_opt

    def clean_and_store(self):
        """Store the output and clean the working direction after the FHI-aims
        calculation has been completed.
        """
        try:
            os.remove('geometry.in')
            os.remove('control.in')
            shutil.copy('result.out', self.dirname)
            os.remove('result.out')
            remover_file('geometry.in.next_step')
            shutil.copy('geometry.out', self.dirname)
            os.remove('geometry.out')
        except IOError:
            pass
