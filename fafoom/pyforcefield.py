"""Wrapper for ForceField energy evaluation."""
import shutil
import os,re
import subprocess
import itertools
import pybel
import numpy as np

from utilities import sdf2xyz




class ForceField():
    """Create and handle ForceField object."""
    def __init__(self, sourcedir):
        """Initialize the object. The sourcedir is the directory
        with forcefield files and utilities"""
        self.sourcedir = sourcedir

    def generate_input(self, sdf_string):
        """ Copy files if necessary"""
        if not os.path.exists(os.path.join(os.getcwd(),'surrounding.pdb')):
            shutil.copy(os.path.join(self.sourcedir, 'surrounding.pdb'), os.getcwd())
        if not os.path.exists(os.path.join(os.getcwd(),'prepare_psf.run')):
            shutil.copy(os.path.join(self.sourcedir, 'prepare_psf.run'), os.getcwd())
        if not os.path.exists(os.path.join(os.getcwd(),'top_all22_prot_metals.inp')):
            shutil.copy(os.path.join(self.sourcedir, 'top_all22_prot_metals.inp'), os.getcwd())
        if not os.path.exists(os.path.join(os.getcwd(),'par_all22_prot_metals.inp')):
            shutil.copy(os.path.join(self.sourcedir, 'par_all22_prot_metals.inp'), os.getcwd())
        if not os.path.exists(os.path.join(os.getcwd(),'Configure.conf')):
            shutil.copy(os.path.join(self.sourcedir, 'Configure.conf'), os.getcwd())
        if not os.path.exists(os.path.join(os.getcwd(),'psfgen')):
            shutil.copy(os.path.join(self.sourcedir, 'psfgen'), os.getcwd())

        """ Update coords in prepared mol.pdb file in the sourcedirectory. """
        self.coords = [i[1:] for i in sdf2xyz(sdf_string)] #Coordinates to update
        """ Extract lines from pdb file for further updating """
        pdb_file = []
        with open(os.path.join(self.sourcedir, 'mol.pdb'),'r') as molfile:
            lines = molfile.readlines()
            for line in lines:
                pdb_line_found = re.match(r'((\w+)\s+(\d+)\s+(...)\s+(\w\w\w)\s+(\w+)\s+(\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(\w+)\s+(\w+))', line)
                if pdb_line_found:
                    pdb_file.append([pdb_line_found.group(2),
                                     pdb_line_found.group(3),
                                     pdb_line_found.group(4),
                                     pdb_line_found.group(5),
                                     pdb_line_found.group(6),
                                     pdb_line_found.group(7),
                                     pdb_line_found.group(8),
                                     pdb_line_found.group(9),
                                     pdb_line_found.group(10),
                                     pdb_line_found.group(11),
                                     pdb_line_found.group(12),
                                     pdb_line_found.group(13),
                                     pdb_line_found.group(14)])
        # print pdb_file
        """ Update coordinates for pdb file from sdf string"""
        updated_pdb = [[pdb_file[i][0],
                        pdb_file[i][1],
                        pdb_file[i][2],
                        pdb_file[i][3],
                        pdb_file[i][4],
                        pdb_file[i][5],
                        self.coords[i][0],
                        self.coords[i][1],
                        self.coords[i][2],
                        pdb_file[i][9],
                        pdb_file[i][10],
                        pdb_file[i][11],
                        pdb_file[i][12]] for i in range(len(pdb_file))]

        """Write to updated pdb file """
        if os.path.exists(os.path.join(os.getcwd(), 'mol.pdb')):
            os.remove(os.path.join(os.getcwd(), 'mol.pdb'))
        # print updated_pdb
        with open(os.path.join(os.getcwd(), 'mol.pdb'), 'a') as updated_file:
            for line in updated_pdb:
                updated_file.write('{:6} {:>4}  {:3} {:>3} {} {:>3}    {:>7} {:>7} {:>7} {:>3} {:>5}   {:4} {:>2}\n'.format(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12]))

        os.system('cd {} && ./prepare_psf.run'.format(os.getcwd()))
