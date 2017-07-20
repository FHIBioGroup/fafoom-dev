"""Wrapper for ForceField energy evaluation."""
import shutil
import os,re
import subprocess
import itertools
import pybel
import numpy as np

from utilities import sdf2xyz
import matplotlib.pyplot as plt

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def produce_xyz(path, sorted_dict):
    with open(os.path.join(path,'test.xyz'),'a') as xyz:
    	xyz.write('{}\n'.format(len(sorted_dict)))
    	xyz.write('Comment\n')
    	for i in sorted_dict:
    	    xyz.write('{}   {}   {}   {}\n'.format(i[0].split(':')[-1], i[1][1][0], i[1][1][1], i[1][1][2]))

class FFobject():
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
        if not os.path.exists(os.path.join(os.getcwd(),'par_all22_prot_metals.inp')):
            shutil.copy(os.path.join(self.sourcedir, 'par_all22_prot_metals.inp'), os.getcwd())
        if not os.path.exists(os.path.join(os.getcwd(),'psfgen')):
            shutil.copy(os.path.join(self.sourcedir, 'psfgen'), os.getcwd())

        with open(os.path.join(os.getcwd(), 'mol.sdf'), 'w') as mol_sdf:
            mol_sdf.write(sdf_string)

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

    def build_storage(self, dirname):
        """ Create folder for storing input and output of FF local optimization"""
        self.dirname = dirname
        os.mkdir(os.path.join(dirname))
        shutil.copy('all.pdb', os.path.join(dirname, 'all.pdb'))
        shutil.copy('all.psf', os.path.join(dirname, 'all.psf'))
        shutil.copy(os.path.join(self.sourcedir, 'Configure.conf'), os.path.join(dirname,'Configure.conf'))
        shutil.copy(os.path.join(self.sourcedir, 'par_all22_prot_metals.inp'),os.path.join(dirname, 'par_all22_prot_metals.inp'))

    def run_FF(self, execution_sctring):
        """ Execute FF local optimization """
        path_to_run = os.path.join(os.getcwd(), self.dirname)
        os.system('cd {} && {}'.format(path_to_run, execution_sctring))

        with open(os.path.join(path_to_run, 'result.out'), 'r') as result:
            energies = []
            lines = result.readlines()
            for line in lines:
                energy_found = re.match(r'(ENERGY:\s+(\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+))', line)
                if energy_found:
                    energies.append(energy_found.group(12))
            self.energy = float(min(energies))
        # os.system('cd {} && babel -ipdb {} -oxyz {}'.format(path_to_run, 'result.coor', 'result.xyz'))
        old_dict = {}
        new_dict = {}

        with open(os.path.join(self.sourcedir, 'mol.pdb'), 'r') as o:
            lines = o.readlines()
            for line in lines:
        	pdb_line_found = re.match(r'((\w+)\s+(\d+)\s+(...)\s+(\w\w\w)\s+(\w+)\s+(\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(\w+)\s+(\w+))', line)
        	if pdb_line_found:
        	    old_dict['{}:{}:{}'.format(pdb_line_found.group(5).strip(' '), pdb_line_found.group(4).strip(' '), pdb_line_found.group(14).strip(' '))] = int(pdb_line_found.group(3))

        with open(os.path.join(path_to_run, 'result.coor'), 'r') as n:
            lines = n.readlines()
            for line in lines:
        	pdb_line_found = re.match(r'((\w+)\s+(\d+)\s+(...)\s+(\w\w\w)\s+(\w+)\s+(\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(\w+)\s+(\w+))', line)
        	if pdb_line_found:
        	    if pdb_line_found.group(13) == 'MOL':
        		new_dict['{}:{}:{}'.format(pdb_line_found.group(5), pdb_line_found.group(4).strip(' '), pdb_line_found.group(14).strip(' '))] = [old_dict['{}:{}:{}'.format(pdb_line_found.group(5), pdb_line_found.group(4).strip(' '), pdb_line_found.group(14).strip(' '))], [float(pdb_line_found.group(8)),float(pdb_line_found.group(9)),float(pdb_line_found.group(10))]]
        new_dict_sorted = sorted(new_dict.items(), key=lambda x: x[1][0])
        produce_xyz(path_to_run, new_dict_sorted)
        with open(os.path.join(path_to_run, 'test.xyz'), 'r') as output:
            self.FF_string_opt = output.read()



    def get_energy(self):
        return self.energy

    def get_FF_string_opt(self):
        if not hasattr(self, 'FF_string_opt'):
            raise AttributeError("The calculation wasn't performed yet.")
        else:
            return self.FF_string_opt



    def analysis(self):
        analysis = {}
        path = os.path.join(os.getcwd() ,'valid_for_FF')
        fig, axs = plt.subplots(facecolor='w', edgecolor='k')
        fig.suptitle('Test', fontsize=16)
        for i in natural_sort(os.listdir(path)):
            analysis[i] = []
            result_out = os.path.join(path, i, 'result.out')
            if os.path.exists(os.path.join(path, i, 'result.out')):
                with open(result_out, 'r') as result:
                    lines = result.readlines()
                    for line in lines:
                        energy_found = re.match(r'(ENERGY:\s+(\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+))', line)
                        if energy_found:
                            analysis[i].append(energy_found.group(12))
            axs.plot(range(1, len(analysis[i])+1), analysis[i])
            # axs.plot(slabs, binding_energy, 'ro')
            # axs.set_xticks(slabs)
            # axs.set_yticks(binding_energy)
            # axs.set_yticklabels(tick_lbls)
            #~ axs.set_ylim(miny, maxy)
            #~ axs.set_xlim(0, 1.05*max(cpus))
        axs.set_xlabel('Energy step', fontsize=16)
        axs.set_ylabel('Total energy, meV', fontsize=16)
            # fig.savefig('Total_energy.png', dpi=150)
        plt.show()                            # print '#{}: vdW: {}, Total: {}'.format(energy_found.group(2), energy_found.group(8), energy_found.group(12))
