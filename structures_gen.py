#!/usr/bin/python
import numpy as np
import sys
import os
import re
import shutil

from fafoom import *
from fafoom import MoleculeDescription, Structure, selection, print_output,\
    remover_dir, set_default, file2dict
import fafoom.run_utilities as run_util
from fafoom.measure import centroid_measure
from fafoom.utilities import sdf2xyz, check_for_clashes
from fafoom.connectivity import *
parser = OptionParser()
parser.add_option("-t", "--test", dest="test", default = None, help="Testing mode will turn on np.random.seed(0) and random number will be predictable and the same. For testing purposes.")
(options, args) = parser.parse_args()

if options.test is not None:
    np.random.seed(0)
"""Find species_defaults folder"""
path = '/'.join(['{}'.format(str(x)) for x in sys.argv[0].split('/')[:-1]])
species_default_folder = os.path.join(path, 'species_defaults')
opt = run_util.simple_or_restart_for_random()
#Atoms available in FHI-AIMS:
def atoms_avail(species_default_folder, basis):
    Atoms = []
    for i in os.listdir(os.path.join(species_default_folder, 'light')):
        Atom = re.match(r'(.+_(.+)_.+)',i)
        if Atom:
            Atoms.append(Atom.group(2))
    return Atoms


def identify_species(geometry_file):
    species = []
    with open(geometry_file, 'r') as geom:
        lines = geom.readlines()
        for line in lines:
            species_found = re.match(r'(\s+?.*?\d+\.\d+\s+.*?\d+\.\d+\s+.*?\d+\.\d+\s+(\w+)\s+.+)', line)
            if species_found:
                if species_found.group(2) not in species:
                    species.append(species_found.group(2))
    return species

def produce_aims_control_light(geometry_file):
    species = identify_species(geometry_file)
    Basis = 'light'
    """ Produce control file for geometry optimization with FHI-AIMS"""
    """                 LIGHT SETTINGS will be used!!!              """

    with open(os.path.join(os.getcwd(), 'adds', 'control_single_point.in'), 'w') as control:
    	control.write('# Physical settings\n')
    	control.write('  xc                 pbe\n')
    	control.write('  spin               none\n')
    	control.write('  relativistic       atomic_zora scalar\n')
        control.write('  vdw_correction_hirshfeld\n')
        control.write('  charge             0.0\n')
    	control.write('\n')
    	control.write('# SCF settings\n')
    	control.write('  sc_accuracy_rho    1E-4\n')
    	control.write('  sc_accuracy_eev    1E-2\n')
    	control.write('  sc_accuracy_etot   1E-5\n')
    	control.write('  sc_iter_limit      200\n')
    	control.write('\n')
        for Atom_type in species:
            with open(glob.glob(os.path.join(species_default_folder, Basis,'*_'+str(Atom_type)+'_default'))[0],'r') as basis_for_atom:
                control.write(basis_for_atom.read())
                control.write('\n')

    with open(os.path.join(os.getcwd(), 'adds', 'control.in'), 'w') as control:
    	control.write('# Physical settings\n')
    	control.write('  xc                 pbe\n')
    	control.write('  spin               none\n')
    	control.write('  relativistic       atomic_zora scalar\n')
        control.write('  vdw_correction_hirshfeld\n')
        control.write('  charge             0.0\n')
    	control.write('\n')
    	control.write('# SCF settings\n')
    	control.write('  sc_accuracy_rho    1E-4\n')
    	control.write('  sc_accuracy_eev    1E-2\n')
    	control.write('  sc_accuracy_etot   1E-5\n')
        control.write('  sc_accuracy_forces 1E-3')
    	control.write('  sc_iter_limit      200\n')
    	control.write('\n')
        control.write('#  Relaxation\n')
        control.write('#RI_method LVL_fast\n')
        control.write('  relax_geometry trm 5E-2\n')
        for Atom_type in species:
            with open(glob.glob(os.path.join(species_default_folder, Basis,'*_'+str(Atom_type)+'_default'))[0],'r') as basis_for_atom:
                control.write(basis_for_atom.read())
                control.write('\n')



#Need to correctly write the one-line blacklist:
np.set_printoptions(suppress=True)
# Decide for restart or a simple run.
opt = run_util.simple_or_restart()
# If genetic algorithm was invoked without additional inputs
# fafoom will try to find parameters.txt file as default.
if len(sys.argv) < 2:
    if os.path.exists(os.path.join(os.getcwd(), 'parameters.txt')):
        p_file = os.path.join(os.getcwd(), 'parameters.txt')
    else:
        raise Exception('Please produce parameters.txt file.')
else:
    p_file = sys.argv[1]
# Assign default parameters for calculation
# Build a dictionary from two section of the parameter file.
params = file2dict(p_file, ['GA settings', 'Run settings'])
# Default parameters:
dict_default = {'energy_var': 0.001, 'selection': "roulette_wheel",
                'fitness_sum_limit': 1.2, 'popsize': 10,
                'prob_for_crossing': 1.0, 'max_iter': 30,
                'iter_limit_conv': 20, 'energy_diff_conv': 0.001}
# Set defaults for parameters not defined in the parameter file.
params = set_default(params, dict_default)
# Maximum number of trials to produce the apropriate geometry.
cnt_max = 250
# Create lists to store Population, minimal energy in the run and
# structures that are already calculated.
population, blacklist, min_energy = [], [], []
new_blacklist = []
#=======================================================================
aims_object = AimsObject(os.path.join(os.getcwd(),'adds'))
#ff_object = FFobject(os.path.join(os.getcwd(),'adds', 'FF'))

if opt == "simple":
    if os.path.exists(os.path.join(os.getcwd(),'RandGen')):
        shutil.rmtree(os.path.join(os.getcwd(),'RandGen'))
        os.mkdir(os.path.join(os.getcwd(),'RandGen'))
    else:
        os.mkdir(os.path.join(os.getcwd(),'RandGen'))

mol = MoleculeDescription(p_file)
# Assign the permanent attributes to the molecule.
mol.get_parameters()
mol.create_template_sdf()
volume = mol.volume
# Check for potential degree of freedom related parameters.
linked_params = run_util.find_linked_params(mol, params)
print_output("Number of atoms: "+str(mol.atoms))
print_output("Number of bonds: "+str(mol.bonds))
for dof in mol.dof_names:
    print_output("Number of identified "+str(dof)+": " +
                 str(len(getattr(mol, dof))))
    print_output("Identified "+str(dof)+": "+str(getattr(mol, dof)))
cnt = 0
population = 0
# Generate sensible and unique 3d structures.

if opt == "restart":
    with open("backup_blacklist.dat", 'r') as inf:
        for line in inf:
            blacklist.append(eval(line))
    population = len(os.listdir(os.path.join(os.getcwd(), 'RandGen')))

while population < params['popsize'] and cnt < cnt_max:
    str3d = Structure(mol)
    str3d.generate_structure()
    if not str3d.is_geometry_valid():
        # print_output("The geometry of "+str(str3d)+" is invalid. Copied to /invalid")
        # with open(os.path.join(os.getcwd(), 'invalid', 'structure_{}.sdf'.format(cnt)), 'w') as rand_structure:
        #     rand_structure.write(str3d.sdf_string) #generates input
        cnt += 1
        continue
    if str3d not in blacklist:
        if not check_for_clashes(str3d.sdf_string, os.path.join(mol.constrained_geometry_file)):
            if 'centroid' not in mol.dof_names:
                str3d.adjust_position()
            else:
                cnt+=1
                continue

        if len(aims2xyz(os.path.join(os.getcwd(), mol.constrained_geometry_file))) < 3:
            if check_geo_if_not_too_far(str3d.sdf_string, os.path.join(os.getcwd(), mol.constrained_geometry_file), flag=1.0) == False:
                str3d.adjust_position_centroid(os.path.join(os.getcwd(), mol.constrained_geometry_file))

        if 'centroid' not in mol.dof_names and len(aims2xyz(os.path.join(os.getcwd(), 'adds', 'geometry.in.constrained'))) > 0:
            if not str3d.check_position(volume):
                str3d.adjust_position()
        aims_object.generate_input(str3d.sdf_string)
        aims_object.build_storage(os.path.join(os.getcwd(),'RandGen', 'structure_{}'.format(population + 1)))
        # with open(os.path.join(os.getcwd(), 'RandGen', 'structure_{}.sdf'.format(population + 1)), 'w') as rand_structure:
            # rand_structure.write(str3d.sdf_string) #generates input
        # name = os.path.join(os.getcwd(), 'valid', str(cnt)+'_geometry')
        str3d.send_to_blacklist(blacklist)
        for dof in str3d.dof:
            print('{} {}'.format(dof.name, dof.values))
        population += 1
    else:
        cnt += 1
        print 'CNT {}'.format(cnt)
        print "Geomerty of "+str(str3d)+" is fine, but already known."
    if cnt == cnt_max:
        print_output("The allowed number of trials for building the "
                     "population has been exceeded. The code terminates.")
        run_util.perform_backup_for_random(mol, blacklist)
run_util.perform_backup_for_random(mol, blacklist)
