#!/usr/bin/python
import numpy as np
import sys, os, re, shutil

from fafoom import *
from fafoom import MoleculeDescription, Structure, selection, print_output,\
    remover_dir, set_default, file2dict
import fafoom.run_utilities as run_util
from visual import draw_picture
from pyforcefield import *

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from measure import centroid_measure
from utilities import sdf2xyz, check_for_clashes

# Decide for restart or a simple run.
opt = run_util.simple_or_restart()
p_file = sys.argv[1]
# Build a dictionary from two section of the parameter file.
params = file2dict(p_file, ['GA settings', 'Run settings'])

dict_default = {'energy_var': 0.001, 'selection': "roulette_wheel",
                'fitness_sum_limit': 1.2, 'popsize': 10,
                'prob_for_crossing': 1.0, 'max_iter': 30,
                'iter_limit_conv': 20, 'energy_diff_conv': 0.001}
# Set defaults for parameters not defined in the parameter file.
params = set_default(params, dict_default)
energy_function = run_util.detect_energy_function(params)

cnt_max = 100
population, blacklist = [], []
min_energy = []

#***********************************************************************
"""
Creation of the folders for valid and invalid structures.
It helps to visually inspect produced structures.
"""
if os.path.exists(os.path.join(os.getcwd(),'invalid')):
    shutil.rmtree(os.path.join(os.getcwd(),'invalid'))
    os.mkdir(os.path.join(os.getcwd(),'invalid'))
else:
    os.mkdir(os.path.join(os.getcwd(),'invalid'))

if os.path.exists(os.path.join(os.getcwd(),'valid')):
    shutil.rmtree(os.path.join(os.getcwd(),'valid'))
    os.mkdir(os.path.join(os.getcwd(),'valid'))
else:
    os.mkdir(os.path.join(os.getcwd(),'valid'))

if os.path.exists(os.path.join(os.getcwd(),'valid_for_FF')):
    shutil.rmtree(os.path.join(os.getcwd(),'valid_for_FF'))
    os.mkdir(os.path.join(os.getcwd(),'valid_for_FF'))
else:
    os.mkdir(os.path.join(os.getcwd(),'valid_for_FF'))
#=======================================================================

if opt == "simple":
    mol = MoleculeDescription(p_file)
    # Assign the permanent attributes to the molecule.
    mol.get_parameters()
    mol.create_template_sdf()
    # Check for potential degree of freedom related parameters.
    linked_params = run_util.find_linked_params(mol, params)
    #~ print_output("Number of atoms: "+str(mol.atoms))
    #~ print_output("Number of bonds: "+str(mol.bonds))
    for dof in mol.dof_names:
        print_output("Number of identified "+str(dof)+": " + str(len(getattr(mol, dof))))
        print_output("Identified "+str(dof)+": "+str(getattr(mol, dof)))
    print_output("Will generate {} structures".format(cnt_max))
    cnt = 0
    # Generate sensible and unique 3d structures.
    while len(population) < params['popsize'] and cnt < cnt_max:
        print_output("\n")
        str3d = Structure(mol)
        str3d.generate_structure()
        #~ print '\n{}'.format(sdf2xyz(str3d.sdf_string))
        test_ff = FFobject(os.path.join(os.getcwd(),'adds','FF'))
        # test_ff.update_pdb_coords(str3d.sdf_string)
        aims_object = AimsObject(os.path.join(os.getcwd(),'adds')) #Need for creation of the input file. Does not affect the algoritm.
        if not str3d.is_geometry_valid():
            print_output("The geometry of "+str(str3d)+" is invalid. Copied to /invalid")
            aims_object.generate_input(str3d.sdf_string) #generates input
            shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            cnt += 1
            continue
        if str3d not in blacklist:
            print_output("The geometry of "+str(str3d)+" is valid, copied to /valid")
            if 'centroid' not in mol.dof_names:
                str3d.adjust_position()
                if not check_for_clashes(str3d.sdf_string, os.path.join(aims_object.sourcedir, 'geometry.in.constrained')):
                    check = False
                    for i in range(50):
                        print 'Before {}'.format(centroid_measure(str3d.sdf_string))
                        str3d.adjust_position()
                        print 'After {}'.format(centroid_measure(str3d.sdf_string))
                        check = check_for_clashes(str3d.sdf_string, os.path.join(aims_object.sourcedir, 'geometry.in.constrained'))
                        if check:
                            break
		if check == False:
		    print 'Increase the volume!!!'
                break
	    # print str3d.sdf_string
            name = str(cnt)+'_geometry'
            # run_util.optimize(str3d, energy_function, params, name)
            test_ff.generate_input(str3d.sdf_string)
            test_ff.build_storage(str(cnt)+'_geometry')
            test_ff.run_FF('~/programs/namd/namd2 Configure.conf > result.out')

            # os.system('cd {} && ~/programs/namd/namd2 Configure.conf > result.out'.format(os.path.join(os.getcwd(), 'valid_for_FF', str(cnt)+'_geometry')))
            # test_ff.analysis()
            #~ aims_object.generate_input(str3d.sdf_string) #generates input
            #~ os.mkdir(os.path.join(os.getcwd(),'valid',str(cnt)+'_geometry')) # creates the folder for particular structure inside th "valid" folder
            #~ shutil.copy('geometry.in',os.path.join(os.getcwd(), 'valid', str(cnt)+'_geometry','geometry.in')) # copy input to self-titled folder
            cnt += 1
