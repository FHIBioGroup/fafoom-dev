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
''' Collection of diverse run controlling utilites '''
from __future__ import division
import glob
import os, sys, shutil, time
import numpy as np
from utilities import print_output, remover_file, remover_dir, backup

def simple_or_restart():
    """ Select the type of run. If the all backup files are present the run
    will be restarted. Otherwise, the directory will be cleaned and a new run
    will be started."""

    for_restart = ["backup_population.dat", "backup_mol.dat",
                   "backup_min_energy.dat", "backup_iteration.dat",
                   "backup_blacklist.dat", "backup_new_blacklist.dat"]
    opt = "restart"
    for filename in for_restart:
        if glob.glob(filename):
            pass
        else:
            opt = "simple"
    if opt == "simple":
        print_output("Cleaning up the directory")
        for d in glob.glob("structure_*"):
            remover_dir(d)
        remover_dir("blacklist")
        for f in ["mol.sdf", "control.in", "geometry.in", "output.txt",
                  "result.out", "kill.dat"]:
            remover_file(f)
        for f in for_restart:
            remover_file(f)
    if opt == "restart":
        remover_file("kill.dat")
    return opt

def simple_or_restart_for_random():
    """ Select the type of run. If the all backup files are present the run
    will be restarted. Otherwise, the directory will be cleaned and a new run
    will be started."""

    for_restart = ["backup_mol.dat", "backup_blacklist.dat"]
    opt = "restart"
    for filename in for_restart:
        if glob.glob(filename):
            pass
        else:
            opt = "simple"
    if opt == "simple":
        print_output("Cleaning up the directory")
        for d in glob.glob("initial_*"):
            remover_dir(d)
        for d in glob.glob("generation_*_child*"):
            remover_dir(d)
        remover_dir("blacklist")
        for f in ["control.in", "geometry.in", "output.txt",
                  "result.out", "kill.dat"]:
            remover_file(f)
        for f in for_restart:
            remover_file(f)
    if opt == "restart":
        remover_file("kill.dat")
    return opt

def str_info(struct):
    """ Prints the information about the structure to the output file"""
    # print_output(struct)
    for dof in struct.dof:
        print_output('{}: {}'.format(dof.name, [float('{:.2f}'.format(float(x))) for x in dof.values]))

def relax_info(struct):
    """ Prints the information about the structure to the output file after
    the local optimization."""
    for dof in struct.dof:
        print_output('{:<9}: {}'.format(dof.name, [float('{:.2f}'.format(x)) for x in dof.values]))
    print_output(' ')

def inter_info(struct, list_to_add):
    """ Prints the information about the structure to the output file after
    the local optimization."""
    list_to_add  = []
    for dof in struct.dof:
        list_to_add.append('{:<9}: {}'.format(dof.name, [float('{:.2f}'.format(x)) for x in dof.values]))
    return list_to_add

def check_for_not_converged(dirname):
    """ Check if the not_converged.dat file is present in the directory or in the
    subdirectories. Folder will be deleted but calculation will be continued."""
    check = False
    if len(glob.glob("*/not_converged.dat")) == 0 and len(glob.glob("not_converged.dat")) == 0:
        pass
    else:
        print("Seems that something didn't converged. Don't worry, no problem.")
        shutil.rmtree(os.path.join(os.getcwd(), dirname))
        os.remove(os.path.join(os.getcwd(), 'not_converged.dat'))
        check = True
    return check

def check_for_kill():
    """ Check if the kill.dat file is present in the directory or in the
    subdirectories. If discoveed the run will be aborted."""
    if len(glob.glob("*/kill.dat")) == 0 and len(glob.glob("kill.dat")) == 0:
        pass
    else:
        print_output("Kill.dat file discovered. The code terminates")
        sys.exit(0)


def detect_energy_function(params):
    """ Detect the energy function that will be used for local optimization."""
    if 'energy_function' not in params:
        energy_function = "no"
        print_output("The energy function hasn't been defined."
                     " Random structures will be produced")
    else:
        if params['energy_function'] in ['No', 'no', 'Random', 'random']:
            print_output("Local optimization will not be performed. Random structures will be produced")
            energy_function = "no"
        elif params['energy_function'] in ['test', 'Test', 'TEST' 
                                                           'Random_test',
                                           'random_test']:
            print_output("Local optimization will not be performed. Random "
                         "input and output files will be produced")
            energy_function = "random_test"
        elif params['energy_function'] in ['aims', 'FHI-aims', 'FHIaims']:
            print_output("Local optimization will be performed with FHI-aims.")
            energy_function = "aims"
        elif params['energy_function'] in ['nwchem', 'NWChem']:
            print_output("Local optimization will be performed with NWChem.")
            energy_function = "nwchem"
        elif params['energy_function'] in ['ORCA', 'Orca', 'orca']:
            print_output("Local optimization will be performed with ORCA.")
            energy_function = "orca"
        elif params['energy_function'] in ['ff', 'force_field', 'RDKit',
                                           'rdkit']:
            print_output("Local optimization will be performed with RDKit.")
            energy_function = "ff"
        elif params['energy_function'] in ['FF', 'ForceField', 'INTERFACE']:
            print_output("Local optimization will be performed with INTERFACE ForceField.")
            energy_function = "INTERFACE"
        else:
            print_output("Unknown type of energy function."
                         " The code terminates.")
            sys.exit(0)
        return energy_function


def optimize(structure, energy_function, params, name=None):
    """Perform local optimization."""
    if energy_function == "no":
        structure.perform_random(params['sourcedir'], name)
    if energy_function == "random_test":
        structure.perform_random_test(params['sourcedir'], name)
    if energy_function == "aims":
        structure.perform_aims(params['sourcedir'], params['aims_call'], name)
    elif energy_function == "nwchem":
        structure.perform_nwchem(params['functional'], params['basis_set'],
                                 params['nwchem_call'])
    elif energy_function == "orca":
        linked_params = {}
        for key in ["chargemult", "nprocs", "optsteps"]:
            if key in params:
                linked_params[str(key)] = params[str(key)]
        structure.perform_orca(params['commandline'],
                               params['memory'],
                               params['orca_call'], **linked_params)
    elif energy_function == "ff":
        linked_params = {}
        for key in ["steps", "force_tol", "energy_tol"]:
            if key in params:
                linked_params[str(key)] = params[str(key)]
        structure.perform_ff(params['force_field'], **linked_params)

    elif energy_function == 'INTERFACE':
        structure.perform_FF(params['sourcedir'], params['ff_call'], name)

def single_point(structure, energy_function, params, name=None):
    """Perform local optimization."""
    if energy_function == "aims":
        structure.perform_aims_single_point(params['sourcedir'], params['aims_call'], name)
    elif energy_function == "nwchem":
        structure.perform_nwchem(params['functional'], params['basis_set'],
                                 params['nwchem_call'])
    elif energy_function == "orca":
        linked_params = {}
        for key in ["chargemult", "nprocs", "optsteps"]:
            if key in params:
                linked_params[str(key)] = params[str(key)]
        structure.perform_orca(params['commandline'],
                               params['memory'],
                               params['orca_call'], **linked_params)
    elif energy_function == "ff":
        linked_params = {}
        for key in ["steps", "force_tol", "energy_tol"]:
            if key in params:
                linked_params[str(key)] = params[str(key)]
        structure.perform_ff(params['force_field'], **linked_params)

    elif energy_function == 'INTERFACE':
        structure.perform_FF(params['sourcedir'], params['ff_call'], name)


def perform_backup(mol, population, blacklist, iteration, min_energy, new_blacklist):
    """Write object representation to files for a future restart."""
    backup("backup_mol.dat", mol)
    backup("backup_population.dat", population)
    backup("backup_blacklist.dat", blacklist)
    backup("backup_iteration.dat", iteration)
    backup("backup_min_energy.dat", min_energy)
    with open(os.path.join(os.getcwd(), "backup_new_blacklist.dat"), 'w') as NB:
        for structure in new_blacklist:
            NB.write(structure)
    # backup("new_blacklist.dat", new_blacklist)

def perform_backup_for_random(mol, blacklist):
    """Write object representation to files for a future restart."""
    backup("backup_mol.dat", mol)
    backup("backup_blacklist.dat", blacklist)

def perform_backup_for_FF(blacklist):
    """Write object representation to files for a future restart."""
    backup("backup_blacklist_FF.dat", blacklist)


def find_linked_params(mol, params):
    """ Check for parameters assosciated with the degrees of freedom."""
    linked_params = {}
    for dof_name in mol.dof_names:
        if 'prob_for_mut_'+str(dof_name) in params:
            linked_params['prob_for_mut_'+str(dof_name)] = \
                params['prob_for_mut_'+str(dof_name)]
        if 'max_mutations_'+str(dof_name) in params:
            linked_params['max_mutations_'+str(dof_name)] = \
                params['max_mutations_'+str(dof_name)]
    return linked_params

def HeadFafoom():
    print_output('          ------------------------------------------------------------')
    print_output('          Fafoom is free software: you can redistribute it and/or modify')
    print_output('          it under the terms of the GNU Lesser General Public License as published by')
    print_output('          the Free Software Foundation, either version 3 of the License, or')
    print_output('          (at your option) any later version.')
    print_output('\n')
    print_output('          Fafoom is distributed in the hope that it will be useful,')
    print_output('          but WITHOUT ANY WARRANTY; without even the implied warranty of')
    print_output('          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the')
    print_output('          GNU Lesser General Public License for more details.')
    print_output('\n')
    print_output('          You should have received a copy of the GNU Lesser General Public License')
    print_output('          along with fafoom.  If not, see <http://www.gnu.org/licenses/>.')
    print_output('\n')
    print_output('          When using Fafoom, please cite the following reference:')
    print_output('          "First-Principles Molecular Structure Search with a Genetic Algorithm"')
    print_output('          Adriana Supady, Volker Blum, and Carsten Baldauf')
    print_output('          J. Chem. Inf. Model., 2015, 55 (11), pp 2338-2348')
    print_output('          DOI: 10.1021/acs.jcim.5b00243')
    print_output('          ------------------------------------------------------------\n')

def ResultFafoom():
    print_output('          ------------------------------------------------------------')
    print_output('          Converged!')
    print_output('          Have a nice day!')
    print_output('          ------------------------------------------------------------')

def TimeSpent(StartTime):
    return time.time() - StartTime

def AnalysisFafoom(Trials, NotValid, Calculated, Known, Unique, TimeSpent):
    print_output(' ')
    print_output('{:<30}{:>15}'.format('Total trials', Trials))
    print_output('{:<30}{:>15}'.format('Invalid geometries', NotValid))
    print_output('{:<30}{:>15}'.format('Found in Blacklist', Known))
    print_output('{:<30}{:>15}'.format('Total structures calculated', Calculated))
    print_output(' ')
    print_output('{:<30}{:>15}'.format('Unique structures found', Unique))
    print_output('{:<30}{:>15.2f} s'.format('Time spent:', TimeSpent))
    print_output(' ')

def CheckForConvergence(Trials, NotValid, Known, Unique, TimeSpent, Calculations, params, min_energy):
    """Check the run for convergence"""
    if len(min_energy) >= params['iter_limit_conv'] :
        # print_output("Checking for convergence")
        d = abs(min_energy[-1] - min_energy[-params['iter_limit_conv']])
        if 'energy_wanted' in params:
            if min_energy[-1] < params['energy_wanted'] or \
               d < params['energy_diff_conv']:
                    ResultFafoom()
                    sys.exit(0)
                # print_output("Converged")
                # killfile = open("kill.dat", "w")
                # killfile.close()
                # sys.exit(0)
            # else:
            #     print_output("Not converged yet")
        else:
            if d < params['energy_diff_conv']:
                    ResultFafoom()
                    sys.exit(0)
                # print_output("Converged")
                # killfile = open("kill.dat", "w")
                # killfile.close()
                # sys.exit(0)
            # else:
            #     print_output("Not converged yet")
    if Calculations == params['max_iter']:
        print_output("Max. number of calculations reached. The code terminates")
        AnalysisFafoom(Trials, NotValid, Calculations, Known, Unique, TimeSpent)
        print('Minimal energy is {}'.format(min_energy[-1]))
        # killfile = open("kill.dat", "w")
        # killfile.close()
        sys.exit(0)

def GeneticOperationsOutput(Unique, Calculated, parent1, parent2, after_crossover, after_mutation):
    print_output('\n------------------------------------------------------------\n')
    print_output('Already calculated {} structures'.format(Calculated))
    print_output('Try to find unique structure {}:\n'.format(Unique))
    print_output('Parent 1: {}'.format(parent1))
    relax_info(parent1)
    print_output('Parent 2: {}'.format(parent2))
    relax_info(parent2)
    print_output('Child after crossover:')
    for item in after_crossover:
        print_output(item)
    print_output('\nChild after mutation:')
    for item in after_mutation:
        print_output(item)
    print_output('------------------------------------------------------------\n')


# def update_shared_blacklist(shared_blacklist, visited_folders, str3d):
#     other_GAs = []
#     selfdirectory = os.getcwd().split('/')[-1]
#     parent_directory = ('/').join(os.getcwd().split('/')[:-1])
#     for i in os.listdir(parent_directory):
#         if 'GA' in i and os.path.isdir(os.path.join(parent_directory, i)) and i != selfdirectory:
#             for calculated_structure in os.listdir(os.path.join(parent_directory, i)):
#                 if calculated_structure != 'adds' and os.path.isdir(os.path.join(parent_directory, i, calculated_structure)):
#                     other_GAs.append(os.path.join(parent_directory, i, calculated_structure))
#     for str_folder in other_GAs:
#         if str_folder not in visited_folders and selfdirectory not in str_folder:
#             if os.path.exists(os.path.join(str_folder, 'geometry_out.sdf')):
#                 with open(os.path.join(str_folder, 'geometry_out.sdf')) as sdf_out:
#                     str3d = Structure(mol)
#                     str3d.index = len(shared_blacklist) + 1000
#                     str3d.generate_structure()
#                     out_string = sdf_out.read()
#                     for dof in str3d.dof:
#                         dof.update_values(out_string)
#                     str3d.send_to_blacklist(shared_blacklist)
#                 with open(os.path.join(str_folder, 'geometry_in.sdf')) as sdf_out:
#                     str3d = Structure(mol)
#                     str3d.index = len(shared_blacklist) + 10000
#                     str3d.generate_structure()
#                     in_string = sdf_out.read()
#                     for dof in str3d.dof:
#                         dof.update_values(in_string)
#                     str3d.send_to_blacklist(shared_blacklist)
#                 visited_folders.append(str_folder)
#     return shared_blacklist, visited_folders


def Goodbye():
    print_output('	    :"":   :"":                            ')
    print_output('	     | \   / |                             ')
    print_output('             |  \_/  |                       ')
    print_output('            /,   ,__  :,-.    ')
    print_output('          .-|\  /  \    ,._')
    print_output('        .,  0/  | 0 |   \___:.')
    print_output('     .-,  _,|    --.:    \# ')
    print_output('      `--,| |      \      \# ')
    print_output('            |      | :::   \#                ')
    print_output('            \     ;|\:::   .\#               ')
    print_output('            |. . //  \::   ::\#              ')
    print_output('            \   /`    \      :\# ')
    print_output('             `"`       \..     \#            ')
    print_output('                        \::.    \#           ')
    print_output('                         \::     \#          ')
    print_output('                          \.    .:\#         ')
    print_output('                           \    :::\#        ')
    print_output('                            \    .::\# ')
    print_output('\n		Have a nice day!                   ')
















