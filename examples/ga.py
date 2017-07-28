import numpy as np
import sys, os, re, shutil

from fafoom import *
from fafoom import MoleculeDescription, Structure, selection, print_output,\
    remover_dir, set_default, file2dict
import fafoom.run_utilities as run_util
#from visual import draw_picture

from numpy import array
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import Draw

from fafoom.measure import *
from fafoom.utilities import sdf2xyz, check_for_clashes

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

cnt_max = 200
population = []
blacklist = [] #Blacklist
min_energy = []

# smil = '[NH3+][C@H](C(=O)N1[C@H](C(=O)N[C@H](C(=O)[O-])Cc2ccccc2)CCC1)Cc1[nH]c[nH+]c1'
# m = Chem.MolFromSmiles(smil)
# Draw.MolToFile(m,'N2_HisHProPhe_C2.pdf', size=(250, 250), imageType='pdf',fitImage=True )
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
aims_object = AimsObject(os.path.join(os.getcwd(),'adds'))

if opt == "simple":
    mol = MoleculeDescription(p_file)
    # Assign the permanent attributes to the molecule.
    mol.get_parameters()
    mol.create_template_sdf()
    # Check for potential degree of freedom related parameters.
    linked_params = run_util.find_linked_params(mol, params)
    print_output("Number of atoms: "+str(mol.atoms))
    print_output("Number of bonds: "+str(mol.bonds))
    for dof in mol.dof_names:
        print_output("Number of identified "+str(dof)+": " +
                     str(len(getattr(mol, dof))))
        print_output("Identified "+str(dof)+": "+str(getattr(mol, dof)))

    print_output("___Initialization___")
    cnt = 0
    # Generate sensible and unique 3d structures.
    while len(population) < params['popsize'] and cnt < cnt_max:
        print_output("New trial")
        str3d = Structure(mol)
        str3d.generate_structure()
        # test_ff = FFobject(os.path.join(os.getcwd(),'adds','FF'))
        #~ print '\n{}'.format(sdf2xyz(str3d.sdf_string))
        # aims_object = AimsObject(os.path.join(os.getcwd(),'adds')) #Need for creation of the input file. Does not affect the algoritm.
        if not str3d.is_geometry_valid():
            print_output("The geometry of "+str(str3d)+" is invalid. Copied to /invalid")
            # aims_object.generate_input(str3d.sdf_string) #generates input
            # shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            cnt += 1
            continue
        if str3d not in blacklist:
            print_output("The geometry of "+str(str3d)+" is valid, copied to /valid")
            str3d.adjust_position()
            if not check_for_clashes(str3d.sdf_string, os.path.join(mol.constrained_geometry_file)):
                str3d.adjust_position()
	    # print str3d.sdf_string
            # test_ff.generate_input(str3d.sdf_string)
            # test_ff.build_storage(str(cnt)+'_geometry')
            # test_ff.run_FF('~/programs/namd/namd2 Configure.conf > result.out')
            # aims_object.generate_input(str3d.sdf_string) #generates input
            # os.mkdir(os.path.join(os.getcwd(),'valid',str(cnt)+'_geometry')) # creates the folder for particular structure inside th "valid" folder
            # shutil.copy('geometry.in',os.path.join(os.getcwd(), 'valid', str(cnt)+'_geometry','geometry.in')) # copy input to self-titled folder
############            draw_picture(os.path.join(os.getcwd(), 'valid', str(cnt)+'_geometry','geometry.in'), image_write = 'yes') # Part of post-processing module. Under construction. Produce nice image with PyMol
            name = os.path.join(os.getcwd(), 'valid', str(cnt)+'_geometry')
            # name = "initial_%d" % (len(population))
            # Perform the local optimization
            # print 'energy_function {}'.format(energy_function)
            # print 'params {}'.format(params)
            # print 'name {}'.format(name)
            run_util.optimize(str3d, energy_function, params, name)
            run_util.check_for_kill()
            str3d.send_to_blacklist(blacklist) #Blacklist
            population.append(str3d)
            print_output(str(str3d)+", energy: "+str(float(str3d)) +
                         ", was added to the population")
            run_util.relax_info(str3d)
            cnt += 1
        else:
            print_output(blacklist) #Blacklist
            print_output("Geomerty of "+str(str3d)+" is fine, but already "
                         "known.")
            cnt += 1
    if cnt == cnt_max:
        print_output("The allowed number of trials for building the "
                     "population has been exceeded. The code terminates.")
        sys.exit(0)
    print_output("___Initialization completed___")
    population.sort()
    print_output("Initial population after sorting: ")
    for i in range(len(population)):
        print_output(str(population[i])+" "+str(float(population[i])))
    min_energy.append(population[0].energy)
    print_output("Blacklist: " + ', '.join([str(v) for v in blacklist])) #Blacklist
    iteration = 0


if opt == "restart":
    # Reconstruct the molecule, population, blacklist and the state of the run.
    print_output(" \n ___Restart will be performed___")
    with open("backup_mol.dat", 'r') as inf:
        mol = eval(inf.readline())
    with open("backup_population.dat", 'r') as inf:
        for line in inf:
            population.append(eval(line))
    with open("backup_blacklist.dat", 'r') as inf:
        for line in inf:
            blacklist.append(eval(line))
    inf.close()  #Blacklist
    with open("backup_min_energy.dat", 'r') as inf:
        for line in inf:
            min_energy.append(eval(line))
    with open("backup_iteration.dat", 'r') as inf:
        iteration_tmp = eval(inf.readline())
    linked_params = run_util.find_linked_params(mol, params)
    population.sort()
    for i in range(len(population)):
        print_output(str(population[i])+" "+str(float(population[i])))
    print_output("Blacklist: " + ', '.join([str(v) for v in blacklist]))
    iteration = iteration_tmp+1
    print_output(" \n ___Reinitialization completed___")
    remover_dir('generation_'+str(iteration)+'_child1')
    remover_dir('generation_'+str(iteration)+'_child2')


def mutate_and_relax(candidate, name, iteration, cnt_max, **kwargs):
    print_output("__%s__" % name)
    found = False
    cnt = 0
    while found is False and cnt < cnt_max:
        candidate_backup = Structure(candidate)
        candidate.mutate(**kwargs)
        print_output("%s after mutation: " % name + str(candidate))
        run_util.str_info(candidate)
        if not candidate.is_geometry_valid():
            print_output(" The geometry of %s is invalid." % name)
            cnt += 1
            # Rebuild the structure
            candidate = candidate_backup
            continue

        if candidate not in blacklist:
            candidate.adjust_position_after_crossover()
            if not check_for_clashes(candidate.sdf_string, os.path.join(mol.constrained_geometry_file)):
                candidate.adjust_position_after_crossover()
            name = "generation_%d_%s" % (iteration, name)
            run_util.optimize(candidate, energy_function, params, name)
            run_util.check_for_kill()
            #candidate.send_to_blacklist(blacklist) #Blacklist
            print_output(str(candidate)+":, energy: "+str(float(
                candidate))+", is temporary added to the population")
            run_util.relax_info(candidate)
            found = True
            population.append(candidate)
        else:
            print_output("Geomerty of "+str(candidate)+" is fine, but already "
                         "known.")
            cnt += 1
            candidate = candidate_backup
        if cnt == cnt_max:
            raise Exception("The allowed number of trials for generating"
                            " a unique child has been exceeded.")

while iteration < params['max_iter']:
    print_output(" \n ___Start of iteration " + str(iteration) + "___")
    (parent1, parent2, fitness) = selection(population, params['selection'],
                                            params['energy_var'],
                                            params['fitness_sum_limit'])
    param = np.random.rand()
    cnt = 0
    while param < params['prob_for_crossing'] and cnt < cnt_max:
        child1, child2 = Structure.crossover(parent1, parent2)
        if child1.is_geometry_valid_after_crossover() and child2.is_geometry_valid_after_crossover():
            aims_object.generate_input(child1.sdf_string) #generates input
            shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_child1_not_adjusted'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder

            child1.adjust_position_after_crossover()
            child2.adjust_position_after_crossover()
            aims_object.generate_input(child1.sdf_string) #generates input
            shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_child1_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            aims_object.generate_input(child2.sdf_string) #generates input
            shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_child2_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            aims_object.generate_input(parent1.sdf_string) #generates input
            shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_parent1_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            aims_object.generate_input(parent2.sdf_string) #generates input
            shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_parent2_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder

            print_output("Crossover outcome: "+str(child1)+(", ")+str(child2))
            break
        else:
            # print 'Parent1 {}'.format(str(getattr(parent1.dof[0], "values")))
            # print 'Parent2 {}'.format(str(getattr(parent2.dof[0], "values")))
            # print 'Child1 {}'.format(str(getattr(child1.dof[0], "values")))
            # print 'Child2 {}\n'.format(str(getattr(child2.dof[0], "values")))
            # aims_object.generate_input(child1.sdf_string) #generates input
            # shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_child1_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            # aims_object.generate_input(child2.sdf_string) #generates input
            # shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_child2_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            # aims_object.generate_input(parent1.sdf_string) #generates input
            # shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_parent1_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            # aims_object.generate_input(parent2.sdf_string) #generates input
            # shutil.copy('geometry.in', os.path.join(os.getcwd(),'invalid','geometry_parent2_'+str(cnt)+'.in')) #copy invalid geometry to "invalid" folder
            #
            # sys.exit(0)
            print_output("The geometries created via crossover are invalid.")
            cnt += 1
            continue
    else:
        child1, child2 = Structure(parent1), Structure(parent2)
        print_output("No crossover was performed. Children are copies of "
                     "parents: " + str(child1) + (": ") + str(child1) +
                     (", ") + str(child2) + (": ") + str(child2))
        # Delete inherited attributes.
        for child in child1, child2:
            attr_list = ["initial_sdf_string", "energy"]
            for attr in attr_list:
                delattr(child, attr)
            for dof in child.dof:
                delattr(dof, "initial_values")

    run_util.str_info(child1)
    run_util.str_info(child2)

    try:
        mutate_and_relax(child1, "child1", iteration, cnt_max, **linked_params)
    except Exception as exc:
        print_output(exc)
        sys.exit(0)
    try:
        mutate_and_relax(child2, "child2", iteration, cnt_max, **linked_params)
    except Exception as exc:
        print_output(exc)
        sys.exit(0)
    population.sort()
    print_output("Sorted population: " + ', '.join([
        str(v) for v in population]))
    del population[-1]
    del population[-1]
    print_output("Sorted population after removing two structures with highest"
                 " energy: " + ', '.join([str(v) for v in population]))
    min_energy.append(population[0].energy)
    print_output("Lowest energy of the population: %.3f" % min_energy[-1])
    print_output("Lowest energies in run: "+str(min_energy))
    run_util.perform_backup(mol, population, blacklist, iteration, min_energy)
    run_util.check_for_convergence(iteration, params, min_energy)
    run_util.check_for_kill()
    iteration += 1
