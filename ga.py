#!/usr/bin/python
import numpy as np
from numpy import array
import sys
import os
from fafoom import *
import fafoom.run_utilities as run_util
from fafoom.utilities import sdf2xyz, check_for_clashes, sdf2coords_and_atomtypes
from optparse import OptionParser
from fafoom.connectivity import *
parser = OptionParser()
parser.add_option("-t", "--test", dest="test", default = None, help="Testing mode will turn on np.random.seed(0) and random number will be predictable and the same. For testing purposes.")
parser.add_option("-r", "--random", dest="random", default = None, help="Generating of random and unique structures")
(options, args) = parser.parse_args()

if options.test is not None:
    np.random.seed(0)

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
if opt == "simple":
    run_util.HeadFafoom()
    # Detect the desired application for energy evaluation.
    energy_function = run_util.detect_energy_function(params)
    # cnt is the number of trials and when it reaches cnt_max algorithm stops.
    cnt = 0
    # Iteration is the maximum number of successful calculations.
    # Example: iteration = 10 means that you have 'population size' + iteration
    # calculated structures.
    iteration = 1
    # Create mol object.
    mol = MoleculeDescription(p_file)
    # Assign the permanent attributes to the molecule.
    mol.get_parameters()
    mol.create_template_sdf()
    # Check for potential degree of freedom related parameters.
    linked_params = run_util.find_linked_params(mol, params)
    # Initialize prefered volume in which geometries will be inirially produced.
    volume = mol.volume
    # Print Head in the Fafoom output file.

    print_output('Atoms: {}, Bonds: {}'.format(mol.atoms, mol.bonds))
    print_output('\n___Initialization___\n')
    # Generate sensible and unique 3d structures.
    surrounding_file = os.path.join(os.getcwd(),mol.constrained_geometry_file)
    flag = 1.0
    generation_trials = 0
    while len(population) < params['popsize'] and cnt < cnt_max:
        Structure.index = len(population)
        str3d = Structure(mol)
        str3d.generate_structure()
        if not str3d.is_geometry_valid(flag = flag):
            generation_trials += 1
            # After 100 trials of failure to generate valid Structure
            # we decrease the criterion of validation (flag):
            if generation_trials == 100:
                if flag >= 0.805:
                    flag -= 0.005
                    generation_trials = 0
                # The lowest value of the flag is 0.75, if reached
                # it is counted as bad trial.
                else:
                    cnt += 1
            continue
        else:
            # If sensible structure is unique:
            if str3d not in blacklist:
                if len(aims2xyz(surrounding_file)) < 1:
                    str3d.put_to_origin()
                # for dof in str3d.dof:
                    # if dof.name == 'Protomeric':
                    #     print('{}: {}'.format(dof.name, [float('{}'.format(x)) for x in dof.values]))
                    # if dof.name == 'Torsion':
                    #     print('{}: {}'.format(dof.name, [float('{:.2f}'.format(x)) for x in dof.values]))
                if not check_for_clashes(str3d.sdf_string, surrounding_file):
                    if 'centroid' not in mol.dof_names and len(aims2xyz(surrounding_file)) > 1:
                        str3d.adjust_position()
                    else:
                        if cnt==cnt_max-1:
                            print('Increase the volume!')
                            print_output('Probably, you should increase the volume.')
                        cnt+=1
                        continue
                if 'centroid' not in mol.dof_names:
                    if not str3d.check_position(volume) and len(aims2xyz(surrounding_file)) > 1:
                        str3d.adjust_position()
                else:
                    if 0 < len(aims2xyz(surrounding_file)) < 3:
                        if check_geo_if_not_too_far(str3d.sdf_string, surrounding_file, flag=1.5) == False:
                            str3d.adjust_position_centroid(surrounding_file)
                name = 'structure_{}'.format(str3d.index)
                # Perform the local optimization
                # print str3d.sdf_string
                run_util.optimize(str3d, energy_function, params, name)
                if run_util.check_for_not_converged(name):
                    continue
                else:
                    str3d.send_to_blacklist(blacklist) #Blacklist
                    str3d.send_to_new_blacklist(new_blacklist)
                    population.append(str3d)
                    print_output('{}\nEnergy: {}'.format(str3d, float(str3d)))
                    run_util.relax_info(str3d)
                    population.sort()
                    min_energy.append(population[0].energy)
                    run_util.perform_backup(mol, population, blacklist, iteration, min_energy, new_blacklist)
            else:
                # Geomerty is fine, but already known.
                cnt += 1
        # run_util.perform_backup(mol, population, blacklist, iteration, min_energy, new_blacklist)
    if cnt == cnt_max:
        print_output("The allowed number of trials for building the "
                     "population has been exceeded. The code terminates.")
        sys.exit(0)
    print_output("___Initialization completed___")
    population.sort()
    print_output("Initial population after sorting: ")
    for i in range(len(population)):
        print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
    min_energy.append(population[0].energy)

""" End of initialization process. Now the population is full.
Starting genetic algorithm: performing of selection, crossing over and mutation
operations for structures in population pool. """
# Flag for checking geometries should be valid for relaxed structures:
# Flag cannot be less than 0.75
flag = adjusted_flag(population)

""" At least for now the flag for checking geometries is adjusted in the way
that all the relaxed geometries are also sensible geometries."""
print_output('Adjusted flag for checking for clashes inside the structures is: {}'.format(flag))

if opt == "restart":
    flag = 1.0
    # Detect the desired application for energy evaluation.
    energy_function = run_util.detect_energy_function(params)
    # Reconstruct the molecule, population, blacklist and the state of the run.
    print_output(" \n ___Restart will be performed___")
    mol = MoleculeDescription(p_file)
    # Assign the permanent attributes to the molecule.
    mol.get_parameters()
    mol.create_template_sdf()
    surrounding_file = os.path.join(os.getcwd(),mol.constrained_geometry_file)
    with open("backup_new_blacklist.dat", 'r') as new:
        #Split everything into structures:
        everything = new.read()
        structures = everything.split('$$$$')[:-1] #correct number of structures
        for structure in structures:
            for lines in structure.splitlines():
                if 'Index = ' in lines:
                    ind = re.search('(Index = (\d+))', lines)
                    header = structure.splitlines().index(lines) -1
                if 'Energy = ' in lines:
                    en = re.search('(Energy = (.*?\d+\.\d+))', lines)
                    break
            Structure.index = int(ind.group(2)) - 1
            str3d = Structure(mol)
            str3d.energy = float(en.group(2))
            str3d.sdf_string = '\n'.join(structure.splitlines()[header:])
            str3d.initial_sdf_string = str3d.sdf_string
            for dof in str3d.dof:
                dof.update_values(str3d.sdf_string)
                setattr(dof, "initial_values", dof.values)
            str3d.send_to_new_blacklist(new_blacklist)
            str3d.send_to_blacklist(blacklist)

    # with open("backup_blacklist.dat", 'r') as inf:
        # for line in inf:
            # blacklist.append(eval(line))
    with open("backup_min_energy.dat", 'r') as inf:
        for line in inf:
            min_energy.append(eval(line))
    with open("backup_iteration.dat", 'r') as inf:
        iteration_tmp = eval(inf.readline())
    linked_params = run_util.find_linked_params(mol, params)
    temp_dic = {}
    for i in range(len(blacklist)):
        temp_dic[blacklist[i].index] = blacklist[i].energy
    temp_sorted = sorted(temp_dic.items(), key=lambda t: t[1])
    for i in range(min(len(blacklist), params['popsize'])):
        population.append(blacklist[temp_sorted[i][0]-1])
    for i in range(len(population)):
        print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
    # print_output("Blacklist: " + ', '.join([str(v) for v in blacklist]))
    if len(new_blacklist) > params['popsize']:
        iteration = len(new_blacklist) + 1
    elif len(population) <= params['popsize']:
        iteration = len(population) + 1
    linked_params = run_util.find_linked_params(mol, params)
    Structure.index = len(blacklist)
    flag = adjusted_flag(blacklist)
    print_output('Adjusted flag for checking for clashes inside the structures is: {}'.format(flag))
    # Need to be adjusted, because, we want to calculate at least one structure.
    print_output(" \n ___Reinitialization completed___")
    remover_dir('structure_{}'.format(len(blacklist) + 1))
    # remover_dir('structure_{}'.format(len(blacklist) + 2))

    """ If initialization is not finished it should be finished"""

    if len(new_blacklist) < params['popsize']:
        cnt = 0
        generation_trials = 0
        volume = mol.volume
        while len(population) < params['popsize'] and cnt < cnt_max:
            Structure.index = len(population)
            str3d = Structure(mol)
            str3d.generate_structure()
            if not str3d.is_geometry_valid(flag = flag):
                generation_trials += 1
                # After 100 trials of failure to generate valid Structure
                # we decrease the criterion of validation (flag):
                if generation_trials == 100:
                    if flag >= 0.805:
                        flag -= 0.005
                        generation_trials = 0
                    # The lowest value of the flag is 0.75, if reached
                    # it is counted as basd trial.
                    else:
                        cnt += 1
                continue
            else:
                # If sensible structure is unique:
                if str3d not in blacklist:
                    if len(aims2xyz(surrounding_file)) < 1:
                        str3d.put_to_origin()
                    for dof in str3d.dof:
                        if dof.name == 'Torsion':
                            print('{}: {}'.format(dof.name, [float('{:.2f}'.format(x)) for x in dof.values]))

                    if not check_for_clashes(str3d.sdf_string, surrounding_file):
                        if 'centroid' not in mol.dof_names and len(aims2xyz(surrounding_file)) > 1:
                            str3d.adjust_position()
                        else:
                            if cnt==cnt_max-1:
                                print('Increase the volume!')
                                print_output('Probably, you should increase the volume.')
                            cnt+=1
                            continue
                    if 'centroid' not in mol.dof_names:
                        if not str3d.check_position(volume) and len(aims2xyz(surrounding_file)) > 1:
                            str3d.adjust_position()
                    else:
                        if 0 < len(aims2xyz(surrounding_file)) < 3:
                            if check_geo_if_not_too_far(str3d.sdf_string, surrounding_file, flag=1.5) == False:
                                str3d.adjust_position_centroid(surrounding_file)
                    name = 'structure_{}'.format(str3d.index)
                    # Perform the local optimization
                    run_util.optimize(str3d, energy_function, params, name)
                    if run_util.check_for_not_converged(name):
                        continue
                    else:
                        str3d.send_to_blacklist(blacklist) #Blacklist
                        str3d.send_to_new_blacklist(new_blacklist)
                        population.append(str3d)
                        print_output('{}\nEnergy: {}'.format(str3d, float(str3d)))
                        run_util.relax_info(str3d)
                else:
                    # Geomerty is fine, but already known.
                    cnt += 1
            run_util.perform_backup(mol, population, blacklist, iteration, min_energy, new_blacklist)
        if cnt == cnt_max:
            print_output("The allowed number of trials for building the "
                         "population has been exceeded. The code terminates.")
            sys.exit(0)
        print_output("___Initialization completed___")
        population.sort()
        print_output("Initial population after sorting: ")
        for i in range(len(population)):
            print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
        min_energy.append(population[0].energy)


def mutate_and_relax(candidate, name, iteration, cnt_max, **kwargs):
    # print_output('__{}__'.format(name))
    found = False
    cnt = 0
    while found is False and cnt < cnt_max:
        Structure.index = len(blacklist)
        candidate_backup = Structure(candidate)
        if candidate in blacklist:
            # if len(aims2xyz(surrounding_file)) < 1:
            #     candidate.put_to_origin()
            # print_output('Candidate in blacklist')
            # print_output('Perform hard_mutate')
            candidate.hard_mutate(**kwargs) #Mutate, since already in blacklist
            if not candidate.is_geometry_valid(flag = flag): #Check geometry after mutation
                # print_output('Geometry is not valid')
                candidate = candidate_backup #Reset structure
                cnt+=1
                continue
            else:
                if len(aims2xyz(surrounding_file)) < 1:
                    candidate.put_to_origin()
                if not check_for_clashes(candidate.sdf_string, os.path.join(mol.constrained_geometry_file)):
                    #print_output('Clash found')
                    if 'centroid' not in mol.dof_names: #If optimization for the COM is turned off
                        candidate.adjust_position() #Adjust position in z direction
                    else:
                        # print_output('Centroid found -- skipp')
                        candidate = candidate_backup #Clash found so structure will be resetted
                        cnt+=1
                        continue
                else:
                    if 0 < len(aims2xyz(surrounding_file)) < 3:
                        if check_geo_if_not_too_far(candidate.sdf_string, surrounding_file, flag=1.5) == False:
                            candidate.adjust_position_centroid(surrounding_file)
                candidate.index = len(blacklist) + 1
                name = 'structure_{}'.format(candidate.index)
                print_output('Values for {} child after mutation'.format(candidate))
                run_util.str_info(candidate)
                print_output('------------------------------------------------------------\n')
                name = 'structure_{}'.format(candidate.index)
                run_util.optimize(candidate, energy_function, params, name)
                if run_util.check_for_not_converged(name):
                    continue
                # run_util.check_for_kill()
                else:
                    candidate.send_to_blacklist(blacklist) #Blacklist
                    candidate.send_to_new_blacklist(new_blacklist) #Blacklist
                    print_output('{}\nEnergy: {}'.format(candidate, float(candidate)))
                    # print_output(str(candidate)+": energy: "+str(float(candidate))+", is temporary added to the population")
                    run_util.relax_info(candidate)
                    found = True
                    population.append(candidate)
        elif candidate not in blacklist:
            # print_output('Candidate not in blacklist')
            candidate.mutate(**kwargs) #Mutatte with some probability
            if not candidate.is_geometry_valid(flag = flag):
                # print_output('Geometry is not fine')
                candidate = candidate_backup # Rebuild the structure
                cnt += 1
                continue
            else:
                if len(aims2xyz(surrounding_file)) < 1:
                    candidate.put_to_origin()
                if not check_for_clashes(candidate.sdf_string, os.path.join(mol.constrained_geometry_file)):
                    #print_output('Clash found')
                    if 'centroid' not in mol.dof_names:
                        # print_output('Perform adjust')
                        candidate.adjust_position()
                    else:
                        # print_output('Perform hard_mutate')
                        candidate.hard_mutate(**kwargs)
                        if not check_for_clashes(candidate.sdf_string, os.path.join(mol.constrained_geometry_file)):
                            candidate = candidate_backup
                            cnt+=1
                            continue
                        else:
                            if 0 < len(aims2xyz(surrounding_file)) < 3:
                                if check_geo_if_not_too_far(candidate.sdf_string, surrounding_file, flag=1.5) == False:
                                    candidate.adjust_position_centroid(surrounding_file)

                else:
                    if  0 < len(aims2xyz(surrounding_file)) < 3:
                        if check_geo_if_not_too_far(candidate.sdf_string, surrounding_file, flag=1.5) == False:
                            candidate.adjust_position_centroid(surrounding_file)
                candidate.index = len(blacklist) + 1
                name = 'structure_{}'.format(candidate.index)
                print_output('Values for {} child after mutation'.format(candidate))
                run_util.str_info(candidate)
                print_output('------------------------------------------------------------\n')
                run_util.optimize(candidate, energy_function, params, name)
                if run_util.check_for_not_converged(name):
                    continue
                else:
                    run_util.check_for_kill()
                    candidate.send_to_blacklist(blacklist) #Blacklist
                    candidate.send_to_new_blacklist(new_blacklist) #Blacklist
                    print_output('{}\nEnergy: {}'.format(candidate, float(candidate)))
                    # print_output(str(candidate)+": energy: "+str(float(candidate))+", is temporary added to the population")
                    run_util.relax_info(candidate)
                    found = True
                    population.append(candidate)

        if cnt == cnt_max:
            raise Exception("The allowed number of trials for generating a unique child has been exceeded.")
iteration = len(new_blacklist) + 1
while iteration < params['max_iter']:
    print_output(' \n ___Start of iteration {}___'.format(iteration))
    (parent1, parent2, fitness) = selection(population, params['selection'],
                                            params['energy_var'],
                                            params['fitness_sum_limit'])
    param = np.random.rand()
    # print_output('Try to crossover.')
    cnt = 0
    while param < params['prob_for_crossing'] and cnt < cnt_max:
        generation_trials = 0
        child1, child2 = Structure.crossover(parent1, parent2, method = 'random_points')
        for child in child1, child2:
            if child.is_geometry_valid(flag = flag):
                if len(aims2xyz(surrounding_file)) < 1:
                    child.put_to_origin()
                for dof in child.dof:
                    if dof.name == 'Torsion':
                        print('{}: {}'.format(dof.name, [float('{:.2f}'.format(x)) for x in dof.values]))
                # run_util.str_info(child)
                if not check_for_clashes(child.sdf_string, surrounding_file):
                    print('Clask found')
                    child.adjust_position_centroid(surrounding_file)
                break

            else:
                generation_trials += 1
                if generation_trials == 10:
                    if flag >= 0.805:
                        flag -= 0.005
                        generation_trials = 0
                Structure.index = len(blacklist)
                cnt += 1
                continue
        break
    else:
        print('Cannot find good crossover')
        child = Structure(parent1)
        # print_output('No crossover was performed. Children are copies of parents.')
        # Delete inherited attributes.
        attr_list = ["initial_sdf_string", "energy"]
        for attr in attr_list:
            delattr(child, attr)
        for dof in child.dof:
            delattr(dof, "initial_values")

    print_output('------------------------------------------------------------')
    print_output('Values for {} parent_1'.format(parent1))
    run_util.str_info(parent1)
    print_output('\n')
    print_output('Values for {} parent_2'.format(parent2))
    run_util.str_info(parent2)
    print_output('\n')
    print_output('Values for {} child after crossover'.format(child))
    run_util.str_info(child)
    try:
        mutate_and_relax(child, "child", iteration, cnt_max, **linked_params)
    except Exception as exc:
        print_output(exc)
        sys.exit(0)
    population.sort()
    # print_output("Sorted population: " + ', '.join([str(v) for v in population]))
    if len(population) > params['popsize']:
        for i in range(len(population) - params['popsize']):
            del population[-1]
    # print_output("Sorted population after removing two structures with highest"
                #  " energy: " + ', '.join([str(v) for v in population]))
    min_energy.append(population[0].energy)
    print_output("Current population after sorting: ")
    for i in range(len(population)):
        print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
    print_output("Lowest energies in run: {}".format(min_energy))
    run_util.perform_backup(mol, population, blacklist, iteration, min_energy, new_blacklist)
    run_util.check_for_convergence(iteration, params, min_energy)
    run_util.check_for_kill()
    iteration += 1
