#!/usr/bin/env python
import numpy as np
from numpy import array
import sys, os, time
from copy import deepcopy
from fafoom import *
import fafoom.run_utilities as run_util
import fafoom.utilities as util
from optparse import OptionParser

StartTime = time.time()
parser = OptionParser()
parser.add_option("-t", "--test", dest="test", default=None,
                  help="Testing mode will turn on np.random.seed(0)")
parser.add_option("-r", "--random", dest="random", default=None, help="Generating of random and unique structures")
parser.add_option("-p", "--prerun", dest="prerun", default=None, help="Updating of the template.sdf file")
parser.add_option("-c", "--clean", dest="clean", default=None, help="Clean the directory")

(options, args) = parser.parse_args()

# Cleaning up the directory if necessary
if options.clean is not None:
    cleanup(os.getcwd())
    sys.exit(0)

if options.test is not None:
    print('TESTMODE is ACTIVATED but not properly working!!!')
    np.random.seed(0)
np.set_printoptions(suppress=True)      # Correctly writes one-line blacklist
opt = run_util.simple_or_restart()      # Decide for restart or a simple run.
""" If genetic algorithm was invoked without additional inputs
FAFOOM will try to find parameters.txt file as default. """
if len(sys.argv) < 2:
    if os.path.exists(os.path.join(os.getcwd(), 'parameters.txt')):
        p_file = os.path.join(os.getcwd(), 'parameters.txt')
    else:
        raise Exception('Please produce parameters.txt file.')
else:
    p_file = sys.argv[1]
params = file2dict(p_file, ['GA settings', 'Run settings'])             # Take parameters from the sections
dict_default = {'energy_var': 0.001, 'selection': "roulette_wheel",     # Default parameters:
                'fitness_sum_limit': 1.2, 'popsize': 10,
                'prob_for_crossing': 1.0, 'max_iter': 30,
                'iter_limit_conv': 20, 'energy_diff_conv': 0.001}
params = set_default(params, dict_default)      # Set defaults for parameters not defined in the parameter file.
"""Create lists to store Population, minimal energies and structures that are already calculated."""
population, blacklist, min_energy, new_blacklist = [], [], [], []
Trials, NotValid, Known, Calculated, found_in_blacklist = 0, 0, 0, 0, 0
BLACKLIST, visited_folders = [], []
# =======================================================================
if opt == "simple":
    run_util.HeadFafoom()
    # Detect the desired application for energy evaluation.
    energy_function = run_util.detect_energy_function(params)
    # 'Calculated' is the number of Calculations performed and when it reaches params['max_iter'] algorithm stops.
    # Create mol object.
    mol = MoleculeDescription(p_file)
    # Assign the permanent attributes to the molecule.
    mol.get_parameters()
    mol.create_template_sdf()
    # Check for potential degree of freedom related parameters.
    linked_params = run_util.find_linked_params(mol, params)
    # Initialize prefered volume in which geometries will be inirially produced.
    # Print Head in the Fafoom output file.
    print_output('Atoms: {}, Bonds: {}'.format(mol.atoms, mol.bonds))
    print_output('\n___Initialization___\n')
    # Generate sensible and unique 3d structures.
    NumOfAtoms_sur, Periodic_sur, Path_sur = mol.analyze_constrained_geometry()
    flag, generation_Trials = 1.0, 0
    BLACKLIST, visited_folders = mol.UpdateBlacklist(blacklist=BLACKLIST, folders=visited_folders)

    trial = 0
    while len(population) < params['popsize'] and Calculated < params['max_iter']:
        trial += 1
        Structure.index = Calculated
        str3d = Structure(mol)
        str3d.generate_structure()
        ensemble = Ensemble(mol)
        lattice = ensemble.extract_lattice_vectors()
        ensemble.create_ensemble(mol)
        ensemble.write_to_separate_files()
        if not ensemble.clashes_in_ensemble(periodicity=lattice, attempt=trial):lem
            ensemble.write_to_aims(trial, lattice)
            population.append(ensemble)
        else:
            continue

            # ensemble.write_to_one_file()
            # ensemble.merge_and_write(ens)

                    # sys.exit(0)

            # sys.exit(0)

        # if mol.conformations == 'same':
        #     if str3d.is_geometry_valid(flag=flag):
        #         for i in range(1, mol.number_of_molecules + 1):
        #             STRUCTURES['structure_{}'.format(i)] = str3d
        #     else:
        #         continue

        # Trials += 1
        # """ In case if created structure is not sensible: """
        # if not str3d.is_geometry_valid(flag=flag):
        #     NotValid+=1
        #     generation_Trials += 1          # Count number of Trials (allowed 100)
        #     if generation_Trials == 10:     # After 100 Trials of failure to generate valid Structure
        #         if flag >= 0.755:           # the criteria of geometry validation (flag) is decreased:
        #             flag -= 0.005           # The lowest value of the flag is 0.80, if reached
        #             generation_Trials = 0   # it is counted as bad Trials and Calculated += 1.
        #         else:
        #             sys.exit(0)   # Terminates the code
        #     continue
        # else:
        #     BLACKLIST, visited_folders = mol.UpdateBlacklist(
        #         blacklist=BLACKLIST, folders=visited_folders)
        #     if str3d not in BLACKLIST:
        #         str3d.prepare_for_calculation(NumOfAtoms_sur, Periodic_sur, Path_sur)
        #         name = '{:04d}_structure'.format(Calculated+1)
        #         """ Perform the local optimization """
        #         run_util.optimize(str3d, energy_function, params, name)
        #         Calculated += 1
        #         if str3d not in BLACKLIST:
        #             str3d.send_to_blacklist(BLACKLIST)
        #             str3d.send_to_new_blacklist(new_blacklist) #  Locally
        #             # calculated structures
        #             population.append(str3d)
        #             print_output('Structure {}{:>15.4f}'.format(Calculated, float(str3d)))
        #             run_util.relax_info(str3d)
        #             population.sort()
        #             min_energy.append(float('{:.3f}'.format(population[0].energy)))
        #             run_util.perform_backup(mol, population, BLACKLIST, Calculated, min_energy, new_blacklist)
        #         else:
        #             print_output('Structure {}{:>15.4f}: Found in Blacklist'.format(Calculated, float(str3d)))
        #             run_util.relax_info(str3d)
        #             found_in_blacklist+=1
        #             continue   # The already known structure was obtained after optimization
        #     else:
        #         found_in_blacklist += 1
        #         Known += 1  # Geometry is fine, but already known.
        #         continue
    if Calculated == params['max_iter']:
        print_output("The allowed number of Trials for building the "
                     "population has been exceeded. The code terminates.")
        population.sort()
        for i in range(len(population)):
            print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
        print_output('Structures found: {}'.format(len(population)))
        run_util.AnalysisFafoom(Trials, NotValid, Calculated, Known, len(blacklist), run_util.TimeSpent(StartTime))
        run_util.perform_backup(mol, population, BLACKLIST, Calculated,
                                min_energy, new_blacklist)
        sys.exit(0)
    print_output("___Initialization completed___")
    population.sort()
    print_output("Initial population after sorting: ")
    for i in range(len(population)):
        print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
    print_output(' ')
    min_energy.append(float('{:.3f}'.format(population[0].energy)))
""" End of initialization process. Now the population is full.
Starting genetic algorithm: performing of selection, crossing over and mutation
operations for structures in population pool. """
# Flag for checking geometries should be valid for relaxed structures:
# Flag cannot be less than 0.80
#flag = adjusted_flag(population)
""" At least for now the flag for checking geometries is adjusted in the way
that all the relaxed geometries are also sensible geometries."""
# print_output('Adjusted flag for checking for clashes inside the structures is: {}'.format(flag))
#if Trials > 0:
    #run_util.AnalysisFafoom(Trials, NotValid, Calculated, Known, len(blacklist), run_util.TimeSpent(StartTime))
#Random_Trials, Random_NotValid, Random_Calculated, Random_Known, Random_Blacklist, Random_Time = Trials, NotValid, Calculated, Known, len(blacklist), run_util.TimeSpent(StartTime)


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
    NumOfAtoms_sur, Periodic_sur, Path_sur = mol.analyze_constrained_geometry()
    str3d = Structure(mol)
    linked_params = run_util.find_linked_params(mol, params)
    BLACKLIST, visited_folders = mol.UpdateBlacklist(
        blacklist=BLACKLIST, folders=visited_folders)

    with open("backup_min_energy.dat") as inf:
        for line in inf:
            min_energy.append(eval(line))
    """Check all the folders"""
    calculated = []
    for i in os.listdir(os.getcwd()):
        if '_structure' in i and os.path.isdir(i):
            calculated.append(int(i.split('_')[0]))
    with open("backup_new_blacklist.dat") as new:
        everything = new.read()                             # Split everything into separate structures:
        structures = everything.split('$$$$')[:-1]          # Correct number of structures
        for structure in structures:
            for lines in structure.splitlines():
                if 'Index = ' in lines:
                    ind = re.search('(Index = (\d+))', lines)
                    header = structure.splitlines().index(lines) -1
                if 'Energy = ' in lines:
                    en = re.search('(Energy = (.*?\d+\.\d+))', lines)
                    break
            Structure.index = int(ind.group(2))-1
            str3d = Structure(mol)
            str3d.energy = float(en.group(2))
            str3d.sdf_string = '\n'.join(structure.splitlines()[header:])
            str3d.initial_sdf_string = str3d.sdf_string
            for dof in str3d.dof:
                dof.update_values(str3d.sdf_string)
                setattr(dof, "initial_values", dof.values)
            str3d.send_to_blacklist(blacklist)
            str3d.send_to_new_blacklist(new_blacklist)

    population = run_util.extract_population(blacklist, params['popsize'])
    for i in range(len(population)):
        print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
    linked_params = run_util.find_linked_params(mol, params)
    Calculated = max(calculated)
    
    print_output(" \n ___Reinitialization completed___")
    # Remove dir with unfinished calculation
    fol = '{:04d}_structure'.format(Calculated)
    if not os.path.exists(os.path.join(fol, 'result.out')):
        remover_dir('{:04d}_structure'.format(Calculated))
        Calculated-=1
        Structure.index = Calculated
    else:
        Structure.index = Calculated+1
    """ If initialization is not finished it should be finished"""
    if len(new_blacklist) < params['popsize']:
        # Calculated = 0
        generation_Trials = 0
        volume = mol.volume
        while len(population) < params['popsize'] and Calculated < params['max_iter']:
            Structure.index = Calculated
            str3d = Structure(mol)
            str3d.generate_structure()
            if not str3d.is_geometry_valid(flag=flag):
                generation_Trials += 1  # Count number of Trials (allowed 100)
                if generation_Trials == 10:  # After 100 Trials of failure to generate valid Structure
                    if flag >= 0.755:  # the criteria of geometry validation (flag) is decreased:
                        flag -= 0.005  # The lowest value of the flag is 0.80, if reached
                        generation_Trials = 0  # it is counted as bad Trials and Calculated += 1.
                    else:
                        sys.exit(0)  # Terminates the code
                continue
            else:
                if str3d not in BLACKLIST:
                # if str3d not in blacklist and str3d not in shared_blacklist:
                    str3d.prepare_for_calculation(NumOfAtoms_sur, Periodic_sur, Path_sur)
                    name = '{:04d}_structure'.format(Calculated+1)
                    # Perform the local optimization
                    run_util.optimize(str3d, energy_function, params, name)
                    Calculated += 1
                    if run_util.check_for_not_converged(name):
                        continue
                    else:
                        str3d.send_to_blacklist(BLACKLIST)          # Blacklist
                        str3d.send_to_new_blacklist(new_blacklist)
                        population.append(str3d)
                        print_output('{:<15}{:>10.4f}'.format(str3d, float(str3d)))
                        run_util.relax_info(str3d)
                else:
                    Trials += 1   # Geomerty is fine, but already known.
            run_util.perform_backup(mol, population, BLACKLIST, Calculated, min_energy, new_blacklist)
        if Calculated == params['max_iter']:
            print_output("The allowed number of Trials for building the "
                         "population has been exceeded. The code terminates.")
            population.sort()
            for i in range(len(population)):
                print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
            sys.exit(0)
        print_output("___Initialization completed___")
        population.sort()
        print_output("Initial population after sorting: ")
        for i in range(len(population)):
            print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
        min_energy.append(float('{:.3f}'.format(population[0].energy)))
#print_output('CHECK FOR ME Calculated structures: {}'.format(Calculated))
""" Start the Genetic Operations routines """
BLACKLIST, visited_folders = mol.UpdateBlacklist(
    blacklist=BLACKLIST, folders=visited_folders)

print_output('Start the Genetic Algorithm part!\n')
while Calculated < params['max_iter']:
    (parent1, parent2, fitness) = selection(population, params['selection'],
                                            params['energy_var'],
                                            params['fitness_sum_limit'])

    param_crossover = np.random.rand()
    if param_crossover < params['prob_for_crossing']:
        # print('Perform crossover')
        after_crossover, after_mutation = [], []
        generation_Trials = 0
        child = Structure.crossover(parent1, parent2, method=mol.crossover_method)
        after_crossover = inter_info(child, after_crossover)
        child.mutate(**linked_params)
        after_mutation = inter_info(child, after_mutation)
        if child.is_geometry_valid(flag=flag):
            BLACKLIST, visited_folders = mol.UpdateBlacklist(
                blacklist=BLACKLIST, folders=visited_folders)
            if child not in BLACKLIST:
                child.prepare_for_calculation(NumOfAtoms_sur, Periodic_sur, Path_sur)
                child.index = Calculated+1
                name = '{:04d}_structure'.format(Calculated + 1)
                Structure.index = Calculated+1
                run_util.GeneticOperationsOutput(len(blacklist)+1, Calculated, parent1, parent2, after_crossover, after_mutation)
                run_util.optimize(child, energy_function, params, name)
                Trials+=1
                Calculated+=1
                if child not in BLACKLIST:
                    print_output('Child after relaxation: Added to Blacklist\n')
                    print_output('{:<15}{:>10.4f}'.format(child, float(child)))
                    run_util.relax_info(child)
                    print_output('------------------------------------------------------------\n')
                    child.send_to_blacklist(BLACKLIST)  # Blacklist
                    child.send_to_new_blacklist(new_blacklist)
                    population.append(child)
                    population.sort()
                    if len(population) > params['popsize']:
                        for i in range(len(population) - params['popsize']):
                            del population[-1]
                    min_energy.append(float('{:.3f}'.format(population[0].energy)))
                    print_output("Current population after sorting: ")
                    for i in range(len(population)):
                        print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
                    print_output("\nLowest energies in run: {}".format(min_energy))
                    run_util.perform_backup(mol, population, BLACKLIST, Calculated, min_energy, new_blacklist)
                    # shared_blacklist, visited_folders = run_util.update_shared_blacklist(shared_blacklist, visited_folders, child)
                    run_util.CheckForConvergence(Trials, NotValid, Known, len(blacklist), run_util.TimeSpent(StartTime), Calculated, params, min_energy)
                    run_util.check_for_kill()
                else:
                    found_in_blacklist+=1
                    #print_output('Child after relaxation: Found in Blacklist\n')
                    print_output('Structure {}{:>15.4f}: Found in Blacklist'.format(Calculated, float(child)))
                    run_util.relax_info(child)
                    continue
            else:
                found_in_blacklist += 1
                print '{} found in blacklist'.format(found_in_blacklist)
                Trials+=1
                Known+=1
                continue
        else:
            Trials+=1
            NotValid+=1
            generation_Trials += 1                          # if will be 10 in raw the decrease the flag
            if generation_Trials == 10:
                if flag >= 0.755:
                    flag -= 0.005
                    generation_Trials = 0
                else:
                    sys.exit(0)  # Terminates the code
    else:
        pass
else:
    print_output('------------------------------------------------------------\n')
    print_output('------------------------------------------------------------\n')
    print_output('------------------------------------------------------------\n')
    print_output('\nAllowed number of calculations has been exceed!\n')
    population.sort()
    for i in range(len(population)):
        print_output('{:<15}{:>10.4f}'.format(population[i], float(population[i])))
    print_output('\nTotal information:')
    run_util.AnalysisFafoom(Trials, NotValid, Calculated, Known, len(blacklist), run_util.TimeSpent(StartTime))
    run_util.perform_backup(mol, population, BLACKLIST, Calculated, min_energy, new_blacklist)
    run_util.Goodbye()
    sys.exit(0)
