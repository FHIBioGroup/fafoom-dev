#!/usr/bin/python
import numpy as np
from numpy import array
import sys
import os
from fafoom import *
import fafoom.run_utilities as run_util
from fafoom.utilities import sdf2xyz, check_for_clashes
from fafoom.measure import centroid_measure, MolecularAngleBetween, MolecularDihedralMeasure
#Need to correctly write the one-line blacklist:
np.set_printoptions(suppress=True)
# Decide for restart or a simple run.
opt = run_util.simple_or_restart()
p_file = sys.argv[1]
if sys.argv is None:
    if os.path.exists(os.path.join(os.getcwd(), 'parameters.txt')):
        p_file = os.path.join(os.getcwd(), 'parameters.txt')
    else:
        pass
        #Assign default parameters for calculation
# Build a dictionary from two section of the parameter file.
params = file2dict(p_file, ['GA settings', 'Run settings'])

dict_default = {'energy_var': 0.001, 'selection': "roulette_wheel",
                'fitness_sum_limit': 1.2, 'popsize': 10,
                'prob_for_crossing': 1.0, 'max_iter': 30,
                'iter_limit_conv': 20, 'energy_diff_conv': 0.001}
# Set defaults for parameters not defined in the parameter file.
params = set_default(params, dict_default)
energy_function = run_util.detect_energy_function(params)
cnt_max = 2500
population, blacklist, min_energy = [], [], []
#=======================================================================

def DeleteFileifExists(path_to_file):
    if os.path.exists(path_to_file):
        os.remove(path_to_file)

def CreateXYZwithStructures(structures):
    with open(os.path.join(os.getcwd(), 'structures.xyz'), 'a') as xyz:
        xyz.write('{}\n'.format(sum([len(sdf2xyz(i.sdf_string)) for i in structures])))
        xyz.write('{}\n'.format('Comment'))
        for structure in structures:
            for i in sdf2xyz(structure.sdf_string):
                xyz.write('{}\n'.format('  '.join([str(k) for k in i])))

def CheckClashesBetweenTwoMolecules(structure_1, structure_2):
    check = True
    molecule_1 = sdf2xyz_list(structure_1.sdf_string)
    molecule_2 = sdf2xyz_list(structure_2.sdf_string)
    for x in molecule_1:
        for y in molecule_2:
            if np.linalg.norm(x[1:]-y[1:]) < x[0] or np.linalg.norm(x[1:]-y[1:]) < y[0]:
                check = False
    return check

def CheckforClashesBetweenMolecules(structures):
    check = False
    k = 0
    for structure in structures:
        k+=1
        for m in range(k, len(structures)):
            if not CheckClashesBetweenTwoMolecules(structure, structures[m]):
                check = True
                print('Clash between structure {} and structure {}'.format(structures.index(structure)+1, m+1))
    return check


def ConnectivityMatrix(structures):
    def CheckConnectivity(structure_1, structure_2):
        connectivity = 0
        molecule_1 = sdf2xyz_list(structure_1.sdf_string)
        molecule_2 = sdf2xyz_list(structure_2.sdf_string)
        for x in molecule_1:
            for y in molecule_2:
                if np.linalg.norm(x[1:]-y[1:]) < x[0]*1.5 or np.linalg.norm(x[1:]-y[1:]) < y[0]*1.5:
                    connectivity = 1
        return connectivity
    Connections = np.array([CheckConnectivity(structures[i], structures[j]) if i!=j
                                                                            else 0 for i in range(len(structures))
                                                                            for j in range(len(structures))])
    return Connections.reshape(len(structures), len(structures))

def GenerateConnectivityMatrix(structures):
    def CheckSumminRaw(Connections):
        check = False
        for i in Connections:
            if sum(i) < 1:
                check = True
        return check
    def GenerateConnectivity(structures):
        conn = np.array([np.random.randint(0,2) if i!=j and j>i
                                                else 0 for i in range(len(structures))
                                                for j in range(len(structures))])
        conn = conn.reshape(len(structures), len(structures))
        for i in range(len(conn)):
            for j in range(len(conn)):
                conn[j][i] = conn[i][j]
        return conn
    Connections = GenerateConnectivity(structures)
    while CheckSumminRaw(Connections):
        Connections = GenerateConnectivity(structures)
    return Connections

def CreateMolecularZMatrix(structures):
    Distances = np.array([np.linalg.norm(centroid_measure(structures[i].sdf_string) -
                                centroid_measure(structures[j].sdf_string))
                                for i in range(len(structures))
                                for j in range(len(structures))])
    Distances.reshape(len(structures), len(structures))
    return Distances

def GenerateStructures(structures):
    for structure in structures:
        structure.generate_structure()
        while not structure.is_geometry_valid():
            structure.generate_structure()

def GenerateSupramoleculeFromConnectivityMatrix(structures, Connectivity):
    for i in range(len(structures)):
        for dof in structures[0].dof:
            if dof.name == 'Centroid':
                dof.get_random_values()
                new_string = structures[0].sdf_string
                new_string = dof.apply_on_string(new_string, values_to_set=np.array([0,0,0]))
                structures[0].sdf_string = new_string
            for dof in structures[0].dof:
                dof.update_values(structures[0].sdf_string)

    print Connectivity[0]
    # popsize = 100
    # Connectivities = []
    # population = []
    # while len(population) != popsize:
    #     for structure in structures:
    #         for dof in structure.dof:
    #             if dof.name == 'Centroid':
    #                 dof.get_random_values()
    #                 new_string = structure.sdf_string
    #                 new_string = dof.apply_on_string(new_string)
    #                 structure.sdf_string = new_string
    #             for dof in structure.dof:
    #                 dof.update_values(structure.sdf_string)
    #     while CheckforClashesBetweenMolecules(structures):
    #         for structure in structures:
    #             for dof in structure.dof:
    #                 if dof.name == 'Centroid':
    #                     dof.get_random_values()
    #                     new_string = structure.sdf_string
    #                     new_string = dof.apply_on_string(new_string)
    #                     structure.sdf_string = new_string
    #                 for dof in structure.dof:
    #                     dof.update_values(structure.sdf_string)
    #     if sum([sum(i) for i in ConnectivityMatrix(structures)])> 0:
    #         population.append(structures)
    #     print len(population)
    #     Connectivities.append(ConnectivityMatrix(structures))
    #     # for structure in structures:
    #     #     print centroid_measure(structure.sdf_string)
    # for matrix in Connectivities:
    #     if matrix.all() == Connectivity.all():
    #         print Connectivity
    #         print matrix
    #         print 'Good News'
    #     else:
    #         print 'Not'
    #     CreateXYZwithStructures(structures)

    # print Connectivity
    # for i in range(len(structures)):
    #     for j in range(len(structures)):
    #         if Connectivity[i][j] == 1 and j>i:
    #             print('{} and {} should be connected'.format(i, j))

        # print centroid_measure(structure.sdf_string)


''' First implementation will be random search '''
if opt == "simple":
    # Assign the permanent attributes to the molecule:
    mol = MoleculeDescription(p_file)
    mol.get_parameters()
    mol.create_template_sdf()
    # Check for potential degree of freedom related parameters.
    linked_params = run_util.find_linked_params(mol, params)
    volume = mol.volume
    print_output('Atoms: {}, Bonds: {}'.format(mol.atoms, mol.bonds))
    print_output('\n___Initialization___\n')
    cnt = 0
    '''Start of moleclar structures creation'''
    molecules = [MoleculeDescription(p_file) for i in range(mol.number_of_molecules)]
    for molecule in molecules:
        molecule.get_parameters()
        molecule.create_template_sdf()

    structures = [Structure(molecule) for molecule in molecules]
    GenerateStructures(structures)
    # while CheckforClashesBetweenMolecules(structures):
    #     GenerateStructures(structures)
    #     print('It is not ok')
    # else:
    #     print('IT is ok')
    Connectivity = GenerateConnectivityMatrix(structures)
    print Connectivity
    GenerateSupramoleculeFromConnectivityMatrix(structures, Connectivity)
    # CheckforClashesBetweenMolecules(structures)
    DeleteFileifExists(os.path.join(os.getcwd(), 'structures.xyz'))
    CreateXYZwithStructures(structures)






    sys.exit(0)
    # Generate sensible and unique 3d structures.
    while len(population) < params['popsize'] and cnt < cnt_max:
        # print_output("New trial")
        Structure.index = len(population)
        str3d = Structure(mol)
        str3d.generate_structure()
        if not str3d.is_geometry_valid():
            cnt += 1
            continue
        else:
            if str3d not in blacklist:
                if not check_for_clashes(str3d.sdf_string, os.path.join(os.getcwd(), mol.constrained_geometry_file)):
                    if 'centroid' not in mol.dof_names:
                        str3d.adjust_position()
                    else:
                        if cnt==cnt_max-1:
                            print('Increase the volume!')
                            print_output('Probably, you should increase the volume.')
                        cnt+=1
                        continue
                if 'centroid' not in mol.dof_names:
                    if not str3d.check_position(volume):
                        str3d.adjust_position()
                else:
                    if len(aims2xyz(os.path.join(os.getcwd(), mol.constrained_geometry_file))) < 3:
                        if check_geo_if_not_too_far(str3d.sdf_string, os.path.join(os.getcwd(), mol.constrained_geometry_file), flag=1.5) == False:
                            str3d.adjust_position_centroid(os.path.join(os.getcwd(), mol.constrained_geometry_file))
                name = 'structure_{}'.format(str3d.index)
                # Perform the local optimization
                run_util.optimize(str3d, energy_function, params, name)
                print 'Optimization started'
                if run_util.check_for_not_converged(name):
                    str3d.send_to_blacklist(blacklist)
                    continue
                else:
                # run_util.check_for_kill()
                    str3d.send_to_blacklist(blacklist) #Blacklist
                    population.append(str3d)
                    print_output('{}\nEnergy: {}'.format(str3d, float(str3d)))
                    run_util.relax_info(str3d)
                # cnt += 1
            else:
                #print_output(blacklist) #Blacklist
                print_output("Geomerty of "+str(str3d)+" is fine, but already known.")
                cnt += 1
    if cnt == cnt_max:
        print_output("The allowed number of trials for building the "
                     "population has been exceeded. The code terminates.")
        sys.exit(0)
    print_output("___Initialization completed___")
    population.sort()
    print_output("Initial population after sorting: ")
    for i in range(len(population)):
        print_output('{:<}   {:>}'.format(population[i], float(population[i])))
    min_energy.append(population[0].energy)
    #print_output("Blacklist: " + ', '.join([str(v) for v in blacklist])) #Blacklist
    iteration = 0


if opt == "restart":
    # Reconstruct the molecule, population, blacklist and the state of the run.
    print_output(" \n ___Restart will be performed___")
    mol = MoleculeDescription(p_file)
    # Assign the permanent attributes to the molecule.
    mol.get_parameters()
    mol.create_template_sdf()
    # with open("backup_mol.dat", 'r') as inf:
    #     mol = eval(inf.readline())

    with open("backup_blacklist.dat", 'r') as inf:
        for line in inf:
            blacklist.append(eval(line))
    with open("backup_min_energy.dat", 'r') as inf:
        for line in inf:
            min_energy.append(eval(line))
    with open("backup_iteration.dat", 'r') as inf:
        iteration_tmp = eval(inf.readline())
    # with open("backup_population.dat", 'r') as inf:
    #     for line in inf:
    #         population.append(eval(line))
    linked_params = run_util.find_linked_params(mol, params)
    temp_dic = {}
    for i in range(len(blacklist)):
        temp_dic[blacklist[i].index] = blacklist[i].energy
    temp_sorted = sorted(temp_dic.items(), key=lambda t: t[1])
        # print('index in blacklist : {}'.format(blacklist[i].energy))
        # population.append(blacklist[i])
    # population.sort()
    if len(blacklist) > params['popsize']:
        for i in range(params['popsize']):
            population.append(blacklist[temp_sorted[i][0]-1])
    else:
        for i in range(len(blacklist)):
            population.append(blacklist[temp_sorted[i][0]-1])
    for i in range(len(population)):
        print_output(str(population[i])+" "+str(float(population[i])))
    print_output("Blacklist: " + ', '.join([str(v) for v in blacklist]))
    iteration = iteration_tmp+1
    linked_params = run_util.find_linked_params(mol, params)
    Structure.index = len(blacklist)
    print_output(" \n ___Reinitialization completed___")
    remover_dir('structure_{}'.format(len(blacklist) + 1))
    remover_dir('structure_{}'.format(len(blacklist) + 2))
    # remover_dir('generation_'+str(iteration)+'_child1')
    # remover_dir('generation_'+str(iteration)+'_child2')


def mutate_and_relax(candidate, name, iteration, cnt_max, **kwargs):
    print_output('__{}__'.format(name))
    found = False
    cnt = 0
    while found is False and cnt < cnt_max:
        Structure.index = len(blacklist)
        candidate_backup = Structure(candidate)
        if candidate in blacklist:
            print_output('Candidate in blacklist')
            print_output('Perform hard_mutate')
            candidate.hard_mutate(**kwargs) #Mutate, since already in blacklist
            if not candidate.is_geometry_valid(): #Check geometry after mutation
                print_output('Geometry is not valid')
                candidate = candidate_backup #Reset structure
                cnt+=1
                continue
            else:
                if not check_for_clashes(candidate.sdf_string, os.path.join(mol.constrained_geometry_file)):
                    #print_output('Clash found')
                    if 'centroid' not in mol.dof_names: #If optimization for the COM is turned off
                        candidate.adjust_position() #Adjust position in z direction
                    else:
                        print_output('Centroid found -- skipp')
                        candidate = candidate_backup #Clash found so structure will be resetted
                        cnt+=1
                        continue
                else:
                    if len(aims2xyz(os.path.join(os.getcwd(), mol.constrained_geometry_file))) < 3:
                        if check_geo_if_not_too_far(candidate.sdf_string, os.path.join(os.getcwd(), mol.constrained_geometry_file), flag=1.5) == False:
                            candidate.adjust_position_centroid(os.path.join(os.getcwd(), mol.constrained_geometry_file))

                name = 'structure_{}'.format(candidate.index)
                run_util.optimize(candidate, energy_function, params, name)
                run_util.check_for_kill()
                candidate.send_to_blacklist(blacklist) #Blacklist
                print_output('{}\nEnergy: {}'.format(candidate, float(candidate)))
                # print_output(str(candidate)+": energy: "+str(float(candidate))+", is temporary added to the population")
                run_util.relax_info(candidate)
                found = True
                population.append(candidate)
        elif candidate not in blacklist:
            print_output('Candidate not in blacklist')
            candidate.mutate(**kwargs) #Mutatte with some probability
            if not candidate.is_geometry_valid():
                print_output('Geometry is not fine')
                candidate = candidate_backup # Rebuild the structure
                cnt += 1
                continue
            else:
                if not check_for_clashes(candidate.sdf_string, os.path.join(mol.constrained_geometry_file)):
                    #print_output('Clash found')
                    if 'centroid' not in mol.dof_names:
                        print_output('Perform adjust')
                        candidate.adjust_position()
                    else:
                        print_output('Perform hard_mutate')
                        candidate.hard_mutate(**kwargs)
                        if not check_for_clashes(candidate.sdf_string, os.path.join(mol.constrained_geometry_file)):
                            candidate = candidate_backup
                            cnt+=1
                            continue
                        else:
                            if len(aims2xyz(os.path.join(os.getcwd(), mol.constrained_geometry_file))) < 3:
                                if check_geo_if_not_too_far(candidate.sdf_string, os.path.join(os.getcwd(), mol.constrained_geometry_file), flag=1.5) == False:
                                    candidate.adjust_position_centroid(os.path.join(os.getcwd(), mol.constrained_geometry_file))

                else:
                    if len(aims2xyz(os.path.join(os.getcwd(), mol.constrained_geometry_file))) < 3:
                        if check_geo_if_not_too_far(candidate.sdf_string, os.path.join(os.getcwd(), mol.constrained_geometry_file), flag=1.5) == False:
                            candidate.adjust_position_centroid(os.path.join(os.getcwd(), mol.constrained_geometry_file))

                name = 'structure_{}'.format(candidate.index)
                run_util.optimize(candidate, energy_function, params, name)
                run_util.check_for_kill()
                candidate.send_to_blacklist(blacklist) #Blacklist
                print_output('{}\nEnergy: {}'.format(candidate, float(candidate)))
                # print_output(str(candidate)+": energy: "+str(float(candidate))+", is temporary added to the population")
                run_util.relax_info(candidate)
                found = True
                population.append(candidate)
        if cnt == cnt_max:
            raise Exception("The allowed number of trials for generating a unique child has been exceeded.")

while iteration < params['max_iter']:
    print_output(' \n ___Start of iteration {}___'.format(iteration))
    (parent1, parent2, fitness) = selection(population, params['selection'],
                                            params['energy_var'],
                                            params['fitness_sum_limit'])
    param = np.random.rand()
    print_output('Try to crossover.')
    cnt = 0
    while param < params['prob_for_crossing'] and cnt < cnt_max:
        # print_output('Values for {} parent_1'.format(parent1))
        # run_util.str_info(parent1)
        # print_output('Values for {} parent_2'.format(parent2))
        # run_util.str_info(parent2)
        # print_output('\n')
        child1, child2 = Structure.crossover(parent1, parent2, method=mol.crossover_method)
        if child1.is_geometry_valid_after_crossover() and child2.is_geometry_valid_after_crossover():
            if not check_for_clashes(child1.sdf_string, os.path.join(mol.constrained_geometry_file)):
                #print_output('Clash found')
                if 'centroid' not in mol.dof_names:
                    print_output('Perform adjust')
                    child1.adjust_position()
                else:
                    Structure.index = len(blacklist)
                    cnt += 1
                    continue
            if not check_for_clashes(child2.sdf_string, os.path.join(mol.constrained_geometry_file)):
                #print_output('Clash found')
                if 'centroid' not in mol.dof_names:
                    print_output('Perform adjust')
                    child2.adjust_position()
                else:
                    Structure.index = len(blacklist)
                    cnt += 1
                    continue
            print_output('Values for {} parent_1'.format(parent1))
            run_util.str_info(parent1)
            print_output('Values for {} parent_2'.format(parent2))
            run_util.str_info(parent2)
            print_output('\n')
            break
        else:
            Structure.index = len(blacklist)
            cnt += 1
            continue
    else:
        child1, child2 = Structure(parent1), Structure(parent2)
        print_output('No crossover was performed. Children are copies of parents.')
        # Delete inherited attributes.
        for child in child1, child2:
            attr_list = ["initial_sdf_string", "energy"]
            for attr in attr_list:
                delattr(child, attr)
            for dof in child.dof:
                delattr(dof, "initial_values")
    print_output('Values for {} child_1'.format(child1))
    run_util.str_info(child1)
    print_output('Values for {} child_2'.format(child2))
    run_util.str_info(child2)
    print_output('\n')
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
    print_output("Sorted population: " + ', '.join([str(v) for v in population]))
    del population[-1]
    del population[-1]
    print_output("Sorted population after removing two structures with highest"
                 " energy: " + ', '.join([str(v) for v in population]))
    min_energy.append(population[0].energy)
    print_output("Current population after sorting: ")
    for i in range(len(population)):
        print_output('{:<}   {:>}'.format(population[i], float(population[i])))
    print_output("Lowest energies in run: {}".format(min_energy))
    run_util.perform_backup(mol, population, blacklist, iteration, min_energy)
    run_util.check_for_convergence(iteration, params, min_energy)
    run_util.check_for_kill()
    iteration += 1
