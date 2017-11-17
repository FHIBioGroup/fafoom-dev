#!/usr/bin/python
import sys
import re
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from operator import itemgetter
from copy import copy

def ig(x):
    return itemgetter(x)

def cleaner(list_to_clean):
    """ Remove duplicate torsion definion from a list of atom ind. tuples."""
    for_remove = []
    for x in reversed(range(len(list_to_clean))):
        for y in reversed(range(x)):
            ix1, ix2 = ig(1)(list_to_clean[x]), ig(2)(list_to_clean[x])
            iy1, iy2 = ig(1)(list_to_clean[y]), ig(2)(list_to_clean[y])
            if (ix1 == iy1 and ix2 == iy2) or (ix1 == iy2 and ix2 == iy1):
                for_remove.append(y)
    clean_list = [v for i, v in enumerate(list_to_clean)
                  if i not in set(for_remove)]
    return clean_list

""" Take smiles code """
if len(sys.argv) == 1 and os.path.exists(os.path.join(os.getcwd(), 'mol.smi')):
    with open(os.path.join(os.getcwd(), 'mol.smi'), 'r') as mol:
        lines = mol.readlines()
        for line in lines:
                smiles_found = re.match(r'(\s*?(.+)\s*?)', line)
                if smiles_found:
                    smiles = smiles_found.group(1)
else:
    if os.path.exists(os.path.join(os.getcwd(), sys.argv[1])):
        with open(os.path.join(os.getcwd(), sys.argv[1]), 'r') as mol:
            lines = mol.readlines()
            for line in lines:
                    smiles_found = re.match(r'(\s*?(.+)\s*?)', line)
                    if smiles_found:
                        smiles = smiles_found.group(1)
    else:
        smiles = sys.argv[1] #Read smiles from input


smarts_torsion= "[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]" #Definitions of torsion in smarts notation
smarts_cistrans= "C~[$(C=O)]-[$(NC)]~[C]"
filter_smarts_torsion= "C~[$(C=O)]-[$(NC)]~[C]" # definition of cistrans for peptides

mol = Chem.MolFromSmiles(smiles) #Produce molecule from smiles code
pattern_tor = Chem.MolFromSmarts(smarts_torsion) #Pattern for torsion
pattern_cis = Chem.MolFromSmarts(smarts_cistrans)
torsion = list(mol.GetSubstructMatches(pattern_tor)) #Obtain all the torsions
cistrans = list(mol.GetSubstructMatches(pattern_cis))


if filter_smarts_torsion: #Filter particular torsions from all obtained previously
    pattern_custom = Chem.MolFromSmarts(filter_smarts_torsion)
    custom = list(mol.GetSubstructMatches(pattern_custom))
    to_del_bef_custom = []
    for x in reversed(range(len(torsion))):
	for y in reversed(range(len(custom))):
	    ix1, ix2 = ig(1)(torsion[x]), ig(2)(torsion[x])
	    iy1, iy2 = ig(1)(custom[y]), ig(2)(custom[y])
	    if (ix1 == iy1 and ix2 == iy2) or (ix1 == iy2 and
					       ix2 == iy1):
		to_del_bef_custom.append(x)
    custom_torsion = copy(torsion)
    custom_torsion = [v for i, v in enumerate(custom_torsion)
		      if i not in set(to_del_bef_custom)]
    torsion = custom_torsion
positions = cleaner(torsion) #Return list consist of tuples which contain 4-atoms define torsion angle

#def add_one_to_positions(positions):
#  positions = [(i[0]+1, i[1]+1, i[2]+1, i[3]+1) for i in positions]
#  return positions
#positions_vmd = add_one_to_positions(positions)
#Produce mol.sdf file of the molecule
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
with open(os.path.join(os.getcwd(),'mol.sdf'),'w') as molecule:
    molecule.write(Chem.MolToMolBlock(mol))

with open(os.path.join(os.getcwd(), 'parameters.txt'), 'w')  as params:
    # params.write('Parameters file fpr Genetic Algorithm')
    # params.write('\n')
    params.write('[Molecule]\n')
    params.write('\n')
    params.write('rmsd_type=\"internal_coord\"\n')
    params.write('rmsd_cutoff_uniq=0.1\n')
    params.write('chiral=True\n')
    params.write('optimize_centroid=False\n')
    params.write('optimize_orientation=False\n')
    params.write('optimize_torsion=False\n')
    params.write('optimize_cistrans=False\n')
    params.write('volume=(-10, 10, -10, 10, -10, 10)\n')
    if len(positions) > 0:
        params.write('list_of_torsion  = {}\n'.format(positions))
#    if len(positions) > 0:
#        params.write('#list_of_torsion_for_vmd  = {}\n'.format(positions_vmd))
    if len(cistrans) > 0:
        params.write('list_of_cistrans = {}\n'.format(cistrans))
    params.write('\n')
    params.write('[GA settings]\n')
    params.write('\n')
    params.write('energy_var=0.001\n')
    params.write('selection=\"roulette_wheel\"\n')
    params.write('fitness_sum_limit= 1.2\n')
    params.write('popsize= 10\n')
    params.write('prob_for_crossing= 0.9\n')
    params.write('prob_for_mut_torsion= 0.1\n')
    params.write('prob_for_mut_cistrans= 0.1\n')
    params.write('max_mutations_torsion= 3\n')
    params.write('max_mutations_cistrans=1\n')
    params.write('prob_for_mut_orientation= 0.1\n')
    params.write('prob_for_mut_centroid= 0.1\n')
    params.write('max_mutations_orientation= 3\n')
    params.write('max_mutations_centroid=1\n')
    params.write('\n')
    params.write('[Run settings]\n'  )
    params.write('\n')
    params.write('energy_function = \"FHI-aims\"\n')
    params.write('sourcedir= \"adds\"\n')
    params.write('aims_call= \"aims.x\"\n')
    #params.write('aims_call= \"mpiexec -n $NSLOTS aims.071914_7.scalapack.mpi.x\"\n')
    params.write('max_iter= 30\n')
    params.write('iter_limit_conv= 20\n')
    params.write('energy_diff_conv= 0.001\n')

print smiles
print 'list_of_torsion  = {}'.format(positions)
print 'list_of_cistrans = {}'.format(cistrans)
print 'Don\'t forget to specify energy_function and aims_call flags!'
