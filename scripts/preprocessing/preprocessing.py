#!/usr/bin/python
import sys
from rdkit import Chem
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

smiles = sys.argv[1] #Read smiles 
smarts_torsion= "[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]" #Definitions of torsion in smarts notation
filter_smarts_torsion= "C~[$(C=O)]-[$(NC)]~[C]" # definition of torsions for peptides???

mol = Chem.MolFromSmiles(smiles) #Produce molecule from smiles code
pattern_tor = Chem.MolFromSmarts(smarts_torsion) #Pattern for torsion
torsion = list(mol.GetSubstructMatches(pattern_tor)) #Obtain all the torsions

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
print positions
