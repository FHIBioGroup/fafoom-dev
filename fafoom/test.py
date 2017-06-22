#!/usr/bin/python
from __future__ import division
import math
from copy import copy
from random import choice
from rdkit import Chem

from utilities import ig, cleaner, get_vec, tor_rmsd, find_one_in_list
from measure import (
    dihedral_measure,
    dihedral_set,
    pyranosering_measure,
    pyranosering_set
)

from fafoom import MoleculeDescription, Structure, selection, print_output,\
    remover_dir, set_default, file2dict
import fafoom.run_utilities as run_util

from genetic_operations import mutation


class DOF:

    def __init__(self, name):
        self.name = name

    def common_function():
        pass

class CentofMass(DOF):
    default_center = [0, 0, 0]

    @staticmethod
    def find(smiles):
        if positions is None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("The smiles is invalid")
            centroid = Chem.rdMolTransforms.ComputeCentroid(mol)
        return centroid


smiles_code = 'CCCCCCCCCCC'
mol = Chem.MolFromSmiles(smiles_code)
#~ mol=Chem.AddHs(mol)
#~ mol.AddConformer(mol)
#~ conf = mol.GetConformer()
#~ centroid = Chem.rdMolTransforms.ComputeCentroid(conf, ignoreHs=True )
#~ print cenroid
#~ conf=mol.GetConformer()
#~ natom=mol.GetNumAtoms()

str3d = Structure(mol)
str3d.generate_structure()
