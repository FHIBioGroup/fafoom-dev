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
''' Communicate between the structure and the degrees of freedom.'''
from __future__ import division
# from rdkit import Chem
# from rdkit.Chem import AllChem
from deg_of_freedom import Torsion, CisTrans, PyranoseRing, Centroid, Orientation, Protomeric
from utilities import *
from measure import *


def get_atoms_and_bonds(sdf_string):
    """Build the molecule from SMILES and return the number of atoms and bonds.

    Args(required):
        smiles (str): one-line representation of the molecule
    Returns:
        Number of atoms, number of bonds
    """

    n_at = len(coords_and_masses_from_sdf(sdf_string))
    n_bonds = len(conn_list_from_sdf(sdf_string))
    return n_at, n_bonds
    # mol = Chem.MolFromSmiles(smiles)
    # if mol is None:
    #     raise ValueError("The smiles is invalid")
    # mol = Chem.AddHs(mol)
    # return mol.GetNumAtoms(), mol.GetNumBonds()
#smiles="[NH3+][C@H](C(=O)N1[C@H](C(=O)N[C@H](C(=O)[O-])Cc2ccccc2)CCC1)Cc1[nH]c[nH+]c1"

def get_positions(type_of_deg, sdf_string, **kwargs):
    """Find the positions (tuples of atom indicies) of the degrees of freedom.

    Args(required):
        type_of_deg (str)
        smiles (str)
        if cistrans should be optimized:
            smarts_cistrans
    Args(optimal):
        list_of_torsion (list)
        smarts_torsion (str)
        filter_smarts_torsion (str)
        list_of_cistrans (list)
        list_of_pyranosering (list)
    Returns:
        list of touples defining the positions of the degree of freedom
    """




    '''    ####################################   '''
    ''' Switch only to [list_of_torsional] option '''
    '''    ####################################   '''

    if type_of_deg == "torsion":
        if 'list_of_torsion' in kwargs:
            return Torsion.find(sdf_string, positions=kwargs['list_of_torsion'])
        # else:
        #     if 'smarts_torsion' in kwargs:
        #         if 'filter_smarts_torsion' in kwargs:
        #             return Torsion.find(smiles,
        #                                 smarts_torsion=kwargs['smarts_torsion'],
        #                                 filter_smarts_torsion=
        #                                 kwargs['filter_smarts_torsion'])
        #         else:
        #             return Torsion.find(smiles,
        #                                 smarts_torsion=kwargs['smarts_torsion'])
        #     else:
        #         return Torsion.find(smiles)

    '''  #####################################  '''

    if type_of_deg == "cistrans":
        if 'list_of_cistrans' in kwargs:
            return CisTrans.find(sdf_string, positions=kwargs['list_of_cistrans'])
        # else:
        #     return CisTrans.find(smiles,
        #                          smarts_cistrans=kwargs['smarts_cistrans'])

    if type_of_deg == "pyranosering":
        if 'list_of_pyranosering' in kwargs:
            return PyranoseRing.find(sdf_string,
                                     positions=kwargs['list_of_pyranosering']) ###NEEED TO CLARIFY
        # else:
        #     return PyranoseRing.find(smiles)

    '''  #####################################  '''

    if type_of_deg == "centroid":
        if 'list_of_centroid' in kwargs:
            return Centroid.find(sdf_string,
                                     positions=kwargs['list_of_centroid'])
        else:
            return Centroid.find(sdf_string)

    if type_of_deg == "orientation":
        if 'list_of_orientation' in kwargs:
            return Orientation.find(sdf_string,
                                     positions=kwargs['list_of_orientation'])
        else:
            return Orientation.find(sdf_string)

    if type_of_deg == "protomeric":
        if 'list_of_protomeric' in kwargs:
            return Protomeric.find(sdf_string, positions=kwargs['list_of_protomeric'])


def create_dof_object(type_of_deg, positions):
    """Initialize the degree of freedom from the positions

    Args:
        type_of_deg (str)
        positsion (list)
    Returns:
        degree of freedom object
    """
    if type_of_deg == "torsion":
        return Torsion(positions)
    if type_of_deg == "cistrans":
        return CisTrans(positions)
    if type_of_deg == "pyranosering":
        return PyranoseRing(positions)
    if type_of_deg == "centroid":
        return Centroid(positions)
    if type_of_deg == "orientation":
        return Orientation(positions)
    if type_of_deg == "protomeric":
        return Protomeric(positions)

def template_sdf(sdf_string):
    """Create a template sdf string and writes it to file.

    Args(required):
        smiles (str): one-line representation of the molecule
    Returns:
        sdf string
    """
    # with open(os.path.join(os.getcwd(), sdf_file), 'r') as sdf_file:
    #     sdf_string = sdf_file.read()
    # cnt = 0
    # sdf_check = True
    # while sdf_check:
    #     mol = Chem.MolFromSmiles(smiles)
    #     mol = Chem.AddHs(mol)
    #     AllChem.EmbedMolecule(mol)
    #     AllChem.UFFOptimizeMolecule(mol)
    #     Chem.SDWriter('mol.sdf').write(mol)
    #     sdf_string = Chem.MolToMolBlock(mol)
    #     check = check_geo_sdf(sdf_string)
    #     if check:
    #         sdf_check = False
    #         Chem.SDWriter('mol.sdf').write(mol)
    #     else:
    #         cnt += 1
    return sdf_string