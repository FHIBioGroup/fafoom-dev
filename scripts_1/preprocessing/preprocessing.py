#!/usr/bin/python
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from operator import itemgetter
from copy import copy
import numpy as np
import math

def ig(x):
    return itemgetter(x)

def connected_components(neighbors):
    seen = set()
    def component(node):
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield component(node)
    
    
    
    
with open(os.path.join(path, 'mol.sdf'), 'r') as mol:
    lines = mol.readlines()
    for line in lines:
	#~ coord.found = re.match(r'()', line)
	bond_found  = re.match(r'(\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+$)', line)
	if bond_found:
	    conn_list.append([int(bond_found.group(2)), int(bond_found.group(3)), int(bond_found.group(4))])


graph = construct_graph(conn_list)
print conn_list


def get_coords(sdf_string):
    mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)    
    pos = mol.GetConformer()   
    coords = np.array([np.array([pos.GetAtomPosition(i).x, pos.GetAtomPosition(i).y, pos.GetAtomPosition(i).z]) for i in range(mol.GetNumAtoms())])        #Return Atom Masses.
    return coords   

def dihedral_measure(sdf_string, position):
    """ Measure the dihedral angle.

    Args:
        sdf_string (string)
        position (list): 4 atoms defining the dihedral
    Returns:
        float value
    Raises:
        ValueError: If the lenght of the list is not equal 4.
    """
    if len(position) != 4:
        raise ValueError("The position needs to be defined by 4 integers")
    mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
    val = float(rdMolTransforms.GetDihedralDeg(
                mol.GetConformer(),
                ig(0)(position), ig(1)(position),
                ig(2)(position), ig(3)(position)))
    return float('{0:.2f}'.format(val))
    
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

#~ smiles = sys.argv[1] #Read smiles 
smiles = "[NH3+][C@H](C(=O)N1[C@H](C(=O)N[C@H](C(=O)O)Cc2ccccc2)CCC1)Cc1nc[nH]c1"
smarts_torsion= "[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]" #Definitions of torsion in smarts notation
filter_smarts_torsion= "C~[$(C=O)]-[$(NC)]~[C]" # definition of torsions for peptides???



mol = Chem.MolFromSmiles(smiles) #Produce molecule from smiles code
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)

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
   

def dihedral_measure_sub(sdf_string, atoms):
    coords = get_coords(sdf_string)
    
    p1 = coords[atoms[0]]    
    p2 = coords[atoms[1]]    
    p3 = coords[atoms[2]]    
    p4 = coords[atoms[3]]    
    
    q1 = p2-p1
    q2 = p3-p2
    q3 = p4-p3
    
    q1_x_q2 = np.cross(q1, q2)
    q2_x_q3 = np.cross(q2, q3)
    
    n1 = q1_x_q2/np.sqrt(np.dot(q1_x_q2,q1_x_q2))
    n2 = q2_x_q3/np.sqrt(np.dot(q2_x_q3,q2_x_q3))
   
    u1 = n2
    u3 = q2/(np.sqrt(np.dot(q2,q2)))
    u2 = np.cross(u3,u1)    

    # Calculate cosine and sine
    cos_theta = np.dot(n1,u1)
    sin_theta = np.dot(n1,u2)
    # Calculate theta
    theta = -math.atan2(sin_theta,cos_theta) # it is different from Fortran math.atan2(y,x)
    theta_deg = np.degrees(theta)

    return theta_deg

###ADRIANA###
def calculate_normal_vector(list_of_atoms, xyz):
    """Calculate the normal vector of a plane by
    cross product of two vectors belonging to it.

    Args:
	list_of_atoms: list of 3 atoms
	xyz: numpy array with atoms xyz position
    """

    r0 = xyz[list_of_atoms[1], :] - xyz[list_of_atoms[0], :]
    r1 = xyz[list_of_atoms[2], :] - xyz[list_of_atoms[1], :]
    cross_product = np.cross(r1, r0)

    return cross_product

def measure_dihedral(list_of_atoms, xyz):
    """Calculate a dihedral angle between two planes defined by
    a list of four atoms. It returns the angle and the rotation axis
    required to set a new dihedral.

    Args:
	list_of_atoms: list of 4 atoms
	xyz: numpy array with atom xyz positions
    """

    plane1 = calculate_normal_vector(list_of_atoms[:3], xyz)
    plane2 = calculate_normal_vector(list_of_atoms[1:], xyz)
    #Calculate the axis of rotation (axor)
    axor = np.cross(plane1, plane2)

    #Calculate a norm of normal vectors:
    norm_plane1 = np.sqrt(np.sum(plane1**2))
    norm_plane2 = np.sqrt(np.sum(plane2**2))
    norm = norm_plane1 * norm_plane2
    #Measure the angle between two planes:
    dot_product = np.dot(plane1, plane2)/norm
    alpha = np.arccos(dot_product)

    #The cosine function is symetric thus, to distinguish between
    #negative and positive angles, one has to calculate if the fourth
    #point is above or below the plane defined by first 3 points:
    ppoint = - np.dot(plane1, xyz[list_of_atoms[0], :])
    dpoint = (np.dot(plane1, xyz[list_of_atoms[3], :])+ppoint)/norm_plane1
    if dpoint >= 0:
	return -(alpha*180.0)/np.pi, axor
    else:
	return (alpha*180.0)/np.pi, axor
    
    
###TEST FOR DIHEDRALS!!!###

for i in positions:
    print dihedral_measure(Chem.MolToMolBlock(mol), i)
    print dihedral_measure_sub(Chem.MolToMolBlock(mol), i)
    print measure_dihedral(i, get_coords(Chem.MolToMolBlock(mol)))
    
    
