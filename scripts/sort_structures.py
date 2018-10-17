#!/usr/bin/env python

import os, re, shutil
from optparse import OptionParser
import numpy as np
np.set_printoptions(suppress=True)	# Prevent numpy exponential
                                    # notation on print, default 
import math
import operator                                    

# Load input:
parser = OptionParser()
parser.add_option("-i", "--inputfile", dest="inputfile", help="Input xyz file with structures and energies in comments")
(options, args) = parser.parse_args()

list_of_torsion  = [(1, 3, 4, 9), (9, 4, 5, 7), (0, 1, 3, 4), (4, 5, 7, 8)]




# Additional lib with atom masses:
atom_masses = {'H':  1.00794, 'He':  4.002602, 'Li':  6.941, 'Be':  9.012182, 'B':  10.811, 'C':  12.011, 'N':  14.00674, 'O':  15.9994, 'F':  18.9984, 'Ne':  20.1797, 'Na':  22.98977, 'Mg':  24.305, 'Al':  26.98154, 'Si':  28.0855, 'P':  30.97376, 'S':  32.066, 'Cl':  35.4527, 'K':  39.0983, 'Ar':  39.948, 'Ca':  40.078, 'Sc':  44.95591, 'Ti':  47.88, 'V':  50.9415, 'Cr':  51.9961, 'Mn':  54.93805, 'Fe':  55.847, 'Ni':  58.6934, 'Co':  58.9332, 'Cu':  63.546, 'Zn':  65.39, 'Ga':  69.723, 'Ge':  72.61, 'As':  74.92159, 'Se':  78.96, 'Br':  79.904, 'Kr':  83.8, 'Rb':  85.4678, 'Sr':  87.62, 'Y':  88.90585, 'Zr':  91.224, 'Nb':  92.90638, 'Mo':  95.94, 'Ru':  101.07, 'Rh':  102.9055, 'Pd':  106.42, 'Ag':  107.8682, 'Cd':  112.411, 'In':  114.818, 'Sn':  118.71, 'Sb':  121.757, 'I':  126.9045, 'Te':  127.6, 'Xe':  131.29, 'Cs':  132.9054, 'Ba':  137.327, 'La':  138.9055, 'Ce':  140.115, 'Pr':  140.9077, 'Nd':  144.24, 'Sm':  150.36, 'Eu':  151.965, 'Gd':  157.25, 'Tb':  158.9253, 'Dy':  162.5, 'Ho':  164.9303, 'Er':  167.26, 'Tm':  168.9342, 'Yb':  173.04, 'Lu':  174.967, 'Hf':  178.49, 'Ta':  180.9479, 'W':  183.85, 'Re':  186.207, 'Os':  190.2, 'Ir':  192.22, 'Pt':  195.08, 'Au':  196.9665, 'Hg':  200.59, 'Tl':  204.3833, 'Pb':  207.2, 'Bi':  208.9804, 'Po':  208.9824, 'At':  209.9871, 'Pa':  213.0359, 'Ra':  226.0254, 'Ac':  227.0728, 'Th':  232.0381, 'Np':  237.0482, 'U':  238.0289, 'Am':  243.0614, 'Pu':  244.0642}


def extract_indexes(inp):
    inds=[]
    with open(inp) as i:
        lines = i.readlines()
        for line in lines:
            numatoms=re.match(r'(^\s*?\d+\s*?$)', line)
            if numatoms:
                inds.append(int(numatoms.group(1)))
    return inds

def split_xyz(inp, inds, folder):
    count=0
    with open(os.path.join(folder, inp)) as i:
        lines=i.readlines()
        first=0
        last=0
        for i in inds:
            count+=1
            last+=i+2
            with open(os.path.join(folder, '{:08d}'.format(count)), 'w') as spl:
                for li in lines[first:last]:
                    spl.write(li)
            first+=i+2

def split_structures(inp, folder):
    inds=extract_indexes(os.path.join(folder,inp))
    split_xyz(inp, inds, folder)

primename = options.inputfile.split('.')[:-1][0]

# Name for temp folder:
TEMP = os.path.join(os.getcwd(), 'temp')

# Create temp folder:
if not os.path.exists(TEMP):
    os.mkdir(TEMP)
else:
    shutil.rmtree(TEMP)
    os.mkdir(TEMP)

# Copy file to temp folder
shutil.copy(options.inputfile, TEMP)
      
# Split structures in temp folder:
split_structures(options.inputfile, TEMP)

# Delete trajectory file in 'temp' folder
os.system('rm {}'.format(os.path.join(TEMP, options.inputfile)))

# Print how many structures we have:
print('Total number of structures: {}'.format(len(os.listdir(TEMP))))

"""
Now we have all structures separated in different files.
And we can perform analysis of properties for each structure
such as Center of Mass (COM), Energy e.t.c
For that also we define help functions:
"""

#-----------------------------------------------------------------------
# Extract coordinates and atomspecies:
# Output will be array [coord1, coord2, coord3, atommass]
def coords_masses(xyz_file):
    def to_array(ll):
        return([float(ll[1]), float(ll[2]), float(ll[3]), atom_masses[ll[0]]])
    data = [to_array(line.split()) for line in xyz_file]
    return(np.array(data))

# Extract energies
def energy(line):
    en = re.match(r'.*?(-?\d+\.\d+).*?', line)
    if en:
        return(float(en.group(1)))

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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if np.linalg.norm(vector) == 0.:
        return vector
    else:
        return vector / np.linalg.norm(vector)

def angle_between(coords, atoms):
    """Returns angle between two vectors in radians"""
    v1_u = unit_vector(coords[atoms[0]]-coords[atoms[1]])
    v2_u = unit_vector(coords[atoms[1]]-coords[atoms[2]])
    return np.arccos(np.dot(v1_u, v2_u))

def dihedral_measure(coords, list_of_atoms):
    """ Measure the dihedral angle.
    Args:
        sdf_string (string)
        position (list): 4 atoms defining the dihedral
    Returns:
        float value
    Raises:
        ValueError: If the lenght of the list is not equal 4.
    """
    if len(list_of_atoms) != 4:
        raise ValueError("The position needs to be defined by 4 integers")

    plane1 = calculate_normal_vector(list_of_atoms[:3], coords)
    plane2 = calculate_normal_vector(list_of_atoms[1:], coords)
    #   Calculate a norm of normal vectors:
    norm_plane1 = np.sqrt(np.sum(plane1**2))
    norm_plane2 = np.sqrt(np.sum(plane2**2))
    norm = norm_plane1 * norm_plane2
    #   Measure the angle between two planes:
    dot_product = np.dot(plane1, plane2)/norm
    alpha = np.arccos(dot_product)
    #   The cosine function is symmetric thus, to distinguish between
    #   negative and positive angles, one has to calculate if the fourth
    #   point is above or below the plane defined by first 3 points:
    ppoint = - np.dot(plane1, coords[list_of_atoms[0], :])
    dpoint = (np.dot(plane1, coords[list_of_atoms[3], :])+ppoint)/norm_plane1
    if dpoint >= 0:
        return -alpha
    else:
        return alpha

#-----------------------------------------------------------------------
IDs = {}
dihedrals, energies, names=[],[],[]

z=-1
for structure in os.listdir(TEMP):
    z+=1
    with open(os.path.join(TEMP, structure)) as f:
        lines = f.readlines()
        en = energy(lines[1])
        energies.append(en)
        # Calculate Dihedral Angles.
        dihed = []
        for l in list_of_torsion:
            dihed.append(dihedral_measure(coords_masses(lines[2:])[:, :3], l))
        names.append(structure)
        IDs[structure] = [en, dihed, structure]

#Sort structures according to the energy:
sorted_IDs = sorted(IDs.items(), key=operator.itemgetter(1))

hidim = open('{}_data.dat'.format(primename), 'w') 
hidim.write('{:<20}{:<20}{:<20}{:<20}'.format('Hierarchy', 'rel_energy', 'energy', 'name'))
for k in range(len(dihed)):
    hidim.write('{:<20}'.format('DIH_{}'.format(k)))
hidim.write('\n')

with open('{}_sorted.xyz'.format(primename), 'w') as sor_str:
    ids=0
    for i in sorted_IDs:
        k = i[1]
        ids+=1
        with open(os.path.join(TEMP, i[0])) as f:
            lines = f.read()
            sor_str.write(lines)
        energyyy=k[0]
        dihedral=k[1]
        name=k[2]
        hidim.write('{:<20}{:<20}{:<20}{:<20}'.format(1, energyyy-min(energies), energyyy, name))
        for z in range(len(dihedral)):
            hidim.write('{:<20}'.format(format(dihedral[z])))
        hidim.write('\n')		
hidim.close()

# Delete temp folder:
# ~ shutil.rmtree('temp')




