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
''' Collection of diverse help/convert functions '''
from __future__ import division
import os, re
import numpy as np
import math
import shutil
import ConfigParser
from operator import itemgetter
np.set_printoptions(suppress=True)

#~ from rdkit import Chem
#~ from rdkit.Chem import AllChem

# Flow-handling
# In Bohr
atom_masses = {'H':  1.00794, 'He':  4.002602, 'Li':  6.941, 'Be':  9.012182, 'B':  10.811, 'C':  12.011, 'N':  14.00674, 'O':  15.9994, 'F':  18.9984, 'Ne':  20.1797, 'Na':  22.98977, 'Mg':  24.305, 'Al':  26.98154, 'Si':  28.0855, 'P':  30.97376, 'S':  32.066, 'Cl':  35.4527, 'K':  39.0983, 'Ar':  39.948, 'Ca':  40.078, 'Sc':  44.95591, 'Ti':  47.88, 'V':  50.9415, 'Cr':  51.9961, 'Mn':  54.93805, 'Fe':  55.847, 'Ni':  58.6934, 'Co':  58.9332, 'Cu':  63.546, 'Zn':  65.39, 'Ga':  69.723, 'Ge':  72.61, 'As':  74.92159, 'Se':  78.96, 'Br':  79.904, 'Kr':  83.8, 'Rb':  85.4678, 'Sr':  87.62, 'Y':  88.90585, 'Zr':  91.224, 'Nb':  92.90638, 'Mo':  95.94, 'Ru':  101.07, 'Rh':  102.9055, 'Pd':  106.42, 'Ag':  107.8682, 'Cd':  112.411, 'In':  114.818, 'Sn':  118.71, 'Sb':  121.757, 'I':  126.9045, 'Te':  127.6, 'Xe':  131.29, 'Cs':  132.9054, 'Ba':  137.327, 'La':  138.9055, 'Ce':  140.115, 'Pr':  140.9077, 'Nd':  144.24, 'Sm':  150.36, 'Eu':  151.965, 'Gd':  157.25, 'Tb':  158.9253, 'Dy':  162.5, 'Ho':  164.9303, 'Er':  167.26, 'Tm':  168.9342, 'Yb':  173.04, 'Lu':  174.967, 'Hf':  178.49, 'Ta':  180.9479, 'W':  183.85, 'Re':  186.207, 'Os':  190.2, 'Ir':  192.22, 'Pt':  195.08, 'Au':  196.9665, 'Hg':  200.59, 'Tl':  204.3833, 'Pb':  207.2, 'Bi':  208.9804, 'Po':  208.9824, 'At':  209.9871, 'Pa':  213.0359, 'Ra':  226.0254, 'Ac':  227.0728, 'Th':  232.0381, 'Np':  237.0482, 'U':  238.0289, 'Am':  243.0614, 'Pu':  244.0642}
VDW_radii = {'H': 3.1000,'He': 2.6500,'Li': 4.1600,'Be': 4.1700,'B':  3.8900,'C':  3.5900,'N':  3.3400,'O':  3.1900,'F':  3.0400,'Ne': 2.9100,'Na': 3.7300,'Mg': 4.2700,'Al': 4.3300, 'Si': 4.2000, 'P':  4.0100,'S':  3.8600,'Cl': 3.7100,'Ar': 3.5500,'K':  3.7100,'Ca': 4.6500,'Sc': 4.5900,'Ti': 4.5100,'V':  4.4400,'Cr': 3.9900,'Mn': 3.9700,'Fe': 4.2300,'Co': 4.1800,'Ni': 3.8200,'Cu': 3.7600,'Zn': 4.0200,'Ga': 4.1900,'Ge': 4.2000,'As': 4.1100,'Se': 4.0400,'Br': 3.9300,'Kr': 3.8200,'Rb': 3.7200,'Sr': 4.5400,'Y':  4.8151,'Zr': 4.53,'Nb': 4.2365,'Mo': 4.099,'Tc': 4.076,'Ru': 3.9953,'Rh': 3.95,'Pd': 3.6600,'Ag': 3.8200,'Cd': 3.9900,'In': 4.2319,'Sn': 4.3030,'Sb': 4.2760,'Te': 4.2200,'I':  4.1700,'Xe': 4.0800,'Cs': 3.78,'Ba': 4.77,'La': 3.14,'Ce': 3.26,'Pr': 3.28,'Nd': 3.3,'Pm': 3.27,'Sm': 3.32,'Eu': 3.40,'Gd': 3.62,'Tb': 3.42,'Dy': 3.26,'Ho': 3.24,'Er': 3.30,'Tm': 3.26,'Yb': 3.22,'Lu': 3.20,'Hf': 4.21,'Ta': 4.15,'W':  4.08,'Re': 4.02,'Os': 3.84,'Ir': 4.00,'Pt': 3.92,'Au': 3.86,'Hg': 3.98,'Tl': 3.91,'Pb': 4.31,'Bi': 4.32,'Po': 4.097,'At': 4.07,'Rn': 4.23,'Fr': 3.90,'Ra': 4.98,'Ac': 2.75,'Th': 2.85,'Pa': 2.71,'U':  3.00,'Np': 3.28,'Pu': 3.45,'Am': 3.51,'Cm': 3.47,'Bk': 3.56,'Cf': 3.55,'Es': 3.76,'Fm': 3.89,'Md': 3.93,'No': 3.78}
bohrtoang=0.52917721

def backup(filename, obj):
    """ Write the representation of an object (or objects) to a file."""
    with open(filename, 'w') as outf:
        if hasattr(obj, "__len__"):
            for i in range(len(obj)):
                outf.write("%s\n" % repr(obj[i]))
        else:
            outf.write("%s\n" % repr(obj).replace('\n', 'NEWLINE'))
    outf.close()


def boolean(string):
    """Recover the boolean value from a string and return it."""
    if string in ["False", "false", "FALSE"]:
        return False
    if string in ["True", "true", "TRUE"]:
        return True
    raise ValueError("Cannot be converted to a boolean type")


def number(s):
    """Convert to integer of float if needed"""
    try:
        return int(s)
    except ValueError:
        return float(s)


def print_output(text):
    """Write text to the 'output.txt'. Create it if needed."""
    if os.path.isfile("output.txt"):
        f = open("output.txt", "a")
        f.write(str(text)+'\n')
        f.close()
    else:
        f = open("output.txt", "w")
        f.write(str(text)+'\n')
        f.close()


def remover_file(instance):
    """Remove a file (if it exists)."""
    try:
        os.remove(instance)
    except OSError:
        pass


def remover_dir(instance):
    """Remove a directory (if it exists)."""
    try:
        shutil.rmtree(instance)
    except OSError:
        pass


def file2string(input_file):
    """Read a file to a string and return it."""
    with open(input_file, 'r') as f:
        string = f.read()
    f.close()
    return string


def string2file(string, filename):
    """Write a string to a file"""
    with open(filename, 'w') as target:
        target.write(string)
    target.close()

def coords_and_masses_from_sdf(sdf_string, removeHs=False):
    """Convert a sdf_string to a xyz_list."""
    xyz_list = []
    if removeHs == False:
        for line in sdf_string.split('\n'):
            coords_found = re.match(r'(\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(\w+)\s+)', line)
            if coords_found:
                xyz_list.append(np.array([float(coords_found.group(2)),
                                        float(coords_found.group(3)),
                                        float(coords_found.group(4)),
                                        float(atom_masses[coords_found.group(5)])]))
    else:
        for line in sdf_string.split('\n'):
            coords_found = re.match(r'(\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(\w+)\s+)', line)
            if coords_found:
                if coords_found.group(5) != 'H':
                    xyz_list.append(np.array([float(coords_found.group(2)),
                                            float(coords_found.group(3)),
                                            float(coords_found.group(4)),
                                            float(atom_masses[coords_found.group(5)])]))
    return np.array(xyz_list)

def get_centre_of_mass_from_sdf(sdf_string, removeHs=False):
    coords_and_masses = coords_and_masses_from_sdf(sdf_string, removeHs)
    center_of_mass = np.average(coords_and_masses[:,:3], axis=0, weights=coords_and_masses[:,3])
    return center_of_mass

def generate_extended_input(string, constrained_part_file, filename):
    with open(constrained_part_file, 'r') as part:
        constrained_part = part.readlines()
    with open(filename, 'w') as target:
        target.write(string)
        for line in constrained_part:
            target.write(line)
        target.write('\n')

def set_default(params, dict_default):
    """Set defaults for missing keys and add the key:value pairs to the
    dict."""
    for key in dict_default:
        if key not in params:
            # print_output("Setting a default value for "+str(key)+": " +
            #              str(dict_default[key]))
            params[str(key)] = dict_default[key]
    return params

def file2dict(filename, sections):
    """Parse a file and create a dictionary"""
    config = ConfigParser.RawConfigParser()
    config.read(filename)
    new_dict = {}
    for section in sections:
        if config.has_section(section):
            for key, value in config.items(section):
                new_dict[str(key)] = eval(value)
    return new_dict

# Help vector/matrix functions

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


def get_vec(vec1, vec2):
    """Calculate difference between vectors of angles [in rad!].
    Args:
        vec1 (list) in deg
        vec2 (list) in deg
    Returns:
        numpy array
    Raises:

        ValueError: if the length of the lists differ

    Warning1: the vectors contain periodic values, i.e -185 -> 175
    Warning2: symmetry is not included here, but can be easily added if the
    index of the 'symmetric' torsion is known
    """
    if len(vec1) != len(vec2):
        raise ValueError("No length match between the lists")
    diff_vec = np.zeros(len(vec1))
    for i in range(0, len(vec1)):
        tor_diff = abs(vec1[i]-vec2[i])
        diff_vec[i] = min(abs(tor_diff), abs(360-tor_diff))/180.0
    return diff_vec


def tor_rmsd(p, vec):
    """Calculate the modified p norm.The difference from standard norm is the
    fact that the addends are divided by the length of the vector."""
    summe = 0
    for i in range(0, len(vec)):
        summe += math.pow(abs(vec[i]), p)
    return math.pow(summe/len(vec), (1.0/p))


def get_cartesian_rms(sdf_string1, sdf_string2, removeHs=False):

    COM1 = get_centre_of_mass_from_sdf(sdf_string1, removeHs = True)
    COM2 = get_centre_of_mass_from_sdf(sdf_string2, removeHs = True)
    """Return the optimal RMS after aligning two structures."""

    coords1 = coords_and_masses_from_sdf(sdf_string1, removeHs = True)[:,:3] - COM1
    coords2 = coords_and_masses_from_sdf(sdf_string2, removeHs = True)[:,:3] - COM2
    '''Kabsh'''
    A = np.dot(coords1.T, coords2)
    V, S, W = np.linalg.svd(A)
    if np.linalg.det(np.dot(V, W)) < 0.0:
        V[:, -1] = -V[:, -1]
        K = np.dot(V, W)
    else:
        K = np.dot(V, W)

    coords1 = np.dot(coords1, K)
    rmsd_kabsh = 0.0
    for v, w in zip(coords1, coords2):
        rmsd_kabsh += sum([(v[i] - w[i])**2.0 for i in range(len(coords1[0]))])
    #~ print 'Kabsh rmsd  {}'.format(np.sqrt(rmsd_kabsh/len(coords1)))
    #~ #rmsd_type="internal_coord"

    #~ ref = Chem.MolFromMolBlock(sdf_string1, removeHs=True)
    #~ probe = Chem.MolFromMolBlock(sdf_string2, removeHs=True)
    #~ rms = AllChem.GetBestRMS(ref, probe)
    #~ print 'OLD ONE RMS {}\n'.format(rms)
    return np.sqrt(rmsd_kabsh/len(coords1))

def lowest_cartesian(string1, string2, **linked_strings):
    """Select lowest Cartesian RMS for two structures (for nonchiral and
    previously optimized structures)."""
    values = []
    get_cartesian_rms(string1, string2)
    values.append(get_cartesian_rms(string1, string2))
    if linked_strings:
        for string in linked_strings:
            values.append(get_cartesian_rms(string1, string))

    return min(values)


def find_one_in_list(sum_array, list_to_search):
    """Generate a random number and return the corresponding index from a
    list. See the description of the method find_two_in_list."""
    nparray_to_search = np.array(list_to_search)
    rn = sum_array*np.random.rand()
    found = False
    index = 0
    while not found:
        if rn <= nparray_to_search[:index+1].sum(axis=0):
            found = True
        else:
            index += 1
    return index

def adjusted_flag(population):
    """ Checks all the geometris in list of structures.
    Reduces flag in order all geometry could pass the test for clashes.
    Especially needed after relaxation of the structures."""
    check = True
    flag = 1.0
    while check:
        check = False
        for structure in population:
            if not structure.is_geometry_valid(flag = flag):
                if flag >= 0.805:
                    flag -= 0.005
                else:
                    raise Exception('Something wrong with criteria.')
                check = True
                continue
    return flag



def find_two_in_list(list_sum, nparray_to_search):
    """A numpy array is mapped to a segment of a line which length is equal to
    1. The lengths of the segments are proportional to the corresponding numpy
    array values. Next, two random numbers between 0 and 1 are generated and
    the segments containing these random numbers are returned."""
    rn1 = list_sum*np.random.rand()
    found1 = False
    index1 = 0
    while not found1:
        if rn1 < nparray_to_search[:index1+1].sum(axis=0):
            found1 = True
        else:
            index1 += 1
    equal = True
    while equal:
        rn2 = list_sum*np.random.rand()
        found2 = False
        index2 = 0
        while not found2:
            if rn2 < nparray_to_search[:index2+1].sum(axis=0):
                found2 = True
            else:
                index2 += 1
        if index2 != index1:
            equal = False
    return index1, index2


def find_closest(numb, list_of_values, periodic=False):
    """For a given number, return the closest value(s) from a given list"""
    all_dist = []
    for value in list_of_values:
        if periodic:
            all_dist.append(min(abs(numb-value), (360-abs(numb-value))))
        else:
            all_dist.append(abs(numb-value))
    m = min(all_dist)
    closest_ind = [i for i, j in enumerate(all_dist) if j == m]
    closest = []
    for ind in closest_ind:
        closest.append(list_of_values[ind])
    return closest


def distance(x, y):
    """"Calculate distance between two points in 3D."""
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2)


def check_geo_sdf(sdf_string, flag):
    """Check geometry from a sdf_string for clashes.
    Returns:
        True for clash-free geometries and False for invalid geometries
    Raises:
        ValueError: if distance cutoffs are non-positive
    """
    atoms, bonds = get_ind_from_sdfline(sdf_string.split('\n')[3])
    coordinates = np.zeros((atoms, 3))
    bonds_list = []
    atoms_names = []
    for i in range(4, atoms+4):
        coordinates[i-4][0:3] = sdf_string.split('\n')[i].split()[0:3]
        atoms_names.append(sdf_string.split('\n')[i].split()[3])
    for i in range(atoms+4, atoms+bonds+4):
        e1, e2 = get_ind_from_sdfline(sdf_string.split('\n')[i])
        bonds_list.append([e1, e2])
    dist = np.zeros((atoms, atoms))
    for x in range(atoms):
        for y in range(x, atoms):
            dist[x][y] = distance(np.array(coordinates[x]),
                                  np.array(coordinates[y]))
            dist[y][x] = dist[x][y]
    check = True
    # k = 0
    for x in range(atoms):
        for y in range(x+1, atoms):
            if [x+1, y+1] not in bonds_list and [y+1, x+1] not in bonds_list:
                # if dist[x][y] < 0.6*(VDW_radii[atoms_names[x]]) or dist[x][y] < 0.7*(VDW_radii[atoms_names[y]]):
                #     k += 1
                    # print 'Bond between {} and {}'.format(x+1, y+1)
                if dist[x][y] < VDW_radii[atoms_names[x]]*bohrtoang*flag or dist[x][y] < VDW_radii[atoms_names[y]]*bohrtoang*flag:
                    check = False
                    return check
    # print 'Number of bonds is {}'.format(k)
    return check

def check_geo_if_not_too_far(sdf_string, constrained_geom_file, flag=1.5):
    """Check geometry from a sdf_string if it is not too far from surrounding.
    Returns:
        True if it is not too far
    """
    check = False
    molecule = sdf2xyz_list(sdf_string)
    constrained = aims2xyz_vdw(constrained_geom_file)
    for x in molecule:
        for y in constrained:
            if np.linalg.norm(x[1:]-y[1:]) < (x[0] + y[0])*flag:
                check = True
    return check

def check_geo_sdf_after_crossover(sdf_string, flag=0.8):
    """Check geometry from a sdf_string for clashes after crossover.

    Returns:
        True for clash-free geometries and False for invalid geometries
    Raises:
        ValueError: if distance cutoffs are non-positive
    'flag' is a positive value that reduces the criteria.
    """
    atoms, bonds = get_ind_from_sdfline(sdf_string.split('\n')[3])
    coordinates = np.zeros((atoms, 3))
    bonds_list = []
    atoms_names = []
    for i in range(4, atoms+4):
        coordinates[i-4][0:3] = sdf_string.split('\n')[i].split()[0:3]
        atoms_names.append(sdf_string.split('\n')[i].split()[3])
    for i in range(atoms+4, atoms+bonds+4):
        e1, e2 = get_ind_from_sdfline(sdf_string.split('\n')[i])
        bonds_list.append([e1, e2])
    dist = np.zeros((atoms, atoms))
    for x in range(atoms):
        for y in range(x, atoms):
            dist[x][y] = distance(np.array(coordinates[x]),
                                  np.array(coordinates[y]))
            dist[y][x] = dist[x][y]
    check = True
    for x in range(atoms):
        for y in range(x+1, atoms):
            if [x+1, y+1] not in bonds_list and [y+1, x+1] not in bonds_list:
                if dist[x][y] < VDW_radii[atoms_names[x]]*bohrtoang*flag or dist[x][y] < VDW_radii[atoms_names[y]]*bohrtoang*flag:
                    check = False
                    return check
    return check


def get_ind_from_sdfline(sdf_line):
    """Extract the indicies from the sdf string (for molecules with more than
    99 atoms)"""
    l = len(sdf_line.split()[0])
    if l < 4:
        ind1 = int(sdf_line.split()[0])
        ind2 = int(sdf_line.split()[1])
    else:
        list_ind = list(sdf_line.split()[0])
        if len(list_ind) == 5:
            ind1 = int(list_ind[0]+list_ind[1])
            ind2 = int(list_ind[2]+list_ind[3]+list_ind[4])
        if len(list_ind) == 6:
            ind1 = int(list_ind[0]+list_ind[1]+list_ind[2])
            ind2 = int(list_ind[3]+list_ind[4]+list_ind[5])

    return ind1, ind2

def update_coords_sdf(sdf_string, new_coords):
    updated_sdf_string  = ''
    k = 0
    for i in sdf_string.split('\n'):
        old_coords_found = re.match(r'(\s+.?\d+\.\d+\s+.?\d+\.\d+\s+.?\d+\.\d+(\s+\w+.+))', i)
        if old_coords_found:
            updated_sdf_string = updated_sdf_string + '{:10.4f}{:10.4f}{:10.4f}{}\n'.format(new_coords[k][0], new_coords[k][1], new_coords[k][2], old_coords_found.group(2))
            k+=1
        else:
            updated_sdf_string = updated_sdf_string + i + '\n'
    return updated_sdf_strings

def update_coords_aims(aims_file, new_coords):
    i = 0
    full_path = os.path.realpath(aims_file)
    path, filename = os.path.split(full_path)
    aims = open(aims_file, 'r')
    lines = aims.readlines()
    temp = open(os.path.join(path, 'temp.in'), 'w')
    for line in lines:
        coord = re.match(r'(.*(atom)\s+(.\d+\.\d+)\s+(.\d+\.\d+)\s+(.\d+\.\d+)\s+(\w+))', line)
        if coord:
            temp.write('{}   {}   {}   {}   {}\n'.format(coord.group(2), new_coords[i][0], new_coords[i][1], new_coords[i][2], coord.group(6)))
            i += 1
        else:
            temp.write(line)
    temp.close()
    aims.close()
    shutil.move(os.path.join(path, 'temp.in'), aims_file)

def get_cm(coords_and_masses):
    center_of_mass = np.average(coords_and_masses[:,:3], axis=0, weights=coords_and_masses[:,3])
    return center_of_mass

def align_to_origin(aims_file):
    coords = aims2xyz_masses(aims_file)
    if len(aims2xyz_masses(aims_file)) == 1:
        update_coords_aims(aims_file, np.array([[0.0,0.0,0.0]]))
    if len(aims2xyz_extended(aims_file)) > 1:
        center = get_cm(aims2xyz_masses(aims_file))
        new_coords = np.array(aims2xyz_masses(aims_file)[:,:3]) - center
        update_coords_aims(aims_file, new_coords)

def check_for_clashes(sdf_string, constrained_geom_file):
    check = True
    if len(aims2xyz_vdw(constrained_geom_file)) < 1:
        return check
    molecule = sdf2xyz_list(sdf_string)
    constrained = aims2xyz_vdw(constrained_geom_file)
    for x in molecule:
        for y in constrained:
            if np.linalg.norm(x[1:]-y[1:]) < x[0] or np.linalg.norm(x[1:]-y[1:]) < y[0]:
                check = False
    return check

# Format conversions

def aims2xyz_masses(aims_file):
    xyz_coords = []
    with open(aims_file, 'r') as aims:
        lines = aims.readlines()
        for line in lines:
            atoms = re.match(r'(.*atom\s+(.\d+\.\d+)\s+(.\d+\.\d+)\s+(.\d+\.\d+)\s+(\w+))', line)
            if atoms:
                xyz_coords.append([float(atoms.group(2)), float(atoms.group(3)), float(atoms.group(4)), atoms.group(5)])
    aims.close()
    xyz_coords_array = np.array([ np.array([i[0], i[1], i[2], float(atom_masses[i[3]]) ]) for i in xyz_coords])
    return xyz_coords_array

def aims2xyz(aims_file):
    xyz_coords = []
    with open(aims_file, 'r') as aims:
        lines = aims.readlines()
        for line in lines:
            atoms = re.match(r'(.*atom\s*?(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(.?\d+\.\d+)\s+(\w+).*?)', line)
            if atoms:
                xyz_coords.append([str(atoms.group(5)), float(atoms.group(2)), float(atoms.group(3)), float(atoms.group(4))])
    aims.close()
    xyz_coords_array = np.array([ np.array([i[0], i[1], i[2], i[3]]) for i in xyz_coords])
    return xyz_coords_array

def aims2xyz_vdw(aims_file):
    xyz_coords = []
    with open(aims_file, 'r') as aims:
        lines = aims.readlines()
        for line in lines:
            atoms = re.match(r'(.*?atom\s*?(.\d+\.\d+)\s*?(.\d+\.\d+)\s*?(.\d+\.\d+)\s*?(\w+))', line)
            if atoms:
                xyz_coords.append([str(atoms.group(5)), float(atoms.group(2)), float(atoms.group(3)), float(atoms.group(4))])
    aims.close()
    xyz_coords_array = np.array([ np.array([VDW_radii[i[0]]*bohrtoang, i[1], i[2], i[3]]) for i in xyz_coords])
    return xyz_coords_array

def aims2xyz_extended(aims_file): # returns [coord_1, coord_2, coord_3, Atom_symbol, Atom_mass, Atom_VDW_radii]
    xyz_coords = []
    with open(aims_file, 'r') as aims:
        lines = aims.readlines()
        for line in lines:
            atoms = re.match(r'(.*atom\s+(.\d+\.\d+)\s+(.\d+\.\d+)\s+(.\d+\.\d+)\s+(\w+))', line)
            if atoms:
                xyz_coords.append([str(atoms.group(5)), float(atoms.group(2)), float(atoms.group(3)), float(atoms.group(4))])
    aims.close()
    xyz_coords_array = [[i[1], i[2], i[3], i[0], atom_masses[i[0]], VDW_radii[i[0]]*bohrtoang] for i in xyz_coords]
    return xyz_coords_array

def sdf2xyz_list(sdf_string):
    xyz_list = []
    for line in sdf_string.split('\n'):
        coords_found = re.match(r'(\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(\w+)\s+)', line)
        if coords_found:
            xyz_list.append(np.array([float(VDW_radii[coords_found.group(5)]*bohrtoang),
                                    float(coords_found.group(2)),
                                    float(coords_found.group(3)),
                                    float(coords_found.group(4))]))
    return np.array(xyz_list)

def sdf2aims(sdf_string):
    """Convert a sdf string to a aims string."""
    atoms = get_ind_from_sdfline(sdf_string.split('\n')[3])[0]
    coord = []
    for i in range(4, 4+atoms):
        x = float(sdf_string.split('\n')[i].split()[0])
        y = float(sdf_string.split('\n')[i].split()[1])
        z = float(sdf_string.split('\n')[i].split()[2])
        name = sdf_string.split('\n')[i].split()[3]
        coord.append('%s%10.4f%10.4f%10.4f%4s' % ('atom', x, y, z, name))
        coord.append('\n')
    aims_string = ''.join(coord)
    return aims_string


def sdf2xyz(sdf_string):
    """Convert a sdf_string to a xyz_list."""
    xyz_list = []
    for line in sdf_string.split('\n'):
        coords_found = re.match(r'(\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(.?\d+\.\d+)\s*(\w+)\s*?)', line)
        if coords_found:
            xyz_list.append([coords_found.group(5),
                            float(coords_found.group(2)),
                            float(coords_found.group(3)),
                            float(coords_found.group(4))])
    return xyz_list


def aims2sdf(aims_string, sdf_template_string):
    """Convert a aims string to a sdf string. Template for the sdf string is
    required."""
    atoms = len(aims_string.splitlines())
    sdf_form = sdf_template_string.splitlines()
    c = []
    cnt = 0
    for i in range(len(sdf_form)):
        if i > 3 and i < 4+atoms:
            line = sdf_form[i].split()
            line[0] = aims_string.split()[5*cnt+1]
            line[1] = aims_string.split()[5*cnt+2]
            line[2] = aims_string.split()[5*cnt+3]
            cnt += 1
            c.append('%10.4f%10.4f%10.4f%s%-2s' % (float(line[0]),
                                                   float(line[1]),
                                                   float(line[2]), str(' '),
                                                   line[3]))
            for j in xrange(4, len(line)):
                if j == 4:
                    c.append('%3d' % int(line[j]))
                elif j == len(line)-1:
                    c.append('%3d\n' % int(line[j]))
                else:
                    c.append('%3d' % int(line[j]))
        else:
            c.append(''.join(sdf_form[i])+'\n')

    sdf_string = ''.join(c)
    return sdf_string

def xyz2sdf(xyz_string, sdf_template_string):
    """Convert a xyz string to a sdf string. Template for the sdf string is
    required."""
    arr = xyz_string.splitlines()
    atoms = int(arr[0].split()[0])
    xyz_string_cut = '\n'.join(arr[2:])
    sdf_form = sdf_template_string.splitlines()
    c = []
    cnt = 0
    for i in range(len(sdf_form)):
        if i > 3 and i < 4+atoms:
            line = sdf_form[i].split()
            line[0] = xyz_string_cut.split()[4*cnt+1]
            line[1] = xyz_string_cut.split()[4*cnt+2]
            line[2] = xyz_string_cut.split()[4*cnt+3]
            cnt += 1
            c.append('%10.4f%10.4f%10.4f%s%-2s' % (float(line[0]),
                                                   float(line[1]),
                                                   float(line[2]), str(' '),
                                                   line[3]))
            for j in xrange(4, len(line)):
                if j == 4:
                    c.append('%3d' % int(line[j]))
                elif j == len(line)-1:
                    c.append('%3d\n' % int(line[j]))
                else:
                    c.append('%3d' % int(line[j]))
        else:
            c.append(''.join(sdf_form[i])+'\n')

    sdf_string = ''.join(c)
    return sdf_string


def mirror_sdf(sdf_string):
    """Mirror the geometry from a sdf string. Return a new sdf string."""
    atoms = get_ind_from_sdfline(sdf_string.split('\n')[3])[0]
    sdf_form = sdf_string.splitlines()
    c = []
    cnt = 0
    for i in range(len(sdf_form)):
        if i > 3 and i < 4+atoms:
            line = sdf_form[i].split()
            line[0] = -1.0*float(line[0])
            line[1] = -1.0*float(line[1])
            line[2] = -1.0*float(line[2])
            cnt += 1
            c.append('%10.4f%10.4f%10.4f%s%-2s' % (float(line[0]),
                                                   float(line[1]),
                                                   float(line[2]), str(' '),
                                                   line[3]))
            for j in xrange(4, len(line)):
                if j == 4:
                    c.append('%3d' % int(line[j]))
                elif j == len(line)-1:
                    c.append('%3d\n' % int(line[j]))
                else:
                    c.append('%3d' % int(line[j]))
        else:
            c.append(''.join(sdf_form[i])+'\n')
    mirror_sdf_string = ''.join(c)
    return mirror_sdf_string
