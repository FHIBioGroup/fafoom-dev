#    Copyright 2015 Adriana Supady & Mateusz Marianski
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
"""Measure and set dihedral angles and rings."""
from __future__ import division
from operator import itemgetter
import numpy as np

import os, sys, re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from numpy.linalg import inv
from utilities import get_vec, tor_rmsd, xyz2sdf, sdf2xyz, coords_and_masses_from_sdf, update_coords_sdf

def ig(x):
    return itemgetter(x)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if np.linalg.norm(vector) == 0.:
        vector = np.array([0.,0.,0.0000000001]) #Not to forget to check again this section
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    '''Returns angle between two vectors'''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))*180./np.pi

def translate(point, coord):
    translated = coord[:] - point[:]
    return translated

def translate_back(point, coord):
    translated = coord[:] + point[:]
    return translated

def mult_quats(q_1, q_2):
    Q_q_2 = np.array([[q_2[0], q_2[1], q_2[2], q_2[3]],
                     [-q_2[1], q_2[0], -q_2[3], q_2[2]],
                     [-q_2[2], q_2[3], q_2[0], -q_2[1]],
                     [-q_2[3], -q_2[2], q_2[1], q_2[0]]])
    q_3 = np.dot(q_1, Q_q_2)
    return q_3

def unit_quaternion(q):
    ones = np.ones((1,4))
    ones[:,0] = np.cos(q[0]*np.pi/180/2)
    vec = np.array([q[1], q[2], q[3]])
    vec = unit_vector(vec)
    ones[:,1:] = vec*np.sin(q[0]*np.pi/180/2)
    quaternion = ones[0]
    return quaternion

def rotation_quat(coord, q):
    q = unit_quaternion(q)
    R_q = np.array([[1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                    [2*q[2]*q[1] + 2*q[0]*q[3], 1 - 2*q[3]**2 - 2*q[1]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
                    [2*q[3]*q[1] - 2*q[0]*q[2], 2*q[3]*q[2] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]])
    rotated = np.dot(R_q, coord.transpose())
    return rotated.transpose()

def Rotation(coord, point, quaternion):
    trans = translate(point, coord)
    rotate = rotation_quat(trans, quaternion)
    final = translate_back(point, rotate)
    return np.array(final)

def produce_quaternion(angle, vector):
    ones = np.ones((1,4))
    ones[:,0] = angle
    ones[:,1:] = unit_vector(vector[:])
    quaternion = ones[0]
    return quaternion

def produce_coords_and_masses(coords, masses):
	zeros = np.zeros((len(coords), 4))
	zeros[:,:3] = coords[:]
	zeros[:,3] = masses[:]
	return zeros

def quaternion_measure(sdf_string, atom_1_indx, atom_2_indx):
    coords_and_masses = coords_and_masses_from_sdf(sdf_string)
    orient_vec = unit_vector(coords_and_masses[atom_2_indx][:3] - coords_and_masses[atom_1_indx][:3])
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    masses = coords_and_masses[:,3]                                     # Obtain masses of the atoms.
    center = get_centre_of_mass(coords_and_masses)                      # Obtain center of mass of the molecule.
    inertia_tensor = get_tensor_of_inertia(coords_and_masses)           # Obtain inertia tensor.
    eigval_1 = get_eigens(inertia_tensor)[0]                            # Eigenvalues of the inertia tensor.
    eigvec_1 = get_eigens(inertia_tensor)[1].T                          # Eigenvectors for inertia tensor. In column-like style!!! Have to be TRANSPOSED!!!
    z_index = np.argmax(eigval_1)                                       # Choose index for eigenvector with highest eigenvalue. Will align it to z direction, so longest axes of molecule will be perependicular to z axis.
    x_index = np.argmin(eigval_1)                                       # Choose index for eigenvector with lowest eigenvalue. Will align it to z direction, so longest axes of molecule will be perependicular to z axis.
    if np.dot(unit_vector(eigvec_1[z_index]), orient_vec) < 0:
        eigvec_1[z_index] = -eigvec_1[z_index]
    ang_1 = angle_between(eigvec_1[z_index], z_axis)                    # Angle is in degrees!
    vec_1 = np.cross(eigvec_1[z_index], z_axis)                         # Vector to rotate around.
    quat_1 = produce_quaternion(ang_1, vec_1)                            # Produce unit quaternion for rotation, simply consists of angle and vector.
    rotated_1 = Rotation(coords_and_masses[:,:3], center, quat_1)       # Coordinates of the molecule after aligning perpendicular to z axis.
    new_coords = produce_coords_and_masses(rotated_1, masses)
    orient_vec_2 = unit_vector(new_coords[atom_2_indx][:3] - new_coords[atom_1_indx][:3])
    eigs_after = get_eigens(get_tensor_of_inertia(new_coords))[1].T
    if np.dot(unit_vector(eigs_after[x_index]), orient_vec_2) < 0:
        eigs_after[x_index] = -eigs_after[x_index]
    angle_x = angle_between(eigs_after[x_index], x_axis)
    if np.dot(np.cross(unit_vector(eigs_after[x_index]), x_axis), z_axis) > 0:
        angle_x[0,0] = -angle_x[0,0]
    quaternion_of_the_molecule = np.array([angle_x[0,0], eigvec_1[z_index, 0], eigvec_1[z_index, 1], eigvec_1[z_index, 2]])
    return quaternion_of_the_molecule

def quaternion_measure_coords(coords_and_masses, atom_1_indx, atom_2_indx):
    orient_vec = unit_vector(coords_and_masses[atom_2_indx][:3] - coords_and_masses[atom_1_indx][:3])
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    masses = coords_and_masses[:,3]                                     # Obtain masses of the atoms.
    center = get_centre_of_mass(coords_and_masses)                      # Obtain center of mass of the molecule.
    inertia_tensor = get_tensor_of_inertia(coords_and_masses)           # Obtain inertia tensor.
    eigval_1 = get_eigens(inertia_tensor)[0]                            # Eigenvalues of the inertia tensor.
    eigvec_1 = get_eigens(inertia_tensor)[1].T                          # Eigenvectors for inertia tensor. In column-like style!!! Have to be TRANSPOSED!!!
    z_index = np.argmax(eigval_1)                                       # Choose index for eigenvector with highest eigenvalue. Will align it to z direction, so longest axes of molecule will be perependicular to z axis.
    x_index = np.argmin(eigval_1)                                       # Choose index for eigenvector with lowest eigenvalue. Will align it to z direction, so longest axes of molecule will be perependicular to z axis.
    if np.dot(unit_vector(eigvec_1[z_index]), orient_vec) < 0:
        eigvec_1[z_index] = -eigvec_1[z_index]
    ang_1 = angle_between(eigvec_1[z_index], z_axis)                    # Angle is in degrees!
    vec_1 = np.cross(eigvec_1[z_index], z_axis)                         # Vector to rotate around.
    quat_1 = produce_quaternion(ang_1, vec_1)                            # Produce unit quaternion for rotation, simply consists of angle and vector.
    rotated_1 = Rotation(coords_and_masses[:,:3], center, quat_1)       # Coordinates of the molecule after aligning perpendicular to z axis.
    new_coords = produce_coords_and_masses(rotated_1, masses)
    orient_vec_2 = unit_vector(new_coords[atom_2_indx][:3] - new_coords[atom_1_indx][:3])
    eigs_after = get_eigens(get_tensor_of_inertia(new_coords))[1].T
    if np.dot(unit_vector(eigs_after[x_index]), orient_vec_2) < 0:
        eigs_after[x_index] = -eigs_after[x_index]
    angle_x = angle_between(eigs_after[x_index], x_axis)
    if np.dot(np.cross(unit_vector(eigs_after[x_index]), x_axis), z_axis) > 0:
        angle_x[0,0] = -angle_x[0,0]
    quaternion_of_the_molecule = np.array([angle_x[0,0], eigvec_1[z_index, 0], eigvec_1[z_index, 1], eigvec_1[z_index, 2]])
    return quaternion_of_the_molecule

def align_to_axes(coords_and_masses, atom_1_indx, atom_2_indx): #Will align...
    center = get_centre_of_mass(coords_and_masses)
    quaternion = quaternion_measure_coords(coords_and_masses, atom_1_indx, atom_2_indx)
    desired_dir = np.array([0, 0, 1])
    vec = np.cross(quaternion[1:], desired_dir)
    angle = angle_between(quaternion[1:], desired_dir)
    quat_1 = produce_quaternion(angle, vec)
    rotation_1 = Rotation(coords_and_masses[:,:3], center, quat_1)
    angle_2 = -quaternion[0]
    quat_2 = produce_quaternion(angle_2, np.array([0, 0, 1]))
    rotation_2 = Rotation(rotation_1, center, quat_2)
    rotated = produce_coords_and_masses(rotation_2, coords_and_masses[:,3])
    return rotated

def quaternion_set(sdf_string, quaternion_to_set, atom_1_indx, atom_2_indx):
    coords_and_masses = coords_and_masses_from_sdf(sdf_string)
    center = get_centre_of_mass(coords_and_masses)
    aligned = align_to_axes(coords_and_masses, atom_1_indx, atom_2_indx)
    first_rot = produce_quaternion(quaternion_to_set[0], np.array([0, 0, 1]))
    rotation_1 = Rotation(aligned[:,:3], center, first_rot)
    angle_2 = angle_between(np.array([0, 0, 1]), quaternion_to_set[1:])
    vec_2 = np.cross(np.array([0, 0, 1]), quaternion_to_set[1:])
    quat_2 = produce_quaternion(angle_2, vec_2)
    rotation_2 = Rotation(rotation_1, center, quat_2)
    updated_sdf_string = update_coords_sdf(sdf_string, rotation_2)
    return updated_sdf_string

def quaternion_set_coords(coords_and_masses, quaternion_to_set, atom_1_indx, atom_2_indx):
    center = get_centre_of_mass(coords_and_masses)
    aligned = align_to_axes(coords_and_masses, atom_1_indx, atom_2_indx)
    first_rot = produce_quaternion(quaternion_to_set[0], np.array([0, 0, 1]))
    rotation_1 = Rotation(aligned[:,:3], center, first_rot)
    angle_2 = angle_between(np.array([0, 0, 1]), quaternion_to_set[1:])
    vec_2 = np.cross(np.array([0, 0, 1]), quaternion_to_set[1:])
    quat_2 = produce_quaternion(angle_2, vec_2)
    rotation_2 = Rotation(rotation_1, center, quat_2)
    return produce_coords_and_masses(rotation_2, coords_and_masses[:,3])

def get_coords(sdf_string):
    coords_and_masses = coords_and_masses_from_sdf(sdf_string)
    return coords_and_masses[:,:3]

def get_coords_and_masses(sdf_string):
    coords_and_masses = coords_and_masses_from_sdf(sdf_string)
    return coords_and_masses

def get_centre_of_mass_from_sdf(sdf_string):
    coords_and_masses = coords_and_masses_from_sdf(sdf_string)
    center_of_mass = np.average(coords_and_masses[:,:3], axis=0, weights=coords_and_masses[:,3])
    return center_of_mass

def get_centre_of_mass(coords_and_masses):
    center_of_mass = np.average(coords_and_masses[:,:3], axis=0, weights=coords_and_masses[:,3])
    return center_of_mass

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
    return updated_sdf_string

def get_tensor_of_inertia(coords_and_masses):
	###Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC15493/pdf/pq000978.pdf ###
    center = get_centre_of_mass(coords_and_masses)
    Ixx = np.sum([coords_and_masses[:,3]*(coords_and_masses[:,1]**2 + coords_and_masses[:,2]**2)])  - (center[1]**2+center[2]**2)*np.sum([coords_and_masses[:,3]])
    Iyy = np.sum([coords_and_masses[:,3]*(coords_and_masses[:,0]**2 + coords_and_masses[:,2]**2)])  - (center[0]**2+center[2]**2)*np.sum([coords_and_masses[:,3]])
    Izz = np.sum([coords_and_masses[:,3]*(coords_and_masses[:,0]**2 + coords_and_masses[:,1]**2)])  - (center[0]**2+center[1]**2)*np.sum([coords_and_masses[:,3]])
    Ixy = -np.sum([coords_and_masses[:,3]*coords_and_masses[:,0]*coords_and_masses[:,1]])           + (center[0]*center[1])*np.sum([coords_and_masses[:,3]])
    Ixz = -np.sum([coords_and_masses[:,3]*coords_and_masses[:,0]*coords_and_masses[:,2]])           + (center[0]*center[2])*np.sum([coords_and_masses[:,3]])
    Iyz = -np.sum([coords_and_masses[:,3]*coords_and_masses[:,1]*coords_and_masses[:,2]])           + (center[1]*center[2])*np.sum([coords_and_masses[:,3]])
    Iyx = Ixy
    Izx = Ixz
    Izy = Iyz
    tensor_of_inertia = np.matrix([[Ixx, Ixy, Ixz],
                                   [Iyx, Iyy, Iyz],
                                   [Izx, Izy, Izz]])
    return tensor_of_inertia

def get_eigens(tensor_of_inertia):
    eigens = np.linalg.eigh(tensor_of_inertia)
    return eigens

def centroid_measure(sdf_string):
    return get_centre_of_mass_from_sdf(sdf_string)

def centroid_set(sdf_string, values_to_set):
    atoms_list = coords_and_masses_from_sdf(sdf_string)[:,:3]
    new_coordinates = []
    shift = values_to_set - centroid_measure(sdf_string)
    new_coordinates = [i + shift for i in atoms_list]
    sdf_string = update_coords_sdf(sdf_string, new_coordinates)
    return sdf_string

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

def dihedral_measure(sdf_string, list_of_atoms):
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
    coords = coords_and_masses_from_sdf(sdf_string)[:,:3]
    plane1 = calculate_normal_vector(list_of_atoms[:3], coords)
    plane2 = calculate_normal_vector(list_of_atoms[1:], coords)
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
    ppoint = - np.dot(plane1, coords[list_of_atoms[0], :])
    dpoint = (np.dot(plane1, coords[list_of_atoms[3], :])+ppoint)/norm_plane1
    if dpoint >= 0:
        return -(alpha*180.0)/np.pi
    else:
        return (alpha*180.0)/np.pi

def construct_graph(sdf_string):
    conn_list = []
    for line in sdf_string.split('\n'):
    	bond_found  = re.match(r'(\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+$)', line)
    	if bond_found:
    	    conn_list.append([int(bond_found.group(2)), int(bond_found.group(3)), int(bond_found.group(4))])
    graph = {}
    for i in conn_list:
	if i[0] not in graph:
	    graph[i[0]] = [i[1]]
	else:
	    graph[i[0]].append(i[1])
    for i in conn_list:
	if i[1] not in graph:
	    graph[i[1]] = [i[0]]
	else:
	    graph[i[1]].append(i[0])
    return graph

def getRoots(aNeigh):
    def findRoot(aNode,aRoot):
        while aNode != aRoot[aNode][0]:
            aNode = aRoot[aNode][0]
        return (aNode,aRoot[aNode][1])
    myRoot = {}
    for myNode in aNeigh.keys():
        myRoot[myNode] = (myNode,0)
    for myI in aNeigh:
        for myJ in aNeigh[myI]:
            (myRoot_myI,myDepthMyI) = findRoot(myI,myRoot)
            (myRoot_myJ,myDepthMyJ) = findRoot(myJ,myRoot)
            if myRoot_myI != myRoot_myJ:
                myMin = myRoot_myI
                myMax = myRoot_myJ
                if  myDepthMyI > myDepthMyJ:
                    myMin = myRoot_myJ
                    myMax = myRoot_myI
                myRoot[myMax] = (myMax,max(myRoot[myMin][1]+1,myRoot[myMax][1]))
                myRoot[myMin] = (myRoot[myMax][0],-1)
    myToRet = {}
    for myI in aNeigh:
        if myRoot[myI][0] == myI:
            myToRet[myI] = []
    for myI in aNeigh:
        myToRet[findRoot(myI,myRoot)[0]].append(myI)
    return myToRet

def insertBreak(graph, atom1, atom2):
    graph[atom1].pop(graph[atom1].index(atom2))
    graph[atom2].pop(graph[atom2].index(atom1))
    return graph

def carried_atoms(sdf_string, positions):
    ''' Returns list of carried atoms'''
    graph = construct_graph(sdf_string)
    graph_with_break = insertBreak(graph, positions[1]+1, positions[2]+1)
    if positions[2] in getRoots(graph_with_break).values()[0]:
        return getRoots(graph_with_break).values()[0]
    else:
        return getRoots(graph_with_break).values()[1]



def dihedral_set(sdf_string, position, value):
    """ Set the dihedral angle.

    Args:
        sdf_string (string):
        position (list): 4 atoms defining the dihedral
        value : value to set
    Returns:
        modified sdf_string
    Raises:
        ValueError: If the lenght of the list is not equal 4.
    """
    if len(position) != 4:
        raise ValueError("The position needs to be defined by 4 integers")

    coords = coords_and_masses_from_sdf(sdf_string)[:,:3]
    atoms_to_carry = carried_atoms(sdf_string, position)
    angle_old = dihedral_measure(sdf_string, position)
    ang_to_rotate = angle_old - value
    vector = coords[position[2]] - coords[position[1]]
    quat = produce_quaternion(ang_to_rotate, vector)
    new_coords = []
    for i in range(len(coords)):
        if i+1 in atoms_to_carry:
            new_coords.append(Rotation(coords[i], coords[position[1]], quat))
        else:
            new_coords.append(coords[i])

    return update_coords_sdf(sdf_string, new_coords)

def pyranosering_set(sdf_string, position, new_dih, new_ang):
    """ Set the pyranosering.

    Args:
        sdf_string (string)
        position (list): 7 atoms defining the ring, i.e. positions of
                        ['C0','C1','C2','C3','C4','O', 'O0']
        new_dih (list) : 5 values for the dihedral angles
        new_ang (list): 5 values for the bond angles
    Returns:
        modified sdf_string
    Raises:
        ValueError: If the lenght of the position is not equal 7 ot if the
        length of new_dih/new_ang is not equal to 5.
    """
    if len(position) != 7:
        raise ValueError("The position needs to be defined by 7 integers")
    if len(new_dih) != 5:
        raise ValueError("Five dihedral angles are needed for the new ring "
                         "conformation.")
    if len(new_ang) != 5:
        raise ValueError("Five bond angles are needed for the new ring "
                         "conformation.")

    from scipy.linalg import expm3

    atoms_ring = {}
    for n, name in zip(range(len(position)),
                       ['C0', 'C1', 'C2', 'C3', 'C4', 'O', 'O0']):
        atoms_ring[name] = position[n]

    def initialize(sdf_string):
        molecule = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        return molecule


    def measure_angle(list_of_atoms, xyz):
        """Calculate an angle between three atoms:
        angle = acos(dot(X,Y)/(norm(X)*norm(Y)))

        Args:
            list_of_atoms: list of 3 atoms
            xyz: numpy array with atoms xyz positions
        """

        r0 = xyz[list_of_atoms[0], :] - xyz[list_of_atoms[1], :]
        r1 = xyz[list_of_atoms[2], :] - xyz[list_of_atoms[1], :]

        norm_r0 = np.sqrt(np.sum(r0**2))
        norm_r1 = np.sqrt(np.sum(r1**2))
        norm = norm_r0*norm_r1

        dot_product = np.dot(r0, r1)/norm
        angle = np.arccos(dot_product)

        #Calculate the axis of rotation (axor):
        axor = np.cross(r0, r1)

        return angle*180.0/np.pi, axor

    # def measure_dihedral(list_of_atoms, xyz):
    #     """Calculate a dihedral angle between two planes defined by
    #     a list of four atoms. It returns the angle and the rotation axis
    #     required to set a new dihedral.
    #
    #     Args:
    #         list_of_atoms: list of 4 atoms
    #         xyz: numpy array with atom xyz positions
    #     """
    #
    #     plane1 = calculate_normal_vector(list_of_atoms[:3], xyz)
    #     plane2 = calculate_normal_vector(list_of_atoms[1:], xyz)
    #     #Calculate the axis of rotation (axor)
    #     axor = np.cross(plane1, plane2)
    #
    #     #Calculate a norm of normal vectors:
    #     norm_plane1 = np.sqrt(np.sum(plane1**2))
    #     norm_plane2 = np.sqrt(np.sum(plane2**2))
    #     norm = norm_plane1 * norm_plane2
    #     #Measure the angle between two planes:
    #     dot_product = np.dot(plane1, plane2)/norm
    #     alpha = np.arccos(dot_product)
    #
    #     #The cosine function is symetric thus, to distinguish between
    #     #negative and positive angles, one has to calculate if the fourth
    #     #point is above or below the plane defined by first 3 points:
    #     ppoint = - np.dot(plane1, xyz[list_of_atoms[0], :])
    #     dpoint = (np.dot(plane1, xyz[list_of_atoms[3], :])+ppoint)/norm_plane1
    #     if dpoint >= 0:
    #         return -(alpha*180.0)/np.pi, axor
    #     else:
    #         return (alpha*180.0)/np.pi, axor

    def determine_carried_atoms(at1, at2, conn_mat):
        """Find all atoms necessary to be carried over during rotation
        of an atom 2:

        Args:
            at1, at2: two atoms number
        """
        #1. Zero the connections in connectivity matrix
        tmp_conn = np.copy(conn_mat)
        tmp_conn[at1, at2] = 0
        tmp_conn[at2, at1] = 0
        #2. Determine the connected atoms:
        carried_atoms = [at2]
        s = True
        while s:
            s = False
            #Always iterate over entire list because I might have branching
            for at in carried_atoms:
                #List of indexes of connected atoms:
                conn_atoms = np.where(tmp_conn[at] != 0)[0]
                conn_atoms.tolist
                for x in conn_atoms:
                    if x not in carried_atoms:
                        carried_atoms.append(x)
                        s = True
        return carried_atoms

    def set_angle(list_of_atoms, new_ang, atoms_ring, xyz, conn_mat):
        """Set a new angle between three atoms

        Args:
            list_of_atoms: list of three atoms
            new_ang: value of dihedral angle (in degrees) to be set
            atoms_ring: dictionary of atoms in the ring. It recognizes
                        if the last atom is 'C0O' (obsolete)
            xyz: numpy array with atoms xyz positions
            conn_mat: connectivity matrix
        Returns:
            xyz: modified numpy array with new atoms positions
        """
        #Determine the axis of rotation:
        old_ang, axor = measure_angle(list_of_atoms, xyz)
        norm_axor = np.sqrt(np.sum(axor**2))
        normalized_axor = axor/norm_axor

        #Determine which atoms should be dragged along with the bond:
        carried_atoms = determine_carried_atoms(list_of_atoms[1],
                                                list_of_atoms[2], conn_mat)

        #Each carried_atom is rotated by euler-rodrigues formula:
        #Also, I move the midpoint of the bond to the mid atom
        #the rotation step and then move the atom back.

        rot_angle = np.pi*(new_ang - old_ang)/180.
        #Shake it, baby! Rotation matrix:
        #print old_ang, new_ang, rot_angle*180./np.pi
        rot1 = expm3(np.cross(np.eye(3), normalized_axor*rot_angle))
        translation = xyz[list_of_atoms[1], :]
        for at in carried_atoms:
            xyz[at, :] = np.dot(rot1, xyz[at, :]-translation)
            xyz[at, :] = xyz[at, :]+translation
        return xyz


    def set_dihedral(list_of_atoms, new_dih, atoms_ring, xyz, conn_mat):
        """Set a new dihedral angle between two planes defined by
        atoms first and last three atoms of the supplied list.

        Args:
            list_of_atoms: list of four atoms
            new_dih: value of dihedral angle (in degrees) to be set
            atoms_ring: dictionary of atoms in the ring. It recognizes
                       if the last atom is 'C0O'
            xyz: numpy array with atoms xyz positions
            conn_mat: connectivity matrix
        Returns:
            xyz: modified numpy array with new atoms positions
        """

        #Determine the axis of rotation:
        old_dih, axor = measure_dihedral(list_of_atoms, xyz)
        norm_axor = np.sqrt(np.sum(axor**2))
        normalized_axor = axor/norm_axor

        #Check if the bond is the last bond, next to broken one.
        #If yes, refer to the oxygen:
        if 'O0a' in atoms_ring.keys():
            if list_of_atoms[-1] == atoms_ring['O0a']:
                new_dih += 120.0
        else:
            if list_of_atoms[-1] == atoms_ring['O0b']:
                new_dih -= 120.0
        #Determine which atoms should be dragged along with the bond:
        carried_atoms = determine_carried_atoms(list_of_atoms[1],
                                                list_of_atoms[2], conn_mat)
        #Each carried_atom is rotated by Euler-Rodrigues formula:
        #Reverse if the angle is less than zero, so it rotates in
        #right direction.
        #Also, I move the midpoint of the bond to the center for
        #the rotation step and then move the atom back.

        if old_dih >= 0.0:
            rot_angle = np.pi*(new_dih - old_dih)/180.
        else:
            rot_angle = -np.pi*(new_dih - old_dih)/180.
        #Shake it, baby! Rotation matrix:
        rot1 = expm3(np.cross(np.eye(3), normalized_axor*rot_angle))
        translation = (xyz[list_of_atoms[1], :]+xyz[list_of_atoms[2], :])/2
        for at in carried_atoms:
            xyz[at, :] = np.dot(rot1, xyz[at, :]-translation)
            xyz[at, :] = xyz[at, :]+translation

        return xyz

    def mutate_ring(molecule, new_dih, new_ang):
        """Mutate a ring to given conformation defined as a list of torsional
        angles accoring to the 10.1016/S0040-4020(00)01019-X (IUPAC) paper
        """
        n_at = molecule.GetNumAtoms()
        n_bonds = molecule.GetNumBonds()
        m_string = Chem.MolToMolBlock(molecule)

        #Split the string to xyz, connectivity matrix and atom list
        m_coords = m_string.split('\n')[4:4+n_at]
        xyz = np.zeros((n_at, 3))
        atom_list = []
        n = 0
        for line in m_coords:
            xyz[n, :] += np.array(map(float, line.split()[:3]))
            atom_list.append(line.split()[3])
            n += 1
        #Molecule Connectivity Matrix
        m_conn = m_string.split('\n')[4+n_at:4+n_at+n_bonds]
        conn_mat = np.zeros((n_at, n_at))
        for line in m_conn:
            at1 = int(line.split()[0])
            at2 = int(line.split()[1])
            conn_mat[at1-1, at2-1] = 1
            conn_mat[at2-1, at1-1] = 1

        #Introduce a cut between ring C0 and C1:
        #I chose these atoms according to the torsion
        #definitions in the IUPAC paper
        #doi: 10.1016/S0040-4020(00)01019-X
        conn_mat[atoms_ring['C0'], atoms_ring['C1']] = 0
        conn_mat[atoms_ring['C1'], atoms_ring['C0']] = 0

        #Construct a list of atoms in order:
        #C0, C1, C2, C3, C4, O, C0, O0a/b (oxygen at anomeric carbon)
        #I use this list to rotate bonds.
        atoms_list = []
        for x in range(0, 5):
            atoms_list.append(atoms_ring['C'+str(x)])
        atoms_list.append(atoms_ring['O'])
        atoms_list.append(atoms_ring['C0'])
        atoms_list.append(atoms_ring['O0'])

        #Determine the anomer - alpha/beta, based on improper
        #dihedral angle C1-C0-O-O0
        imdih = []
        for at in ['C1', 'C0', 'O', 'O0']:
            imdih.append(atoms_ring[at])
        test_anomer = measure_dihedral(imdih, xyz)[0]
        if test_anomer > 0.0:
            atoms_ring['O0b'] = atoms_ring.pop('O0')
        else:
            atoms_ring['O0a'] = atoms_ring.pop('O0')

        #Adjust the 'internal' angles in the ring:
        for n in range(len(new_ang)):
            xyz = set_angle(atoms_list[n:n+3], new_ang[n], atoms_ring, xyz,
                            conn_mat)
        #Rotate the dihedral angles in the ring:
        for n in range(len(new_dih)):
            xyz = set_dihedral(atoms_list[n:n+4], new_dih[n], atoms_ring, xyz,
                               conn_mat)
        a = []
        a.append("%10s\n" % n_at)
        for n in new_dih:
            a.append("%10.4f" % n)
        a.append("\n")
        for n in range(n_at):
            a.append("%10s%10.4f%10.4f%10.4f\n" % (atom_list[n], xyz[n, 0],
                                                   xyz[n, 1], xyz[n, 2]))
        xyz_string = ''.join(a)
        return xyz_string

    molecule = initialize(sdf_string)
    sdf_string = xyz2sdf(mutate_ring(molecule, new_dih, new_ang), sdf_string)

    return sdf_string


def pyranosering_measure(sdf_string, position, dict_of_options):
    """Assign the ring to a conformation from the dictionary of options.

    Args:
        sdf_string (string)
        position (list): 7 atoms defining the ring
        dict_of_options (dict) : options for the ring
    Returns:
        An integer that corresponds to the best matching dict key
    Raises:
        ValueError: If the lenght of the position is not equal 7.
    """
    if len(position) != 7:
        raise ValueError("The position needs to be defined by 7 integers")
    ang1 = dihedral_measure(sdf_string, position[0:4])
    ang2 = dihedral_measure(sdf_string, position[1:5])
    ang3 = dihedral_measure(sdf_string, position[2:6])
    ang4 = dihedral_measure(sdf_string, (ig(3)(position), ig(4)(position),
                                         ig(5)(position), ig(0)(position)))
    ang5 = dihedral_measure(sdf_string, (ig(4)(position), ig(5)(position),
                                         ig(0)(position), ig(1)(position)))
    ang6 = dihedral_measure(sdf_string, (ig(5)(position), ig(0)(position),
                                         ig(1)(position), ig(2)(position)))

    all_ang = [ang1, ang2, ang3, ang4, ang5, ang6]

    rmsd_dict = {}

    for key in dict_of_options:
        rmsd_dict[key] = (tor_rmsd(2, get_vec(all_ang, dict_of_options[key])))

    return int(min(rmsd_dict.iteritems(), key=ig(1))[0])



#~ smiles = '[NH3+][C@H](C(=O)N1[C@H](C(=O)N[C@H](C(=O)O)Cc2ccccc2)CCC1)Cc1nc[nH]c1'
#~ pat_1 = Chem.MolFromSmarts('[C@H]C(=O)O')
#~ pat_2 = Chem.MolFromSmarts('[NX3H2,NX4H3+][C@H]')
#~ mol = Chem.MolFromSmiles(smiles)
#~ mol = Chem.AddHs(mol)
#~ AllChem.EmbedMolecule(mol)
#~ string = Chem.MolToMolBlock(mol)

#~ first_heavy_atom_indx = 0                           #First heavy atom
#~ last_heavy_atom = mol.GetNumHeavyAtoms() - 1        #Last Heavy atom
#~ coords_and_masses = get_coords_and_masses(string)
#~ center = get_centre_of_mass(coords_and_masses)
#~ coords_and_masses = produce_coords_and_masses(Rotation(coords_and_masses[:,:3], center, np.array([45, 1, 1, 1])), coords_and_masses[:,3])
#~ string = update_coords_sdf(string, coords_and_masses[:,:3])
#~ if 'test.xyz' in os.listdir(os.getcwd()):
    #~ os.remove('test.xyz')
#~ initial = open('initial.xyz', 'w')
#~ initial.write(sdf2xyz(string))
#~ initial.close()

#~ aligned = align_to_axes(coords_and_masses, first_heavy_atom_indx, last_heavy_atom)
#~ cent = get_centre_of_mass(aligned)
#~ al = update_coords_sdf(string, aligned[:,:3])
#~ aligned_file = open('aligned.xyz', 'w')
#~ aligned_file.write(sdf2xyz(al))
#~ aligned_file.close()
#~ quat_set = np.array([0, 1, 1, 1])
#~ after_setting = quaternion_set_coords(coords_and_masses, quat_set, first_heavy_atom_indx, last_heavy_atom)
#~ sett = update_coords_sdf(string, after_setting[:,:3])
#~ setted = open('setted_111.xyz', 'w')
#~ setted.write(sdf2xyz(sett))
#~ setted.close()
