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
from __future__ import division
import math
from copy import copy
from random import choice
from rdkit import Chem

from utilities import ig, cleaner, get_vec, tor_rmsd, find_one_in_list
from measure import get_coords

from genetic_operations import mutation

from rdkit import Chem
from rdkit.Chem import AllChem

from operator import itemgetter
from rdkit.Chem import rdMolTransforms
from utilities import *

from rdkit.Chem import Draw
from rdkit.Chem import rdShapeHelpers


q_1 = np.array([0, 6, 1, 7.5])
q_2 = np.array([34, 1, 5, -45])

def translate(point, coord):
    matr = np.array([[1, 0, 0, -point[0]],
                      [0, 1, 0, -point[1]],
                      [0, 0, 1, -point[2]],
                      [0, 0, 0, 1]]) 
    ones = np.ones((coord.ndim, 4))
    ones[:,:3] = coord[:]
    translated = np.dot(matr, ones.transpose())
    return translated[:3].transpose()

def translate_back(point, coord):
    matr = np.array([[1, 0, 0, point[0]],
                      [0, 1, 0, point[1]],
                      [0, 0, 1, point[2]],
                      [0, 0, 0, 1]]) 
    ones = np.ones((coord.ndim, 4))
    ones[:,:3] = coord[:]
    translated = np.dot(matr, ones.transpose())
    return translated[:3].transpose()

def unit(vec):
    mag = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    unit = np.array([vec[0]/mag, vec[1]/mag, vec[2]/mag])
    return unit

def mult_quats(q_1, q_2):
    Q_q_2 = np.array([[q_2[0], q_2[1], q_2[2], q_2[3]],
                     [-q_2[1], q_2[0], -q_2[3], q_2[2]],
                     [-q_2[2], q_2[3], q_2[0], -q_2[1]],
                     [-q_2[3], -q_2[2], q_2[1], q_2[0]]])
    q_3 = np.dot(q_1, Q_q_2)
    return q_3

def rotation_quat(coord, q):
    q[0] = q[0]*np.pi/180
    v = unit([q[1], q[2], q[3]])
    q = np.array([np.cos(q[0]/2), np.sin(q[0]/2)*v[0], np.sin(q[0]/2)*v[1], np.sin(q[0]/2)*v[2]])
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
    
point = [0,0,100]    
q = [90, 0, 0, 1]
coord = np.array([[1, 0,0], [0, 1, 0]])
print Rotation(coord, point, q)


def atoms_central_bond_indx(string):
    mol = Chem.MolFromMolBlock(string, removeHs=False)
    heavy_bonds = []
    for i in range(mol.GetNumBonds()):
        if mol.GetBonds()[i].GetBeginAtom().GetSymbol() != 'H' and mol.GetBonds()[i].GetEndAtom().GetSymbol() != 'H':
            heavy_bonds.append(mol.GetBonds()[i])
    index = len(heavy_bonds)//2
    bond = heavy_bonds[index]
    
    start = bond.GetBeginAtomIdx()
    end = bond.GetEndAtomIdx()
    
    poss = mol.GetConformer().GetAtomPosition(start)
    pose = mol.GetConformer().GetAtomPosition(end)

    position_start = np.array([poss.x, poss.y, poss.z])
    position_end = np.array([pose.x, pose.y, pose.z])
    
    vector = position_end - position_start
    two_atoms_indx = [start, end]
    return two_atoms_indx


def central_point(sdf_string):
    mol = Chem.MolFromMolBlock(sdf_string, removeHs=True)
    pos = mol.GetConformer()
    centroid = rdMolTransforms.ComputeCentroid(pos, ignoreHs=True) 
    return np.array([centroid.x, centroid.y, centroid.z])
    
    
def longest_from_center(sdf_string):
    list_dist = []
    mol = Chem.MolFromMolBlock(sdf_string, removeHs=True)
    pos = mol.GetConformer()
    centr = central_point(sdf_string)
    for i in range(pos.GetNumAtoms()):
        list_dist.append(np.linalg.norm(np.array([pos.GetAtomPosition(i).x, pos.GetAtomPosition(i).y, pos.GetAtomPosition(i).x]) - centr))
    print list_dist
    index_of_farest_atom = list_dist.index(max(list_dist))
    return index_of_farest_atom








#~ #=======================================================================
'''
For test of the module only
'''
smiles = 'CC(=O)N[C@H](C(=O)NC)C'


mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
string = Chem.MolToMolBlock(mol)
#~ print 'Initial coordinates:'
#~ print sdf2xyz(string)

#~ #***********************************************************************

#~ print central_bond(m).GetBeginAtomIdx()
#~ print vector_from_bond(m, central_bond(m))




'''
Rotations around arbitrary line in the space.
'''
import numpy as np
from numpy.linalg import inv

def Tp(point):
    return np.matrix([[1, 0, 0, -point[0]],[0, 1, 0, -point[1]],[0, 0, 1, -point[2]],[0, 0, 0, 1]])  
def Rot_x(angle):
    return np.matrix([[1,0,0,0],[0, np.cos(angle*np.pi/180), -np.sin(angle*np.pi/180), 0],[0,np.sin(angle*np.pi/180),np.cos(angle*np.pi/180),0],[0,0,0,1]])
def Rot_y(angle):
    return np.matrix([[np.cos(angle*np.pi/180),0,np.sin(angle*np.pi/180),0],[0, 1, 0, 0],[-np.sin(angle*np.pi/180),0,np.cos(angle*np.pi/180),0],[0,0,0,1]])
def Rot_z(angle):
    return np.matrix([[np.cos(angle*np.pi/180), -np.sin(angle*np.pi/180), 0, 0], [np.sin(angle*np.pi/180), np.cos(angle*np.pi/180), 0 ,0],[0,0, 1,0],[0,0,0,1]])
def Txz(direction):
    u = direction[0]
    v = direction[1]
    w = direction[2]
    sqr = np.sqrt(u**2+v**2)
    if sqr == 0:
        return np.matrix([[1, 1 , 0,0],[1, 1,0,0],[0,0,1,0],[0,0,0,1]])
    else:
        return np.matrix([[u/sqr, v/sqr , 0,0],[-v/sqr, u/sqr,0,0],[0,0,1,0],[0,0,0,1]])
def Tz(direction):
    u = direction[0]
    v = direction[1]
    w = direction[2]
    sq = np.sqrt(u**2 + v**2)
    sqr = np.sqrt(u**2+v**2+w**2)
    if sqr == 0:
        return np.matrix([[1, 0, 1, 0],[0, 1, 0, 0],[1, 0, 1,0],[0,0,0,1]])
    else:
        return np.matrix([[w/sqr, 0, -sq/sqr, 0],[0, 1, 0, 0],[sq/sqr, 0, w/sqr,0],[0,0,0,1]])


def rotation(coord, point, direction, angle):
    matr = inv(Tp(point))*inv(Txz(direction))*inv(Tz(direction))*Rot_z(angle)*Tz(direction)*Txz(direction)*Tp(point)
    rotated = np.dot(np.array(matr), np.array([[coord[0]], [coord[1]], [coord[2]], [1]]))
    return rotated[:3]

point     = [4, 4, -4]
direction = [3, 1, 2]
coord     = [-3, -5, 0]
angle     = 10
#~ print  rotation(coord, point, direction, angle)

#~ for i in range(len(get_coords(string))):
    #~ print rotation(np.array(get_coords(string)[i]).tolist(), [0,0,0], [0,0,1], 90)










