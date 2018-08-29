#!/usr/bin/python
from utilities import atom_masses, VDW_radii
from itertools import combinations
import math

def testFormat(mol):
    """ Returns format of file"""
    """Perfectly identifies single XYZ file"""

    # For now format is unknown
    FORMAT='IDK'

    def add_as_coord(line):
        coords=[float(i) for i in line[1:]]
        return [line[0], coords]

    def calcLinesAtoms(lines):
        """ Return number of atoms
            length of lines
            initial guess"""
        numAtoms=0
        COORDINATES=[]
        numLines = len(lines)
        guess = 'IDK'
        for line in lines:
            lineSplit = line.split()
            if lineSplit >= 4:
                floats, atomTypes=0,0
                for i in lineSplit:
                    try:
                        if float(i):
                            floats+=1
                    except:
                        if i in atom_masses:
                            atomTypes+=1
                if floats>=3 and atomTypes>=1:
                    numAtoms+=1
                    if len(lineSplit)==4:
                       COORDINATES.append(add_as_coord(lineSplit))
                       guess='XYZ'
        return numAtoms, numLines, guess, COORDINATES

    with open(mol) as f:
        lines = f.readlines()
        numAt, numLin, guess, COORDINATES = calcLinesAtoms(lines)
        if numLin - numAt == 2 and guess=='XYZ':
            FORMAT='XYZ'
    return FORMAT, COORDINATES

def takeConnectivity(coords):
    atoms=zip(*coords)[0]
    coord=zip(*coords)[1]

    atomCobinations =list(combinations(atoms, 2))
    coordCobinations=list(combinations(coord, 2))

    all_combinations = list(zip(atomCobinations, coordCobinations))

    def testConnect(element, coords):

        def take_vdWs(atoms):
            return max([VDW_radii[atoms[0][0]], VDW_radii[atoms[0][1]]])

        def take_length(atoms):
            p0, p1=atoms[1][0], atoms[1][1]
            return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)

        vdWs=[take_vdWs(i) for i in element]
        lengths=[take_length(i) for i in element]
        inxxx=list(combinations(range(len(coords)), 2))
        return [inxxx[i] for i in range(len(vdWs)) if vdWs[i]>=lengths[i]]
    return testConnect(all_combinations, coord)


def constructGraph(conn_list):
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


if __name__ == "__main__":
    testFormat(sys.argv[1])





