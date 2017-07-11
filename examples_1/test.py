#!/usr/bin/python
#-----------------------
from rdkit import Chem
from rdkit.Chem import AllChem
core = Chem.MolFromSmiles('c1cccc2c1nc(CC)cc2C(=O)O')

# first scale the core so that a single bond is 1.5A:
center = AllChem.ComputeCentroid(core.GetConformer())
import numpy
tf = numpy.identity(4,numpy.float)
tf[0][3] -= center[0]
tf[1][3] -= center[1]
tf[0][0] = tf[1][1] = tf[2][2] = 1.5
AllChem.TransformMol(core,tf)

m = Chem.MolFromSmiles('c1cccc2c1nc(CC)cc2C(=O)O')
from rdkit import Geometry
coords = [core.GetConformer().GetAtomPosition(x) for x in
range(core.GetNumAtoms())]
coords2D = [Geometry.Point2D(pt.x,pt.y) for pt in coords]
cd = {}
match = m.GetSubstructMatch(core)
for i,coord in enumerate(coords2D): cd[match[i]] = coord
AllChem.Compute2DCoords(m,coordMap=cd)
