#!/usr/bin/python
import os

with open('all_structures.xyz', 'w') as all_struc:
    for i in os.listdir(os.getcwd()):
        if os.path.isdir(i) and 'structure' in i:
            geom = os.path.join(i, 'geometry_out.xyz')
            if os.path.exists(geom):
                with open(geom) as gg:
                    lines=gg.read()
                    all_struc.write(lines)
