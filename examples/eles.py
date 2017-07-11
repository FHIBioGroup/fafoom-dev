#!/usr/bin/python

import os, re

with open('new.txt', 'a') as new:
    new.write('atom_masses = {')
    with open('elem.txt', 'r') as sdf:
	lines = sdf.readlines()
	for line in lines:
	    asd = re.match(r'((\d+\.\d+)\s+(\w+))', line)
	    if asd:
		new.write('\'{}\':  {}, '.format(asd.group(3), float(asd.group(2))))
		
    new.write('}')
new.close()
sdf.close()
