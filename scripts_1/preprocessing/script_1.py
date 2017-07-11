import os, re
import numpy as np

path = os.getcwd()

def construct_graph(some_list):
    graph = {}
    for i in some_list:
	if i[0] not in graph:
	    graph[i[0]] = [i[1]]
	else:
	    graph[i[0]].append(i[1])
    for i in some_list:
	if i[1] not in graph:
	    graph[i[1]] = [i[0]]
	else:
	    graph[i[1]].append(i[0])
    return graph

conn_list = []
connectivity = {}

with open(os.path.join(path, 'mol.sdf'), 'r') as mol:
    lines = mol.readlines()
    for line in lines:
	#~ coord.found = re.match(r'()', line)
	bond_found  = re.match(r'(\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+$)', line)
	if bond_found:
	    conn_list.append([int(bond_found.group(2)), int(bond_found.group(3)), int(bond_found.group(4))])


graph = construct_graph(conn_list)
print conn_list
#~ def obtain_single_bonds(conn_list):
    #~ for i in conn_list:
    
def singles(graph):
    singles = []
    for key in graph.keys():
	if len(graph[key]) == 1:
	    singles.append(key)
    return singles
    
def triplets(graph, singles):
    triplets = []
    for key in graph.keys():
	if (len(graph[key])  == 3 and len(set(graph[key]) & set(singles)) == 0) or  (len(graph[key])  == 4 and len(set(graph[key]) & set(singles)) == 1):
	    triplets.append(key)
    return triplets

def douplets(graph, singles):
    douplets = []
    for key in graph.keys():
	if (len(graph[key])  == 2 and len(set(graph[key]) & set(singles)) == 0) or (len(graph[key])  == 3 and len(set(graph[key]) & set(singles)) == 1) or (len(graph[key])  == 4 and len(set(graph[key]) & set(singles)) == 2):
	    douplets.append(key)
    return douplets    

def atoms_in_rings(graph, triplets, singles):
    visited = []
    for i in triplets:
	visited.append(i)
	for k in graph[i]:
	    if k not in set(graph[i]) & set(singles):
		print k

def others(graph, singles, douplets, triplets):
    others = []
    for i in graph.keys():
	if i not in singles and i not in douplets and i not in triplets:
	    others.append(i)
    return others


singles = singles(graph)   
triplets = triplets(graph, singles)
douplets = douplets(graph, singles)
others = others(graph, singles, douplets, triplets)

#~ print singles
#~ print douplets
#~ print triplets
#~ print others

def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path+[next_state]))

def clear_loops(loop_list):
    temp_list = []
    clear_list = []
    for i in loop_list:
	temp_list.append(sorted(list(set(i))))
    for i in temp_list:
	if i not in clear_list:
	    clear_list.append(i)
    return clear_list


cycles = [[node]+path  for node in graph for path in dfs(graph, node, node)]

loops = []
for i in cycles:
    if len(i) > 3:
	loops.append(i)
atoms_in_rings = []
for i in clear_loops(loops):
    atoms_in_rings +=  i

print 'Termination atoms are: {}'.format(singles)
print 'Atoms in rings are: {}'.format(atoms_in_rings)













	    
