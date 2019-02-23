import sys
sys.path.append('')

import numpy as np

from LinearOptimization.kruskal import kruskal_mst


incidence_matrix = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1]
])

edge_costs = np.array([2, 8, 2, 4, 1, 4, 8, 2, 2])

used_path, total_cost = kruskal_mst(incidence_matrix=incidence_matrix, edge_costs=edge_costs)
print(
    f'''
Used edges are {used_path}.
Total cost of Minimum spanning tree {total_cost}.
''')