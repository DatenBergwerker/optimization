import numpy as np
from algorithms import LinearProgram, exhaustive_search

problems = [
    {'Description': 'Exercise 3.3, First Example',
     'Problem': LinearProgram(
        constraints=np.array([
            [2, 4, 1, 1, 0],
            [4, 6, 4, 0, 1]
        ]), constraint_values=np.array([
            [8],
            [6]
        ]), cost_vector=np.array([
            [1, 1, 1, 0, 0]
        ])
    )},
    {'Description': 'Exercise 3.3, LP 1',
     'Problem': LinearProgram(
        constraints=np.array([
            [-2, 1, 1, 0],
            [-1, 10, 0, 1]
        ]), constraint_values=np.array([
            [4],
            [135]
        ]), cost_vector=np.array([
            [1, 2, 0, 0]
        ])
    )},
    {'Description': 'Exercise 3.3, LP 2 (d = 3)',
     'Problem': LinearProgram(
         constraints=np.array([
             [-2, 1, 1, 0, 0, 0],
             [-1, 10, 0, 1, 0, 0],
             [1, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 1]
         ]), constraint_values=np.array([
             [4],
             [135],
             [8],
             [8]
         ]), cost_vector=np.array([
             [1, 1, 0, 0, 0, 0]
         ])
     )},
    {'Description': 'Exercise 3.3, LP 2 (d = 4)',
     'Problem': LinearProgram(
         constraints=np.array([
             [-2, 1, 1, 0, 0, 0],
             [-1, 10, 0, 1, 0, 0],
             [1, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 1]
         ]), constraint_values=np.array([
             [4],
             [135],
             [16],
             [16]
         ]), cost_vector=np.array([
             [1, 1, 0, 0, 0, 0]
         ])
     )},
    {'Description': 'Exercise 3.3, LP 3 (d = 3)',
     'Problem': LinearProgram(
         constraints=np.array([
             [-2, 1, 1, 0, 0, 0],
             [-1, 10, 0, 1, 0, 0],
             [1, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 1]
         ]), constraint_values=np.array([
             [4],
             [135],
             [9],
             [9]
         ]), cost_vector=np.array([
             [1, 1, 0, 0, 0, 0]
         ])
     )},
    {'Description': 'Exercise 3.3, LP 3 (d = 4)',
     'Problem': LinearProgram(
         constraints=np.array([
             [-2, 1, 1, 0, 0, 0],
             [-1, 10, 0, 1, 0, 0],
             [1, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 1]
         ]), constraint_values=np.array([
             [4],
             [135],
             [17],
             [17]
         ]), cost_vector=np.array([
             [1, 1, 0, 0, 0, 0]
         ])
     )}
]

for lp in problems:
    print(f'Currently working on {lp["Description"]}')
    exhaustive_search(lp=lp['Problem'])
