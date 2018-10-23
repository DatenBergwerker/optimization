import sys
sys.path.append('')

import numpy as np
from dikin import LinearProgram, dikin_interior_points


problems = [
    {'Description': 'Exercise 7.3, First Example, Beta = 0.5',
     'Problem': LinearProgram(
         constraints=np.array([
             [0, 1, 1, 0],
             [5, 4, 0, 1]
         ], dtype=np.float), constraint_values=np.array([
             [4],
             [20]
         ], np.float), cost_vector=np.array([
             [1],
             [1],
             [0],
             [0]
         ], dtype=np.float), xstart=np.array([
             [1],
             [1],
             [3],
             [11]
         ], dtype=np.float), beta=0.5, epsilon=1e-5, iterations=10000
     )},
    {'Description': 'Exercise 7.3, First Example, Beta = 1/20',
     'Problem': LinearProgram(
         constraints=np.array([
             [0, 1, 1, 0],
             [5, 4, 0, 1]
         ], dtype=np.float), constraint_values=np.array([
             [4],
             [20]
         ], np.float), cost_vector=np.array([
             [1],
             [1],
             [0],
             [0]
         ], dtype=np.float), xstart=np.array([
             [1],
             [1],
             [3],
             [11]
         ], dtype=np.float), beta=0.05, epsilon=1e-5, iterations=10000
     )},
    {'Description': 'Exercise 7.3, First Example, Beta = 1/200',
     'Problem': LinearProgram(
         constraints=np.array([
             [0, 1, 1, 0],
             [5, 4, 0, 1]
         ], dtype=np.float), constraint_values=np.array([
             [4],
             [20]
         ], np.float), cost_vector=np.array([
             [1],
             [1],
             [0],
             [0]
         ], dtype=np.float), xstart=np.array([
             [1],
             [1],
             [3],
             [11]
         ], dtype=np.float), beta=0.005, epsilon=1e-5, iterations=10000
     )},
    {'Description': 'Exercise 7.3, LP 7',
     'Problem': LinearProgram(
         constraints=np.array([
             [1, 0, 1, 0],
             [0, 1, 0, 1]
         ], dtype=np.float), constraint_values=np.array([
             [2],
             [2]
         ], np.float), cost_vector=np.array([
             [1],
             [1],
             [0],
             [0]
         ], dtype=np.float), xstart=np.array([
             [1],
             [1],
             [1],
             [1]
         ], dtype=np.float), beta=0.5, epsilon=1e-5, iterations=10000

     )},
    {'Description': 'Exercise 7.3, LP 8',
     'Problem': LinearProgram(
         constraints=np.array([
             [1, 4, 1, 0],
             [2, 1, 0, 1]
         ], dtype=np.float), constraint_values=np.array([
             [4],
             [2]
         ], np.float), cost_vector=np.array([
             [1],
             [3],
             [0],
             [0]
         ], dtype=np.float), xstart=np.array([
             [0.5],
             [0.5],
             [1.5],
             [0.5]
         ], dtype=np.float), beta=0.5, epsilon=1e-5, iterations=10000

     )}
]

for i, problem in enumerate(problems):
    print(f'Currently working on {problem["Description"]}')
    if i == 0:
        dikin_interior_points(lp=problem['Problem'])#, save_state=True)
    else:
        dikin_interior_points(lp=problem['Problem'])