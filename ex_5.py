import sys
sys.path.append('')
import numpy as np
from simplex import simplex
from simplex import LinearProgram

problems = [
    {'Description': 'Exercise 5.3, First Example',
     'Problem': LinearProgram(
         constraints=np.array([
             [2, 4, 1, 1, 0],
             [3, 6, 4, 0, 1]
         ]), constraint_values=np.array([
             [8],
             [6]
         ]), cost_vector=np.array([
             [1],
             [1],
             [1],
             [0],
             [0]
         ]), base=[4, 5]
     )},
    {'Description': 'Exercise 5.3, Second Example',
     'Problem': LinearProgram(
         constraints=np.array([
             [2, 4, 1, 1, 0],
             [3, 6, 4, 0, 1]
         ]), constraint_values=np.array([
             [8],
             [6]
         ]), cost_vector=np.array([
             [1],
             [1],
             [1],
             [0],
             [0]
         ]), base=[2, 4]
     )},
    {'Description': 'Exercise 5.3, LP1',
     'Problem': LinearProgram(
         constraints=np.array([
             [-2, 1, 1, 0],
             [-1, 10, 0, 1]
         ]), constraint_values=np.array([
             [4],
             [135]
         ]), cost_vector=np.array([
             [1],
             [2],
             [0],
             [0]
         ]), base=[3, 4]
     )},
    {'Description': 'Exercise 5.3, LP2',
     'Problem': LinearProgram(
         constraints=np.array([
             [1, -1, 1, 0],
             [-1, 1, 0, 1]
         ]), constraint_values=np.array([
             [-1],
             [-1]
         ]), cost_vector=np.array([
             [0],
             [1],
             [0],
             [0]
         ]), base=[3, 4]
     )},
]

for problem in problems:
    # reduce base indices by 1 because of 0 indexing
    print(problem['Description'])
    simplex(problem['Problem'])