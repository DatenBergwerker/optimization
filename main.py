import numpy as np
from algorithms import LinearProgram, exhaustive_search

lp = LinearProgram(
    constraints=np.array([
        [2, 4, 1, 1, 0],
        [4, 6, 4, 0, 1]
    ]), constraint_values=np.array([
        [8],
        [6]
    ]), cost_vector=np.array([
        [1, 1, 1, 0, 0]
    ])
)

exhaustive_search(lp)