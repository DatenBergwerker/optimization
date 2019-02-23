import numpy as np


class LinearProgram:
    """
    A utility object representing the linear program in normal form as three
    distinct matrices: constraints (A), constraint_values (b) and
    the cost vector (c).
    """

    def __init__(self, constraints: np.array, constraint_values: np.array,
                 cost_vector: np.array, base: np.array = None):
        self.constraints = constraints
        self.constraint_values = constraint_values
        self.cost_vector = cost_vector
        self.base = base
        self.tableau = None
        self.optimum_configuration = {
            'base_solution': set(),
            'optimum': float('-inf'),
        }