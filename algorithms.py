from dataclasses import dataclass
from itertools import permutations
import numpy as np


@dataclass
class LinearProgram:
    """
    A utility object representing the linear program in normal form as three
    distinct matrices: constraints (A), constraint_values (b) and
    the cost vector (c).
    """
    constraints: np.array
    constraint_values: np.array
    cost_vector: np.array
    optimum_configuration = {
        'baseindex': [],
        'optimum': float('-inf'),
    }


def exhaustive_search(lp: LinearProgram):
    """
    This is the main function execution the
    exhaustive search algorithm. All distinct permutations have to fit in memory.
    :param lp: The LP in normalform.
    :return:
    """
    # matrix row dimensions (no of constraints)
    m = lp.constraint_values.shape[0]

    # length of base set (length of cost vector)
    Blen = lp.cost_vector.shape[1]

    # permutation sets with length m of all bases
    B = set(permutations(range(Blen), m))

    for baseindex in B:
        baseindex = list(baseindex)
        base = lp.constraints[:, baseindex]

        # Check linear independence
        if not np.linalg.matrix_rank(M=base) == m:
            print(f'{base} is no base solution (not invertable).')
            break
        else:
            # if independent solve constraints for cost coefficients
            cost_coefficients = np.linalg.solve(base.T, lp.constraint_values)

            if not all(cost_coefficients > 0):
                print(f'{base} is no valid base solution (Coefficients < 0).')
                break
            else:
                targetvalue = np.sum(np.dot(cost_coefficients, lp.cost_vector[:, baseindex]))

                # if targetvalue is new current optimum
                if targetvalue > lp.optimum_configuration['optimum']:
                    lp.optimum_configuration['optimum'] = targetvalue
                    lp.optimum_configuration['baseindex'] = baseindex

                # target has the same value as current optimum
                elif targetvalue == lp.optimum_configuration['optimum']:
                    lp.optimum_configuration['baseindex'].add(baseindex)

    solution_space_size = len(lp.optimum_configuration['baseindex'])

    # solution space is not empty
    if solution_space_size:

        # more than one optimal solution
        if solution_space_size > 1:
            print(f'''
                    Linear Program has more than one optimal solution.
                    Target function value: {lp.optimum_configuration['optimum']}. 
                    No of solutions found {solution_space_size}.
                   ''')
        else:
            print(f'''
                    Linear Program exactly one optimal solution. 
                    Target function value: {lp.optimum_configuration['optimum']}.
                    No of solutions found {solution_space_size}.
                   ''')
        return {'maximum': lp.optimum_configuration['optimum'],
                'baseindex': lp.optimum_configuration['baseindex']}
    else:
        print('Linear Program no valid solution. Solution space is empty.')