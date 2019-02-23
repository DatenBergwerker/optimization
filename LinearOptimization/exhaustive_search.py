# from dataclasses import dataclass
from itertools import permutations
import numpy as np

from LinearOptimization.linear_program import LinearProgram


def exhaustive_search(lp: LinearProgram):
    """
    This is the main function execution the
    exhaustive search algorithm. All distinct permutations have to fit in memory.
    :param lp: The LP in normalform.
    :return:
    """
    # reset lp
    lp.optimum_configuration = {
        'base_solution': set(),
        'optimum': float('-inf'),
    }

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
            # print(f'{base} is no base solution (not invertable).')
            continue
        else:
            # if independent solve constraints for cost coefficients
            cost_coefficients = np.linalg.solve(base, lp.constraint_values)

            if not all(cost_coefficients > 0):
                # print(f'{base} is no valid base solution (Coefficients < 0).')
                continue
            else:
                # compute inner (elementwise) product
                targetvalue = np.sum(np.inner(cost_coefficients.T, lp.cost_vector[:, baseindex]))
                base_solution = np.zeros(Blen)
                base_solution[baseindex] = cost_coefficients.T

                # if targetvalue is new current optimum
                if targetvalue >= lp.optimum_configuration['optimum']:
                    if targetvalue > lp.optimum_configuration['optimum']:
                        print(f'New optimum {targetvalue}')
                        lp.optimum_configuration['optimum'] = targetvalue
                    lp.optimum_configuration['base_solution'].add(tuple(base_solution))

    solution_space_size = len(lp.optimum_configuration['base_solution'])

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
                   ''')

        print(f'''Valid base solutions: {lp.optimum_configuration['base_solution']}''')
    else:
        print('Linear Program no has valid solution. Solution space is empty.')
