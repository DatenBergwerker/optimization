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
        self.status = None


def create_tableau(lp: LinearProgram):
    """
    Create simplex tableau from normalized linear program.
    """
    # get dimensions and construct A_base
    m, n = lp.constraints.shape

    # cut out base values from constraint matrix A and cost vector c
    cost_base = lp.cost_vector[lp.base]
    A_base = lp.constraints[:, lp.base]

    # Check conditions for valid base (full rank, no negative coefficients)
    if not np.linalg.matrix_rank(M=A_base) == m:
        print(f' {lp.base} No valid start base. Program aborted.')
        return None

    cost_coefficients = np.linalg.solve(A_base, lp.constraint_values)
    if not all(cost_coefficients > 0):
        print(f'{lp.base} is no valid base solution (Coefficients < 0).')
        return None

    # define placeholder matrix
    TB = np.zeros([m + 1, n + 1])

    # get tableau for base B
    tmp = np.linalg.solve(A_base, lp.constraints)
    TB[0:m, 0:n] = tmp
    TB[m, 0:n] = -lp.cost_vector.T + np.dot(cost_base.T, tmp)
    tmp = np.linalg.solve(A_base, lp.constraint_values)
    TB[:, n] = np.vstack((tmp, np.dot(cost_base.T, tmp))).T
    return TB


def pivot_element(lp: LinearProgram):
    """
    Find the pivot element in a given simplex tableau. If multiple coefficients yield
    the same valid values, choose the one with the lower index.
    """
    # restore original dimension values
    tab = lp.tableau
    m, n = tab.shape
    m, n = m - 1, n - 1
    pivot = {}

    # m is enough in numpy because of zero indexing
    # no more negative coefficients, return -1 because 0 is a valid index in python
    if all(tab[m, :] >= 0):
        print('No more negative coefficients. Solution seems optimal.')
        pivot.update({'pivot_row': -1, 'pivot_col': -1, 'status': 'optimal'})
        return pivot

    pivot_col = np.argmin(tab[m, :])
    pivot.update({'col': pivot_col})
    pivot_row = tab[:m, n] / tab[:m, pivot_col]

    # check conditions for pivot column coefficients
    # if the boolean sum equals m, all coefficients are either Inf or negative
    if np.array((pivot_row < 0, pivot_row == np.Inf)).sum() == m:
        print('Linear Program seems to be unbounded.')
        pivot.update({'row': -1, 'status': 'unbounded'})
    else:
        pivot_row, = np.where(pivot_row.flatten() == np.min(pivot_row[pivot_row >= 0]))
        pivot_row = np.asscalar(pivot_row)
        pivot.update({'row': pivot_row})

    return pivot


def pivot_operation(lp: LinearProgram):
    """
    This function computes the pivotization op on a given simplex tableau
    and performs the base change.
    """
    tab = lp.tableau
    m, n = tab.shape

    pivot = pivot_element(lp=lp)
    if any([pivot['row'], pivot['col']]) < 0:
        print(
            f'''No valid pivot element found. Program execution aborted.
                Reason: {pivot['status']}.
            ''')

    pivot_val = tab[pivot['row'], pivot['col']]
    tab[pivot['row'], :] = tab[pivot['row'], :] / pivot_val

    for i in range(m):
        if i != pivot['row']:
            # check how often the pivot row needs to be added / substracted from the current row
            x = tab[i, pivot['col']]
            tab[i, :] = tab[i, :] - x * tab[pivot['row'], :]

    # base change
    lp.base[pivot['row']] = pivot['col']
    lp.tableau = tab
    return lp


def tableau_analyzer(lp: LinearProgram, iteration: int):
    """
    Convenience function to report on a single simplex iteration.
    The coefficients are taken from the right hand side of the tableau (the b vector)
    and inserted at the base positions.
    """
    coefficients = np.zeros(shape=(lp.tableau.shape[1], ))
    coefficients[lp.base] = lp.tableau[, :-1]
    print(
        f'''
        Simplex iteration {iteration}
        Current base values {[val + 1 for val in lp.base]}
        Current coefficients {coefficients}
        Current target function value {lp.tableau[-1, -1]}
        Current full tableau
        {lp.tableau}
        '''
    )


def simplex(LinProg: LinearProgram):
    """
    Wrapper function for simplex algorithm.
    """
    lp = LinProg

    # reduce base by 1 because for 0 indexing
    lp.base = [val - 1 for val in lp.base]
    lp.tableau = create_tableau(lp=lp)
    if lp.tableau is None:
        return lp

    # reduce because of 0 indexing
    m, n = lp.tableau.shape
    m, n = m - 1, n - 1

    iteration = 1

    # while there are negative coefficients in the cost line of the tableau
    while any(lp.tableau[m, :] < 0):
        lp = pivot_operation(lp=lp)
        tableau_analyzer(lp=lp, iteration=iteration)
        iteration += 1

    return lp