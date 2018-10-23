import numpy as np


class LinearProgram:
    """
    A utility object representing the linear program in normal form as three
    distinct matrices: constraints (A), constraint_values (b) and
    the cost vector (c). It also contains
    """

    def __init__(self, constraints: np.array, constraint_values: np.array,
                 cost_vector: np.array, beta: np.float, epsilon: np.float,
                 iterations: int, xstart: np.array):
        self.constraints = constraints
        self.constraint_values = constraint_values
        self.cost_vector = cost_vector
        self.beta = beta
        self.epsilon = epsilon
        self.iterations = iterations
        self.start = xstart


def terminate_algorithm(lp: LinearProgram = None, **kwargs):
    """
    Convenience function to gracefully end the algorithm execution due to either
    the realization of an unbounded lp, finding the optimal solution or reaching
    the maximum number of iterations.
    """
    params = dict(**kwargs)
    reason = params.get('reason', None)

    if reason == 'optimal':
        print(
            f'''
            Optimal solution found after {params.get('iterations')} iterations.
            Objective function value {np.asscalar(sum(params.get('cost_vector') * params.get('xvalues')))}
            Optimal x values = {params.get('xvalues')}'''
        )
    elif reason == 'unbounded':
        print(
            f'''
            Linear Problem seems to be unbounded.
            Objective function value {np.Inf}
            Optimal x values = [ ]'''
        )
    elif reason == 'maxiter':
        print(
            f'''
            Linear Problem approximation has reached max set iterations of {params.get('iterations') + 1}.
            Objective function value {np.asscalar(sum(params.get('cost_vector') * params.get('xvalues')))}
            Current x values = {params.get('xvalues')}'''
        )
        

def dikin_interior_points(lp: LinearProgram, save_state = False):
    """
    Inner points method after dikin. Input is a properly specified linear program
    with all necessary parameters.
    """

    for i in range(lp.iterations):
        if i == 0:
            current_point = [lp.start]

        D = np.zeros(shape=(current_point[i].shape[0],
                            current_point[i].shape[0]))
        np.fill_diagonal(D, current_point[i])
        A_hat = np.matmul(lp.constraints, D)
        c_hat = np.dot(D, lp.cost_vector)
        P_hat = np.identity(n=A_hat.shape[1]) - \
                np.linalg.multi_dot([A_hat.T, np.linalg.solve(np.matmul(A_hat, A_hat.T), A_hat)])
        r_prime = np.matmul(P_hat, c_hat)
        if all(r_prime == 0):
            terminate_algorithm(reason='optimal', xvalues=current_point[i], cost_vector=lp.cost_vector, iterations=i)
            # if save_state: np.savetxt('dikin_interior_points.csv', np.asarray(current_point), delimiter=',')
            break
        elif all(r_prime > 0):
            terminate_algorithm(reason='unbounded', iterations=i)
            break

        alpha = -lp.beta / min(r_prime)
        xhat_next = np.ones(shape=lp.start.shape[1]).T + alpha * r_prime
        x_next = np.dot(D, xhat_next)

        if np.linalg.norm(x_next - current_point[i]) < lp.epsilon:
            terminate_algorithm(reason='optimal', xvalues=current_point[i], cost_vector=lp.cost_vector, iterations=i)
            # if save_state: np.savetxt('dikin_interior_points.csv', np.asarray(current_point), delimiter=',')
            break

        current_point.append(x_next)

        if i == lp.iterations - 1:
            # if save_state: np.savetxt('dikin_interior_points.csv', np.asarray(current_point), delimiter=',')
            terminate_algorithm(reason='maxiter', xvalues=current_point[i], cost_vector=lp.cost_vector, iterations=i)