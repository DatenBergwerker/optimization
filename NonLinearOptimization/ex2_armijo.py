import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# filter out warnings regarding double precision of fractions
warnings.filterwarnings("ignore", category=RuntimeWarning)


def function_exercise_d(x: np.array):
    """
    This function returns the functional value of the exercise
    :param x:
    :return:
    """
    x = x.reshape((1,))
    return 2 / 3 * x[0] ** 3 + 0.5 * x[0] ** 2


def function_exercise_d_gradient(x: np.array):
    """
    This function returns the gradient of the function in exercise d.
    :return:
    """
    x = x.reshape((1,))
    return np.array([2 * x[0] ** 2 + x[0]])


def rosenbrock(x: np.array) -> float:
    """
    This function returns the functional value of the Rosenbrock function for a given x.
    """
    x = x.reshape((2,))
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_gradient(x: np.array) -> np.array:
    """
    This function returns the gradient of the Rosenbrock for a given x.
    """
    x = x.reshape((2,))
    partial_diff_x1 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    partial_diff_x2 = 200 * (x[1] - x[0] ** 2)
    return np.array([partial_diff_x1, partial_diff_x2])


def descent_direction(gradfx: np.array) -> np.array:
    """
    This function negates the gradient vector to produce the descent direction.
    """
    return -1 * gradfx


def armijo_step_size(f, x: np.array, gradient: np.array, d: np.array,
                     gamma: float = 10e-3, beta: float = 0.5, verbose: bool = False) -> float:
    """
    This function determines the armino step size given a function, its gradients, current iterates,
    direction and stopping tresholds.
    """
    sigma = 1
    while f(x=x + sigma * d) - f(x=x) > sigma * gamma * np.dot(gradient, d):
        if verbose:
            logging.info(f(x=x + sigma * gradient) - f(x=x))
            logging.info(sigma * gamma * np.dot(gradient, d))
            logging.info(sigma)
        sigma = beta * sigma

    return sigma


def steepest_descent(gradf, f, x_start: np.array, epsilon: float = 10e-3, x_dim: int = 2,
                     maxit: int = 1000, verbose: bool = False) -> np.array:
    """
    This function tries to find the minimum value of a function until a prespecified number of iterations
    has passed or the value difference is smaller than a given epsilon treshold. It uses the method of steepest descents
    combined with armijo step size.
    """
    logging.info(f'Gradient Descent optimization initialized with parameters: \n'
                 f'Epsilon: {epsilon} \n'
                 f'Max Iterations: {maxit}')
    iteration = 0
    x = np.zeros(shape=(maxit, x_dim))
    x[iteration, :] = x_start
    stepsizes = []
    func_values = []
    directions = []

    while iteration < maxit - 1:
        if (iteration + 1) % 50 == 0 and iteration > 0:
            logging.info(f'Iteration {iteration + 1}')

        gradient = gradf(x=x[iteration, :])
        d = descent_direction(gradfx=gradient)

        if verbose:
            logging.info(np.linalg.norm(gradient))

        if np.linalg.norm(gradient) <= epsilon:
            logging.info(
                f'Change in gradient is smaller than specified epsilon of {epsilon} in interation {iteration + 1}')
            break

        sigma = armijo_step_size(x=x[iteration, :], gradient=gradient, d=d, f=f)
        stepsizes.append(sigma)
        directions.append(d)
        func_values.append(f(x[iteration]))

        if sigma == 0:
            logging.info(f'No change for gradient with Armijo step rule in iteration {iteration + 1}')
            break

        x[iteration + 1, :] = x[iteration, :] + sigma * d

        iteration += 1

    if iteration == maxit - 1:
        logging.info(f'Max iterations ({maxit}) reached')

    logging.info(f'Optimization terminated at iteration {iteration + 1}. \n'
                 f'Current function value {f(x[iteration, :])} with {x_dim} parameters {x[iteration, :]}')

    return {'iterates': x[0:iteration + 1, :], 'sigmas': stepsizes,
            'func_values': func_values, 'directions': directions}


if __name__ == '__main__':
    x_c1 = np.array([1, -0.5]).reshape((1, 2))
    x_c1_opt = steepest_descent(f=rosenbrock, gradf=rosenbrock_gradient, x_start=x_c1)
    x_c1_opt_vals = x_c1_opt['iterates']

    plt.plot(x_c1_opt_vals[:, 0], x_c1_opt_vals[:, 1])
    plt.savefig('ex2_c1.png')

    x_c2 = np.array([-1.2, 1]).reshape((1, 2))
    x_c2_opt = steepest_descent(f=rosenbrock, gradf=rosenbrock_gradient, x_start=x_c2)
    x_c2_opt_vals = x_c2_opt['iterates']

    plt.plot(x_c2_opt_vals[:, 0], x_c2_opt_vals[:, 1])
    plt.savefig('ex2_c2.png')

    x_d1 = np.array([1])
    x_d1_opt = steepest_descent(f=function_exercise_d, gradf=function_exercise_d_gradient,
                                x_start=x_d1, x_dim=1)
    print(f'''Exercise d (1): \n
              Iterates': {x_d1_opt['iterates'][0:5,:].tolist()} \n
              Search Directions: {x_d1_opt['directions'][0:5]} \n
              Step lengths: {x_d1_opt['sigmas'][0:5]} \n
              Function values: {x_d1_opt['func_values'][0:5]}''')

    x_d2 = np.array([0.5])
    x_d2_opt = steepest_descent(f=function_exercise_d, gradf=function_exercise_d_gradient,
                                x_start=x_d1, x_dim=1)
    print(f'''Exercise d (1): \n
              Iterates': {x_d2_opt['iterates'][0:5,:].tolist()} \n
              Search Directions: {x_d2_opt['directions'][0:5]} \n
              Step lengths: {x_d2_opt['sigmas'][0:5]} \n
              Function values: {x_d2_opt['func_values'][0:5]}''')
    
    x_d3 = np.array([0.1])
    x_d3_opt = steepest_descent(f=function_exercise_d, gradf=function_exercise_d_gradient,
                                x_start=x_d1, x_dim=1)
    print(f'''Exercise d (1): \n
              Iterates': {x_d3_opt['iterates'][0:5,:].tolist()} \n
              Search Directions: {x_d3_opt['directions'][0:5]} \n
              Step lengths: {x_d3_opt['sigmas'][0:5]} \n
              Function values: {x_d3_opt['func_values'][0:5]}''')
