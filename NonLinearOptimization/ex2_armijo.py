import numpy as np


def f(x: np.array) -> float:
    """
    This function returns the functional value of the Rosenbrock function for a given x.
    """
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def gradf(x: np.array) -> np.array:
    """
    This function returns the gradients of the Rosenbrock for a given x.
    """
    partial_diff_x1 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] * x[0]**2)
    partial_diff_x2 = 200 * (x[1] - x[0]**2)
    return np.array([partial_diff_x1, partial_diff_x2])


def descent_direction(gradfx: np.array) -> np.array:
    """
    This function negates the gradient vector to produce the descent direction.
    """
    return -1 * gradfx


def armijo_step_size(x: np.array, gamma: float, beta: float) -> float:
    """
    This function determines the armino step size given a function, its gradients, current iterates,
    direction and stopping tresholds.
    """
    sigma = 1
    gradfx = gradf(x=x)
    d = descent_direction(gradfx=gradfx)
    while f(x=x + sigma * gradfx) - f(x=x) > sigma * gamma * gradfx * d:
        sigma = beta * sigma

    return sigma


def steepest_descent(x_start: np.array, epsilon: float = 10e-3, maxit: int = 1000):
    """
    This function tries to find the minimum value of a function until a prespecified number of iterations
    has passed or the value difference is smaller than a given epsilon treshold. It uses the method of steepest descents
    combined with armijo step size.
    """
