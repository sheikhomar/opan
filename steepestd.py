import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
import utils as utils
from search1d import SecantSearch

from func import Func

from IPython.display import HTML, display

class SteepestDescent:
    def __init__(self, f: Func):
        self._f = f

    def _compute_alpha(self, x_k, grad_at_x_k):
        f = self._f
        alpha = sy.symbols('alpha')

        # Construct phi_k
        input_vec = x_k - alpha * grad_at_x_k
        phi_k = f.subs(input_vec)[0]

        # Find alpha_k using secant search
        line_search = SecantSearch(phi_k, alpha)
        # TODO: How do you find the minimum when secant 
        # search requires two points. We know that alpha > 0
        # so we can just find
        point1 = 0.01
        point2 = 0.02

        return line_search.find_minimum(point1, point2, verbose=False)

    def run(self, initial_point, alpha=None, epsilon=10**(-6), max_iterations=20, verbose=True):
        f = self._f
        x_k = initial_point
        
        for k in range(0, max_iterations):
            # Compute the gradient at x_k
            grad_at_x_k = f.gradient_at(x_k)

            if grad_at_x_k.norm() <= epsilon:
                if verbose:
                    print('Stopping condition reached. ||g(k)|| = {}'.format(grad_at_x_k.norm()))
                break

            # Compute alpha_k
            alpha_k = self._compute_alpha(x_k, grad_at_x_k) if alpha is None else alpha

            # Compute x(k+1)
            x_kp1 = x_k - alpha_k * grad_at_x_k

            if verbose:
                print('Iteration {0}:\n x({0})={1}\n alpha={2}\n gradient={3}\n x({4})={5}\n\n'.format(
                    k, list(x_k), alpha_k, list(grad_at_x_k), k+1, list(x_kp1)
                ))

            # Replace the current x so x(k) = x(k+1)
            x_k = x_kp1
        return x_k
